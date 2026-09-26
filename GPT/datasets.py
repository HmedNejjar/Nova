import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(1, str(ROOT))

import json
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

class Phase_1_2_ShardManifest:
    def __init__(self, split_dir: str | Path) -> None:
        self.split_dir = Path(split_dir)
        with open(self.split_dir / "manifest.json", "r") as f:
            self.data = json.load(f)
            
        self.seq_len = self.data["seq_len"]
        self.shards = self.data["shards"] # list of {shard, num_block, seq_len...}
        # Phase 2 blocks are <pad>-filled after the last doc; Phase 1 blocks have no padding
        self.pad_id = self.data.get("pad_id")
        
    def shard_path(self, prefix: str) ->  tuple:
        return (self.split_dir / f"{prefix}_tokens.npy",
                self.split_dir / f"{prefix}_bpos.npy",
                self.split_dir / f"{prefix}_bptr.npy")
        
class Phase_1_2_Dataset(IterableDataset):
    """
    Yields dicts: {"input_ids": LongTensor(seq_len,), "boundaries": LongTensor(k,)}
    "boundaries" are local doc-end offsets within the block (0..seq_len),
    """
    def __init__(self, split_dir: str | Path, shuffle: bool = True, shuffle_buffer_shards: int = 1, seed: int = 21, epoch: int = 0) -> None:
        super().__init__()
        self.manifest = Phase_1_2_ShardManifest(split_dir)
        self.seq_len  = self.manifest.seq_len
        self.pad_id = self.manifest.pad_id
        self.shuffle = shuffle
        
        # How many shards to load before flattening and shuffling their blocks.
        # 1 means shuffle only within each shard.
        # >1 means blocks from multiple shards are mixed together before yielding.
        self.shuffle_buffer_shards = shuffle_buffer_shards
        
        self.seed = seed
        self.epoch = epoch
        
    def set_epoch(self, epoch: int) -> None:
        """
        Changes the RNG seed used in __iter__ so each epoch gets a different
        shuffle order.
        """
        self.epoch = epoch
        
    def _worker_shard_list(self) -> list:
        """
        Decides which shards this DataLoader worker should read.

        Intended behavior:
          1. Optionally shuffle the full shard list.
          2. If running inside a DataLoader worker, take every `num_workers`-th
             shard: worker 0 gets indices 0, N, 2N, ...; worker 1 gets
             1, N+1, 2N+1, ...; etc.
        """
        
        shards = list(self.manifest.shards)
        if self.shuffle:
            # Shared shuffle seed, so all workers see the same
            # shuffled order, then each worker takes its disjoint slice.
            shard_rng = np.random.default_rng(self.seed + self.epoch * 1000)
            shard_rng.shuffle(shards)
        info = get_worker_info()
        if info is not None:
            # Each worker takes every num_workers-th shard, starting at info.id.
            shards = shards[info.id::info.num_workers]
        return shards
    
    def _load_shard(self, entry: dict) -> tuple:
        shard_path, bpos_path, bptr_path = self.manifest.shard_path(entry["shard"])
        
        tokens = np.load(shard_path) # (num_blocks, seq_len)
        bpos = np.load(bpos_path)
        bptr = np.load(bptr_path) # (num_blocks+1,)
        
        return (tokens, bpos, bptr)
    
    def _shard_block(self, tokens: np.ndarray, bpos: np.ndarray, bptr: np.ndarray, rng: np.random.Generator) -> list:
        num_blocks = tokens.shape[0]
        order = rng.permutation(num_blocks) if self.shuffle else np.arange(num_blocks)
        
        items = []
        for i in order:
            lo, hi = bptr[i], bptr[i + 1]
            bounds = bpos[lo:hi]
            if len(bounds) == 0 or bounds[-1] != self.seq_len:
                bounds = np.append(bounds, self.seq_len)
            items.append((tokens[i], bounds))
        return items
    
    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        block_rng = np.random.default_rng(self.seed + self.epoch * 1000 + worker_id)

        shards = self._worker_shard_list()
        buf = []
        for entry in shards:
            tokens, bpos, bptr = self._load_shard(entry)
            buf.append(self._shard_block(tokens, bpos, bptr, block_rng))
            if len(buf) < self.shuffle_buffer_shards:
                continue
            yield from self._drain(buf, block_rng)
            buf = []
        if buf:
            yield from self._drain(buf, block_rng)
 
    def _drain(self, shard_blocks_list: list, rng: np.random.Generator):
        """
        Flattens buffered shards into one pool of blocks, optionally shuffles
        the pool, and yields individual samples.
        """
        
        pool = [item for shard_items in shard_blocks_list for item in shard_items]
        if self.shuffle:
            rng.shuffle(pool)
        for block, bounds in pool:
            yield {
                "input_ids": torch.from_numpy(block.astype(np.int64)),
                "boundaries": torch.from_numpy(bounds.astype(np.int64)),
            }
    
    def collate(self, batch: list) -> dict:
        """
        Collates a batch of packed blocks into tensors for causal LM training.

        Each block may contain multiple documents concatenated together.
        This method:
          1. Shifts input_ids to create next-token prediction labels.
          2. Resets position_ids to 0 at each document boundary so RoPE
             encodes intra-document positions only.
          3. Builds a block-diagonal causal attention mask so tokens can
             attend only to earlier tokens within the same document.
          4. Masks out the label at each interior document boundary (-100)
             to prevent the model from learning cross-document transitions.
          5. Masks out the label on padding tokens (Phase 2 blocks are
             <pad>-filled after their last document).
        """
        input_ids = torch.stack([b["input_ids"] for b in batch])  # (B, seq_len)
        batch_size, seq_len = input_ids.shape
        
        # Initialize labels with -100 
        labels = torch.full_like(input_ids, fill_value=-100)
        # Shift input_ids by 1 to the left for next-token prediction targets
        labels[:, :-1] = input_ids[:, 1:]
        
        # Create a 1D tensor of absolute sequence positions [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, device=input_ids.device)
        # Create a standard lower-triangular causal mask (True = allowed to attend)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device))
        # Initialize tensors for document-local position IDs and attention masks
        position_ids = torch.empty_like(input_ids)
        attn_mask = torch.empty((batch_size, seq_len, seq_len), dtype=torch.bool, device=input_ids.device)
        
        for i, sample in enumerate(batch):
            # Get document boundary offsets for the current sample
            bounds = sample["boundaries"].to(device=input_ids.device, dtype=torch.long)
            # Assign a document ID to each absolute position based on the boundaries.
            doc_ids = torch.searchsorted(bounds, positions, right=True)
            
            # Get the starting absolute position of each document (prepends 0 for the first doc)
            starts = torch.cat((bounds.new_zeros(1), bounds[:-1]))
            
            # Calculate local position IDs by subtracting the document's start offset.
            # This resets the position ID to 0 at the beginning of each document.
            position_ids[i] = positions - starts[doc_ids]
            
            # Create a block-diagonal causal attention mask:
            # 1. (doc_ids[:, None] == doc_ids[None, :]) ensures tokens only attend to the SAME document.
            # 2. & causal_mask ensures tokens only attend to PREVIOUS tokens (causal ordering).
            attn_mask[i] = (doc_ids[:, None] == doc_ids[None, :]) & causal_mask
            
            # Find boundaries that are strictly inside the sequence (excluding 0 and seq_len)
            interior_bounds = bounds[(bounds > 0) & (bounds < seq_len)]
            
            # Mask out the loss for the token immediately preceding each interior document boundary.
            # This prevents the model from learning to predict the first token of Document B 
            # from the last token of Document A, effectively enforcing document separation.
            labels[i, interior_bounds - 1] = -100
        
        if self.pad_id is not None:
            # Padding is neither an input to learn from nor a target to predict
            labels[input_ids == self.pad_id] = -100
            labels[labels == self.pad_id] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attn_mask": attn_mask,
        }

class Phase_3_ShardManifest:
    def __init__(self, split_dir: str | Path) -> None:
        self.split_dir = Path(split_dir)
        with open(self.split_dir / "manifest.json") as f:
            self.data = json.load(f)
        self.seq_len = self.data["seq_len"]
        self.shards = self.data["shards"]
 
    def shard_paths(self, prefix: str) -> tuple:
        return (self.split_dir / f"{prefix}_tokens.npy",
                self.split_dir / f"{prefix}_convs.npy",
                self.split_dir / f"{prefix}_block_conv_ptr.npy",
                self.split_dir / f"{prefix}_spans.npy")


class Phase3Dataset(IterableDataset):
    """
    Yields dicts:
      input_ids:    LongTensor(seq_len,)
      labels:       LongTensor(seq_len,)              -100 outside assistant spans
      conv_bounds:  LongTensor(num_convs_in_block, 2)  local (start, end) per conversation
      case_labels:  LongTensor(num_convs_in_block,)    -1/0/1/2, aligned with conv_bounds
    """
 
    def __init__(self, split_dir: str | Path, shuffle: bool = True,
                 shuffle_buffer_shards: int = 1, seed: int = 21, epoch: int = 0) -> None:
        super().__init__()
        self.manifest = Phase_3_ShardManifest(split_dir)
        self.seq_len = self.manifest.seq_len
        self.shuffle = shuffle
        self.shuffle_buffer_shards = shuffle_buffer_shards
        self.seed = seed
        self.epoch = epoch
 
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
 
    def _worker_shard_list(self) -> list:
        """Shuffle once with a seed shared across workers, THEN slice by
        worker — see module docstring for why order matters here."""
        shards = list(self.manifest.shards)
        if self.shuffle:
            shard_rng = np.random.default_rng(self.seed + self.epoch * 1000)
            shard_rng.shuffle(shards)
        info = get_worker_info()
        if info is not None:
            shards = shards[info.id::info.num_workers]
        return shards
 
    def _load_shard(self, entry: dict) -> tuple:
        tok_path, convs_path, bptr_path, spans_path = self.manifest.shard_paths(entry["shard"])
        tokens = np.load(tok_path)           # (num_blocks, seq_len) uint32
        convs = np.load(convs_path)          # (num_convs, 5) int64
        block_conv_ptr = np.load(bptr_path)  # (num_blocks+1,) int32
        spans = np.load(spans_path)          # (num_spans, 2) int32
        return tokens, convs, block_conv_ptr, spans
 
    def _block_items(self, tokens: np.ndarray, convs: np.ndarray, block_conv_ptr: np.ndarray, spans: np.ndarray, rng: np.random.Generator) -> list:
        n_blocks = tokens.shape[0]
        order = rng.permutation(n_blocks) if self.shuffle else np.arange(n_blocks)
 
        items = []
        for i in order:
            lo, hi = block_conv_ptr[i], block_conv_ptr[i + 1]
            block_convs = convs[lo:hi]           # (k, 5): start,end,span_off,span_cnt,case
            block_start = i * self.seq_len
 
            trainable = np.zeros(self.seq_len, dtype=bool)
            conv_bounds, case_labels = [], []
            for start, end, span_off, span_cnt, case in block_convs:
                conv_bounds.append((int(start - block_start), int(end - block_start)))
                case_labels.append(int(case))
                for s, e in spans[span_off:span_off + span_cnt]:
                    trainable[s - block_start:e - block_start] = True
 
            items.append((tokens[i], trainable, conv_bounds, case_labels))
        return items
 
    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        # block-order shuffle CAN be worker-dependent — it doesn't affect
        # which shards a worker sees, only the order it yields its own blocks
        block_rng = np.random.default_rng(self.seed + self.epoch * 1000 + worker_id + 1)
 
        shards = self._worker_shard_list()
        buf = []
        for entry in shards:
            tokens, convs, block_conv_ptr, spans = self._load_shard(entry)
            buf.append(self._block_items(tokens, convs, block_conv_ptr, spans, block_rng))
            if len(buf) < self.shuffle_buffer_shards:
                continue
            yield from self._drain(buf, block_rng)
            buf = []
        if buf:
            yield from self._drain(buf, block_rng)
 
    def _drain(self, shard_items_list: list, rng: np.random.Generator):
        pool = [item for shard_items in shard_items_list for item in shard_items]
        if self.shuffle:
            rng.shuffle(pool)
        for block, trainable, conv_bounds, case_labels in pool:
            tokens = block.astype(np.int64)
            labels = np.full(self.seq_len, -100, dtype=np.int64)
            # standard next-token shift, but only where the TARGET token is trainable
            labels[:-1] = np.where(trainable[1:], tokens[1:], -100)
            yield {
                "input_ids": torch.from_numpy(tokens),
                "labels": torch.from_numpy(labels),
                "conv_bounds": torch.tensor(conv_bounds, dtype=torch.long),
                "case_labels": torch.tensor(case_labels, dtype=torch.long),
            }
 
 
def collate(batch: list) -> dict:
    """
    Builds the block-diagonal causal attention mask and per-conversation
    RoPE position_ids from each sample's conv_bounds. Padding positions
    (outside every conv's range) get self-attention only — otherwise an
    all-False mask row produces NaN in softmax; the -100 label already
    keeps them out of the loss, this just keeps the forward pass clean.
 
    case_labels is left as a per-sample list (ragged — conversation count
    varies per block), for slicing eval loss by RAG case later.
    """
    seq_len = batch[0]["input_ids"].shape[0]
    B = len(batch)
 
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
 
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    attn_mask = torch.eye(seq_len, dtype=torch.bool).unsqueeze(0).repeat(B, 1, 1)
    position_ids = torch.zeros(B, seq_len, dtype=torch.long)
 
    for i, b in enumerate(batch):
        for start, end in b["conv_bounds"].tolist():
            length = end - start
            position_ids[i, start:end] = torch.arange(length)
            attn_mask[i, start:end, start:end] |= causal[:length, :length]
 
    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "attn_mask": attn_mask,  # (B, seq_len, seq_len) bool, True = attend
        "case_labels": [b["case_labels"] for b in batch],
    }