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
        
    def _worker_shard_list(self, rng: np.random.Generator) -> list:
        """
        Decides which shards this DataLoader worker should read.

        Intended behavior:
          1. Optionally shuffle the full shard list.
          2. If running inside a DataLoader worker, take every `num_workers`-th
             shard: worker 0 gets indices 0, N, 2N, ...; worker 1 gets
             1, N+1, 2N+1, ...; etc.

        If `shuffle=True`, each worker calls this with a *different* RNG because
        `__iter__` seeds the RNG with `worker_id`. Therefore each worker shuffles
        the full shard list differently before slicing. That means the workers
        are not guaranteed to receive disjoint sets of shards; some shards may be
        duplicated and others skipped. If strict disjoint coverage is required,
        shuffle once with a shared seed before slicing, or slice first and then
        shuffle only within the worker's assigned shards.
        """
        
        shards = list(self.manifest.shards)
        if self.shuffle:
            rng.shuffle(shards)
        info = get_worker_info()
        if info is not None:
            # Split shards across workers.
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
        rng = np.random.default_rng(self.seed + self.epoch * 1000 + worker_id)
 
        shards = self._worker_shard_list(rng)
        buf = []
        for entry in shards:
            tokens, bpos, bptr = self._load_shard(entry)
            buf.append(self._shard_block(tokens, bpos, bptr, rng))
            if len(buf) < self.shuffle_buffer_shards:
                continue
            yield from self._drain(buf, rng)
            buf = []
        if buf:
            yield from self._drain(buf, rng)
 
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
        Turns a list of samples into a training batch.

        Input samples contain both `input_ids` and `boundaries`, but this
        collate function currently returns only `input_ids` and `labels`.
        The `boundaries` are therefore discarded here.

        Label creation is standard causal-LM next-token prediction:
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1]  = -100  (ignored by CrossEntropyLoss)
        """
        input_ids = torch.stack([b["input_ids"] for b in batch])  # (B, seq_len)
        labels = torch.full_like(input_ids, fill_value=-100)
        labels[:, :-1] = input_ids[:, 1:]
        return {
            "input_ids": input_ids,
            "labels": labels,
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