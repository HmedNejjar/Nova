import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(1, str(ROOT))

import json
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

class ShardManifest:
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
        self.manifest = ShardManifest(split_dir)
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
    