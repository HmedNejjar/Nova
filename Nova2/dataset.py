from pathlib import Path
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset

class TokenShardDataset(Dataset):
    """Memory-mapped contiguous uint32 token shards for causal LM training."""
    def __init__(self, directory: str | Path, seq_len: int, stride: int | None = None) -> None:
        self.directory = Path(directory); self.seq_len = seq_len; self.stride = stride or seq_len
        self.shards = sorted(self.directory.glob("*.bin"))
        if not self.shards: raise FileNotFoundError(f"No .bin token shards found in {self.directory}")
        self.maps, self.counts = [], []
        for path in self.shards:
            count = path.stat().st_size // np.dtype(np.uint32).itemsize
            if count >= seq_len + 1:
                self.maps.append(np.memmap(path, dtype=np.uint32, mode="r")); self.counts.append(int(count))
        if not self.maps: raise ValueError(f"No shard contains at least {seq_len + 1} tokens")
        self.cumulative = []; total = 0
        for count in self.counts:
            total += 1 + (count - seq_len - 1) // self.stride; self.cumulative.append(total)
        self.total_samples = total
    def __len__(self): return self.total_samples
    def __getitem__(self, index: int):
        if index < 0: index += self.total_samples
        if index < 0 or index >= self.total_samples: raise IndexError(index)
        shard = next(i for i, end in enumerate(self.cumulative) if index < end)
        previous = 0 if shard == 0 else self.cumulative[shard - 1]
        start = (index - previous) * self.stride; tokens = self.maps[shard]
        x = torch.from_numpy(np.asarray(tokens[start:start+self.seq_len], dtype=np.int64).copy())
        y = torch.from_numpy(np.asarray(tokens[start+1:start+self.seq_len+1], dtype=np.int64).copy())
        return x, y

def write_token_shards(token_ids, output_dir: str | Path, tokens_per_shard: int = 10_000_000) -> None:
    """Write token IDs as little-endian uint32 shards with metadata."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(token_ids, dtype=np.uint32)
    if ids.ndim != 1: raise ValueError("token_ids must be one-dimensional")
    for p in output_dir.glob("shard-*.bin"): p.unlink()
    for p in output_dir.glob("shard-*.json"): p.unlink()
    for i in range(math.ceil(len(ids) / tokens_per_shard)):
        start, end = i * tokens_per_shard, min(len(ids), (i + 1) * tokens_per_shard)
        ids[start:end].tofile(output_dir / f"shard-{i:05d}.bin")
        (output_dir / f"shard-{i:05d}.json").write_text(json.dumps({
            "format": "nova2-token-shard-v1", "dtype": "uint32", "token_count": end-start
        }, indent=2), encoding="utf-8")
