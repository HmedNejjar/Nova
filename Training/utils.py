#type: ignore

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(1, str(ROOT))

import yaml
import json
import numpy as np
import random
from os import environ, fsync
import torch
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask, BlockMask
from torch.utils.data import DataLoader

from typing import Callable, Literal
from GPT.datasets import Phase_1_2_Dataset, Phase3Dataset

# --------------------------------------------------------------------------- #
# Config / seed
# --------------------------------------------------------------------------- #

def load_config(config_path: Path | str | None = None) -> dict:
    """
    Load the YAML configuration file.
    """
    if config_path is None:
        config_path = ROOT / "config.yaml"
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def set_seed(seed: int = 67) -> None:
    """Seed python/numpy/torch (and CUDA) for reproducible tiny runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    environ["PYTHONHASHSEED"] = str(seed)
    
# --------------------------------------------------------------------------- #
# Precision
# --------------------------------------------------------------------------- #

def get_precision(device: torch.device) -> tuple:
    """Pick mixed-precision mode for `device`.

    Returns (autocast_dtype, scaler):
      - bf16 supported   -> (torch.bfloat16, None)      no GradScaler needed
      - fp16 fallback    -> (torch.float16, GradScaler) loss scaling required
      - cpu / no AMP     -> (torch.float32,  None)
    """
    device = torch.device(device)
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported(including_emulation=False):
            return torch.bfloat16, None
        else:
            return torch.float16, torch.amp.GradScaler("cuda")
    else:
        return torch.float32, None
    
# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def _dataset_dirs(cfg: dict, phase: int, split: str) -> str:
    key = f"Phase_{phase}"
    # config paths use Windows separators; normalize for any OS
    return str(ROOT / cfg["Datasets"][key][split].replace("\\", "/"))

def set_dataloader(phase: Literal[1, 2, 3], split: Literal["train", "test"], config: dict, epoch: int = 0, skip_blocks: int = 0) -> DataLoader:
    """
    Set up the DataLoader for the specified phase and split.

    Args:
        phase: The training phase (1, 2, or 3).
        split: The dataset split, matching the config keys ("train" or "test"). Only "train" is shuffled.
        config: Configuration dictionary.
        epoch: Epoch number; seeds the shuffle order.
        skip_blocks: Number of blocks to skip per worker when resuming.

    Returns:
        DataLoader instance for the specified dataset.
    """
    training_cfg = config["Train"]
    ds_dir = _dataset_dirs(config, phase, split)
    seed = int(training_cfg.get("seed", 67))
    
    is_train_split: bool = (split == "train")
    
    if phase in (1, 2):
        dataset = Phase_1_2_Dataset(ds_dir, shuffle= is_train_split, seed= seed, epoch= epoch, skip_blocks= skip_blocks)
        collate_fn = dataset.collate
    
    elif phase == 3:
        dataset = Phase3Dataset(ds_dir, shuffle= is_train_split, seed= seed, epoch= epoch, skip_blocks= skip_blocks)
        collate_fn = dataset.collate
        
    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 1, 2, or 3.")
    
    micro_batch = int(training_cfg[f"Phase_{phase}"]["micro_batch"])
    num_workers = int(training_cfg.get("num_workers", 2))
    
    return DataLoader(dataset, batch_size= micro_batch, num_workers= num_workers, collate_fn= collate_fn, pin_memory= (torch.cuda.is_available() and is_train_split), persistent_workers= (num_workers > 0 and is_train_split), drop_last= is_train_split)
    
def to_device(batch: dict, device: torch.device) -> dict:
    """Move tensor values to `device`; pass ragged list values through as-is.

    Phase 1/2 batches carry `boundaries` and Phase 3 batches carry
    `case_labels` / `conv_bounds` as per-sample lists — they can't be stacked
    and aren't needed on the GPU.
    """
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out

def infinite_loader(make_dl: Callable[[int, int], DataLoader], epoch: int = 0, skip_blocks: int = 0):
    """Yield (batch, epoch) forever, starting a new pass when the data runs out.

    IterableDatasets have no length, and we train by step count, so the stream
    must be endless. `make_dl(epoch, skip_blocks)` builds the DataLoader for one
    pass; a NEW DataLoader is built for every pass because with
    persistent_workers each worker holds its own copy of the dataset, so calling
    set_epoch() on the main-process copy would never reach them (every pass
    would repeat epoch 0's order, and skip_blocks would never be cleared).

    `skip_blocks` applies only to the first pass (resuming mid-epoch).
    """
    empty_passes = 0
    while True:
        dl = make_dl(epoch, skip_blocks)
        n = 0
        for batch in dl:
            n += 1
            yield batch, epoch
        del dl  # shuts the pass's workers down before the next pass starts its own
        # One empty pass is legitimate (a resume can skip the rest of an epoch);
        # two in a row means the dataset itself is empty, which would loop forever
        empty_passes = empty_passes + 1 if n == 0 else 0
        if empty_passes >= 2:
            raise RuntimeError("dataloader produced 0 batches for two consecutive epochs — "
                               "check dataset path / skip_blocks")
        epoch += 1
        skip_blocks = 0
        
# --------------------------------------------------------------------------- #
# Attention mask
# --------------------------------------------------------------------------- #

def document_block_mask(doc_ids: Tensor) -> BlockMask:
    """
    Create a block mask for flex attention based on document IDs.

    Args:
        doc_ids: A tensor of shape (batch_size, seq_len) containing document IDs.
    Returns:
        A BlockMask object that can be used with flex_attention.
    """
    batch_size, seq_len = doc_ids.shape
    
    def mask_mod(b, h, q_idx, kv_idx):
        # Determine if the query and key-value indices belong to the same document and key not in the future
        return (doc_ids[b, q_idx] == doc_ids[b, kv_idx]) & (q_idx >= kv_idx)
    
    return create_block_mask(mask_mod, batch_size, None, seq_len, seq_len, device= doc_ids.device)
            
# --------------------------------------------------------------------------- #
# JSONL metric logs
# --------------------------------------------------------------------------- #

def append_jsonl(path: str | Path, data: dict) -> None:
     """Append one JSON record as a line, flushed immediately (crash-safe-ish)."""
     path = Path(path)
     path.parent.mkdir(parents= True, exist_ok= True)
     
     with open(path, "a", encoding= "utf-8") as f:
         f.write(json.dumps(data) + "\n")
         f.flush()
         fsync(f.fileno())
         
def truncate_jsonl(path: str | Path, max_step: int) -> int:
    """Drop records whose "step" is greater than `max_step` (used on resume, so
    steps logged after the last checkpoint aren't duplicated). Keeps everything
    up to and including `max_step`. Returns the number of lines removed."""
    path = Path(path)
    if not path.exists():
        return 0
    
    kept, dropped = [], 0
    with open(path, "r", encoding= "utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            
            if isinstance(data, dict) and "step" in data and data["step"] > max_step:
                dropped += 1
            else:
                kept.append(data)
        
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")
    tmp.replace(path)
    return dropped
         
        