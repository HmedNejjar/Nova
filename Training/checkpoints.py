#type: ignore

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(1, str(ROOT))

import json
import random
import shutil
import numpy as np
import torch, torch.nn as nn
from safetensors.torch import save_model, load_model

# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #

def _capture_rng() -> dict:
    """Capture the current RNG state for reproducibility."""
    rng_state = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state()
                }
    
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    
    return rng_state

def _restore_rng(rng: dict) -> None:
    """Restore the RNG state from a saved state."""
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if "torch_cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["torch_cuda"]])
        
def remove_old_checkpoints(dir: str | Path, keep_last: int) -> list[Path]:
    """Delete all but the newest `keep_last` `step_*` dirs. Returns survivors."""
    dir = Path(dir)
    steps = sorted(d for d in dir.glob("step_*") if d.is_dir())
    keep = int(keep_last)
    for d in steps[:max(0, len(steps) - keep)]:      # oldest first — delete these
        shutil.rmtree(d, ignore_errors=True)
    return steps[max(0, len(steps) - keep):]
        
def checkpoint_save(dir: str | Path, step: int, model: nn.Module, opt: torch.optim.Optimizer, sched: torch.optim.lr_scheduler.LambdaLR, scaler: torch.amp.GradScaler | None, meta: dict | None = None, keep_last: int = 2) -> Path:
    """
    Save the model, optimizer, scheduler, and scaler states to a checkpoint file.

    Args:
        dir (str | Path): Directory to save the checkpoint.
        step (int): Current training step.
        model (nn.Module): The model to save.
        opt (torch.optim.Optimizer): The optimizer to save.
        sched (torch.optim.lr_scheduler.LambdaLR): The scheduler to save.
        scaler (int): The scaler state to save.
        meta (dict | None): Additional metadata to save.
        keep_last (int): Number of last checkpoints to keep.

    Returns:
        Path: The path to the saved checkpoint file.
    """
    dir = Path(dir)
    checkpoint = dir / f"step_{step:07d}"
    tmp  = dir / f".tmp_step_{step:07d}"  
    tmp.mkdir(parents= True, exist_ok= True)
    
    # Save model state
    save_model(model= model, filename= str(tmp / "model.safetensors"))
    
    state = {
            "step": step,
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": _capture_rng(),
            # caller data (epoch, samples_seen, tokens_seen, phase, ...) lives in its
            # own namespace so it can never overwrite the keys above
            "meta": dict(meta or {}),
            }

    torch.save(state, tmp / "state.pt")
    
    # Atomically move the temporary checkpoint to the final location
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    tmp.rename(checkpoint)
    
    # Remove older checkpoints if exceeding keep_last
    remove_old_checkpoints(dir, keep_last)
    
    return checkpoint

def copy_checkpoint(src: str | Path, dst: str | Path, info: dict | None = None) -> Path:
    """Refresh `dst` (best/ or final/) with the weights of checkpoint `src`.

    Only model.safetensors is copied: best/ and final/ are used for
    --init-from and inference, never to resume, so the optimizer state
    (2x the weights in fp32) would be dead weight. `info` (e.g. step,
    eval loss) is written alongside as info.json.
    """
    src, dst = Path(src), Path(dst)
    tmp = dst.with_name(f".tmp_{dst.name}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    shutil.copy2(src / "model.safetensors", tmp / "model.safetensors")
    with open(tmp / "info.json", "w", encoding="utf-8") as f:
        json.dump({"source": src.name, **(info or {})}, f, indent=2)
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    return dst


def save_weights(dst: str | Path, model: nn.Module, info: dict | None = None) -> Path:
    """Write only the model weights (+ info.json) to `dst`, e.g. best/.

    Used when eval finds a new best loss on a step that isn't a checkpoint
    step: a full checkpoint would also write the optimizer state (~2x the
    weights), which best/ never needs. Written to a temp dir and renamed, so a
    crash never leaves a half-written `dst`.
    """
    dst = Path(dst)
    tmp = dst.with_name(f".tmp_{dst.name}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_model(model=model, filename=str(tmp / "model.safetensors"))
    with open(tmp / "info.json", "w", encoding="utf-8") as f:
        json.dump(info or {}, f, indent=2)
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    return dst


def latest_checkpoint(dir: str | Path) -> Path | None:
    """Newest `step_*` dir in `dir`, or None. `--resume <dir>` accepts the phase dir itself."""
    dir = Path(dir)
    steps = sorted(d for d in dir.glob("step_*") if d.is_dir())
    return steps[-1] if steps else None

# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

def checkpoint_load(dir: str | Path, model: nn.Module, opt: torch.optim.Optimizer, sched: torch.optim.lr_scheduler.LambdaLR, scaler: torch.amp.GradScaler | None, resume: bool = True, device: torch.device | str = "cpu") -> dict:
    """
    Load the model, optimizer, scheduler, and scaler states from a checkpoint file.

    Args:
        dir (str | Path): Directory to load the checkpoint from.
        model (nn.Module): The model to load.
        opt (torch.optim.Optimizer): The optimizer to load.
        sched (torch.optim.lr_scheduler.LambdaLR): The scheduler to load.
        scaler (int): The scaler state to load.
        resume (bool): Whether to resume training from the checkpoint.
        device: Device to map the loaded tensors.

    Returns:
        dict: {"step", "meta", "path", ...}; caller data saved via `meta` is under "meta".
    """
    dir = Path(dir)
    if dir.is_dir() and not (dir / "model.safetensors").exists():
        nxt = latest_checkpoint(dir)
        if nxt is None:
            raise FileNotFoundError(f"no model.safetensors in {dir} and no step_* inside it")
        dir = nxt
        
    # Load model state
    load_model(model= model, filename= str(dir / "model.safetensors"), strict= True, device= str(device))
    
    state_path = dir / "state.pt"
    if not state_path.exists():
        if resume:
            raise FileNotFoundError(f"{dir} has no state.pt — cannot full-resume "
                                    "(use resume=False for weights-only init)")
        # best/ and final/ hold weights + info.json only
        info_path = dir / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        return {"step": info.get("step", 0), "meta": info, "path": str(dir)}
    
    # Load onto the CPU: RNG states must stay CPU ByteTensors, and
    # opt.load_state_dict() moves optimizer tensors to each param's device itself
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    state["path"] = str(dir)

    if resume:
        if opt is None:
            raise ValueError("load_checkpoint(resume=True) requires an optimizer")
        if state.get("opt") is not None:
            opt.load_state_dict(state["opt"])
        if sched is not None and state.get("sched") is not None:
            sched.load_state_dict(state["sched"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        elif scaler is None and state.get("scaler") is not None:
            print("[checkpoint] warning: checkpoint saved with a GradScaler but "
                  "current precision has none (bf16/fp32) — scaler state dropped")
        if state.get("rng"):
            _restore_rng(state["rng"])

    return state
