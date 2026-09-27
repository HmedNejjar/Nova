#type: ignore

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(1, str(ROOT))

import argparse
import importlib.util
import torch, torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from typing import Generator
from dataclasses import dataclass, asdict, fields

from GPT.Nova import NovaLM
from Training.utils import load_config, set_seed, get_precision, set_dataloader, infinite_loader, to_device, truncate_jsonl, append_jsonl
from Training.optimizer import build_optimizer, build_scheduler
from Training.checkpoints import checkpoint_save, checkpoint_load, save_weights, copy_checkpoint
from Training.eval import chunked_lm_loss, evaluate, sample_generations, DEFAULT_PROMPTS
from Training.metrics import MetricTracker, plot_metrics

@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    blocks_seen: int = 0
    best_eval: float = float("inf")
    last_checkpt: Path | None = None
    
    def to_meta(self, phase: int) -> dict:
        meta = asdict(self)
        meta.pop("last_checkpt")
        meta["phase"] = phase
        return meta
 
    @classmethod
    def from_checkpoint(cls, checkkpt: dict) -> "TrainState":
        meta = checkkpt["meta"]
        saved = {f.name for f in fields(cls)} - {"step", "last_checkpt"}
        return cls(step=checkkpt["step"], last_checkpt=Path(checkkpt["path"]),
                   **{k: meta[k] for k in saved if k in meta})


@dataclass(frozen=True)
class TrainConfig:
    PHASE: int
    MAX_STEPS: int
    GRAD_ACCUM: int
    NUM_WORKERS: int
    GRAD_CLIP: float
    LOG_EVERY: int
    EVAL_EVERY: int
    EVAL_BATCHES: int
    CHECKPT_EVERY: int
    KEEP_LAST: int
    CHECKPT_DIR: Path
    METRICS_DIR: Path
    DEVICE: torch.device
    AMP_DTYPE: torch.dtype
    
    @property
    def use_amp(self) -> bool:
        return self.DEVICE.type == "cuda" and self.AMP_DTYPE != torch.float32
 
    @property
    def train_log(self) -> Path:
        return self.METRICS_DIR / "train.jsonl"
 
    @property
    def eval_log(self) -> Path:
        return self.METRICS_DIR / "eval.jsonl"
 

# Parse arguments
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=(1, 2, 3), required=True, help= "Phase to train (1, 2, 3).")
    ap.add_argument("--resume", type=str, default=None, help= "Path to resume training from last checkpoint.")
    ap.add_argument("--init-from", type=str, default=None, help= "Path to checkpoint to initialize training from.")
    args = ap.parse_args()
    
    if args.resume and args.init_from:
        ap.error("--resume and --init-from are mutually exclusive")
    return args

"""
---------------------------------------------------------------------------
Train Step
---------------------------------------------------------------------------
"""

def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR, scaler: torch.amp.GradScaler, stream: Generator, train_state: TrainState, config: TrainConfig, want_acc: bool) -> dict | None:
    """One optimizer step over GRAD_ACCUM micro-batches. Returns stats, or None if the window had no targets."""
    
    # 1. Fetch the whole window first (CPU, pinned) to know the total target count
    window = []
    for _ in range(config.GRAD_ACCUM):
        batch, new_epoch = next(stream)
        
        if new_epoch != train_state.epoch:
            train_state.epoch, train_state.blocks_seen = new_epoch, 0
            
        train_state.blocks_seen += batch["input_ids"].size(0)
        window.append(batch)
        
    n_total = sum(int((batch["labels"] != -100).sum()) for batch in window)
    if n_total == 0:
        scheduler.step()
        return None
    
    # 2. Forward/backward, token-weighted over the whole window
    
        # Initialize
    optimizer.zero_grad()
    loss_total  = torch.zeros((), device= config.DEVICE)
    correct = acc_n = input_tokens = 0
    
        # Forward pass
    for batch in window:
        batch = to_device(batch, config.DEVICE)
        
        with torch.autocast(device_type= config.DEVICE.type, dtype= config.AMP_DTYPE, enabled= config.use_amp):
            pred, _ = model(batch["input_ids"], None, batch["position_ids"], batch["attn_mask"], return_hidden= True)
            
            loss_sum, corct, acc = chunked_lm_loss(model, pred, batch["labels"], want_acc= want_acc) 
            
        if loss_sum.requires_grad:
            loss = loss_sum / n_total
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
        loss_total += loss_sum.detach()
        correct, acc_n = correct + corct, acc_n + acc
        input_tokens += batch["input_ids"].numel()
        
        del pred, loss_sum, batch    
        
    # 3. unscale -> clip -> step -> update -> schedule
    if scaler is not None:
        scaler.unscale_(optimizer)
    grad_norm = clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
    
    if scaler is not None:
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        stepped = scaler.get_scale() >= scale_before      # scale dropped -> step was skipped
        
    elif torch.isfinite(grad_norm):
        optimizer.step()
        stepped = True
    else:
        stepped = False
    scheduler.step()
    
    return {"loss_sum": loss_total, "n_target": n_total, "correct": correct, "acc_n": acc_n, "input_tokens": input_tokens, "grad_norm": grad_norm, "stepped": stepped}

def log_step(tracker: MetricTracker, scheduler, scaler, state: TrainState, config: TrainConfig, grad_norm: torch.Tensor, nonfinite: int, pbar: tqdm) -> None:
    m = tracker.compute()
    rec = {"step": state.step, "loss": m["loss"], "acc": m["acc"], "lr": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm),
           "tok_per_s": m["throughput"], "epoch": state.epoch, "nonfinite_steps": nonfinite}
    
    if scaler is not None:
        rec["loss_scale"] = scaler.get_scale()
    append_jsonl(config.train_log, rec)
    pbar.set_postfix(loss=f"{m['loss']:.3f}", lr=f"{rec['lr']:.2e}", gn=f"{rec['grad_norm']:.2f}")
    tracker.reset()

def run_eval(model: nn.Module, config: dict, state: TrainState, train_cfg: TrainConfig) -> None:
    # Fresh test loader every time so each eval scores the same batches (comparable)
    test_dl = set_dataloader(train_cfg.PHASE, "test", config)
    eval_res = evaluate(model, test_dl, train_cfg.EVAL_BATCHES, train_cfg.AMP_DTYPE, train_cfg.DEVICE, train_cfg.PHASE)
    del test_dl
    eval_res["samples"] = sample_generations(model, DEFAULT_PROMPTS[train_cfg.PHASE], train_cfg.DEVICE)
    append_jsonl(train_cfg.eval_log, {"step": state.step, **eval_res})
    tqdm.write(f"[eval {state.step}] loss {eval_res['loss']:.4f}  ppl {eval_res['perplexity']:.2f}")
 
    if eval_res["loss"] < state.best_eval:
        state.best_eval = eval_res["loss"]
        save_weights(train_cfg.CHECKPT_DIR / "best", model, {"step": state.step, "eval_loss": state.best_eval})
 
 
def save_checkpoint(model: nn.Module, optimizer, scheduler, scaler, state: TrainState, train_cfg: TrainConfig) -> None:
    state.last_checkpt = checkpoint_save(train_cfg.CHECKPT_DIR, state.step, model, optimizer, scheduler, scaler, meta=state.to_meta(train_cfg.PHASE), keep_last=train_cfg.KEEP_LAST)

"""
---------------------------------------------------------------------------
Train Loop
---------------------------------------------------------------------------
"""
def train(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR, scaler: torch.amp.GradScaler, stream: Generator, train_state: TrainState, train_cfg: TrainConfig, config: dict) -> TrainState:
    model.train()
    tracker = MetricTracker()
    nonfinite_steps = 0
    
    pbar = tqdm(total= train_cfg.MAX_STEPS, initial=train_state.step, dynamic_ncols=True, desc="Training...")
    
    try:
        while train_state.step < train_cfg.MAX_STEPS:
            want_acc = (train_state.step + 1) % train_cfg.LOG_EVERY == 0
            
            stats = train_step(model, optimizer, scheduler, scaler, stream, train_state, train_cfg, want_acc)
            train_state.step += 1
            pbar.update(1)
            
            if stats is not None:
                tracker.update(stats['loss_sum'], None, None, stats["n_target"], update_acc= False, input_tokens= stats["input_tokens"])
                tracker.add_accuracy(stats["correct"], stats["acc_n"])
                nonfinite_steps += int(not stats["stepped"])
                
                if want_acc:
                    log_step(tracker, scheduler, scaler, train_state, train_cfg, stats["grad_norm"], nonfinite_steps, pbar)
            
            if train_state.step % train_cfg.EVAL_EVERY == 0:
                run_eval(model, config, train_state, train_cfg)
                
            if train_state.step % train_cfg.CHECKPT_EVERY == 0 or train_state.step == train_cfg.MAX_STEPS:
                save_checkpoint(model, optimizer, scheduler, scaler, train_state, train_cfg)
                
    finally:
        pbar.close()
    
    return train_state
    

def main() -> None:
    # Parse arguments
    args = parse_args()
    
    # Load Config
    config = load_config(ROOT / "config.yaml")
    
    train_cfg = config["Train"]
    phase_cfg = train_cfg[f"Phase_{args.phase}"]
    
    # Set seed
    set_seed(train_cfg["seed"])
    
    # Device / Precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, scaler = get_precision(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set Variables
    train_cfg = TrainConfig(
        PHASE=args.phase,
        MAX_STEPS=int(phase_cfg["max_steps"]),
        GRAD_ACCUM=int(phase_cfg["grad_accum"]),
        NUM_WORKERS=int(train_cfg["num_workers"]),
        GRAD_CLIP=float(train_cfg["grad_clip"]),
        LOG_EVERY=int(train_cfg["log_every"]),
        EVAL_EVERY=int(train_cfg["eval_every"]),
        EVAL_BATCHES=int(train_cfg["eval_batches"]),
        CHECKPT_EVERY=int(train_cfg["ckpt_every"]),
        KEEP_LAST=int(train_cfg["keep_last"]),
        CHECKPT_DIR=ROOT / Path(train_cfg["save_dir"]) / f"Phase_{args.phase}",
        METRICS_DIR=ROOT / Path(config["Metrics"]["savepath"]) / f"Phase_{args.phase}",
        DEVICE=device,
        AMP_DTYPE=amp_dtype,
    )
    
    train_log, eval_log = train_cfg.METRICS_DIR / "train.jsonl", train_cfg.METRICS_DIR / "eval.jsonl"
    
    if train_cfg.NUM_WORKERS > 0:
        assert train_cfg.GRAD_ACCUM % train_cfg.NUM_WORKERS == 0, "grad_accum must be divisible by num_workers for exact resume"
    
    # Model / optimizer / scheduler
    model = NovaLM(config).to(train_cfg.DEVICE)
    if train_cfg.DEVICE.type == "cuda" and importlib.util.find_spec("triton") is not None:
        model.compile()
    
    optimizer = build_optimizer(model, phase_cfg["lr"], phase_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, phase_cfg["warmup_steps"], train_cfg.MAX_STEPS, phase_cfg["lr"], phase_cfg["min_lr"])
    
    # Training dependencies
    train_state = TrainState()
    
    if args.resume:
        state = checkpoint_load(args.resume, model, optimizer, scheduler, scaler, resume= True, device= train_cfg.DEVICE)
        train_state = TrainState.from_checkpoint(state)
        truncate_jsonl(train_log, train_state.step)
        truncate_jsonl(eval_log, train_state.step)
        tqdm.write(f"[resume] step {train_state.step}, epoch {train_state.epoch}, blocks_seen {train_state.blocks_seen}")
        
    elif args.init_from:
        checkpoint_load(args.init_from, model, optimizer, scheduler, scaler, resume=False, device=train_cfg.DEVICE)
        tqdm.write(f"[init] weights from {args.init_from}")
        
    # Set Dataloader
    skip_blocks = train_state.blocks_seen // max(train_cfg.NUM_WORKERS, 1)
    
        # Train DataLoader is built fresh once per epoch inside infinite_loader
    stream = infinite_loader(lambda ep, sk: set_dataloader(args.phase, "train", config, epoch= ep, skip_blocks= sk), train_state.epoch, skip_blocks)
    
    # Training loop
    state = train(model, optimizer, scheduler, scaler, stream, train_state, train_cfg, config)
    
    if state.last_checkpt is not None:
        if state.step % train_cfg.EVAL_EVERY != 0:
            run_eval(model, config, state, train_cfg)
        copy_checkpoint(state.last_checkpt, train_cfg.CHECKPT_DIR / "final", {"step": state.step, "best_eval": state.best_eval})
    
    plot_metrics(train_cfg.METRICS_DIR)
    
if __name__ == "__main__":
    main()
        
        
        
    