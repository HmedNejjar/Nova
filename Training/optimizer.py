from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(1, str(ROOT))

from math import pi, cos
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    """
    Build an AdamW optimizer for the model.

    Args:
        model (nn.Module): The model to optimize.
        lr (float): The learning rate.
        weight_decay (float): The weight decay factor.

    Returns:
        AdamW: The constructed optimizer.
    """
    decay, no_decay = [], []
    
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        if p.ndim >= 2 and not (name.endswith(".scale") or name.endswith(".bias")):
            decay.append(p)
        else:
            no_decay.append(p)
    
    fused = torch.cuda.is_available()
    
    optimizer_params = [{"params": decay, "weight_decay": weight_decay},
                        {"params": no_decay, "weight_decay": 0.0}]
    
    return AdamW(optimizer_params, lr= float(lr), betas=(0.9, 0.95), fused= fused)

def build_scheduler(opt: torch.optim.Optimizer, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> LambdaLR:
    """
    Build a learning rate scheduler with linear warmup and cosine decay.

    Args:
        opt (torch.optim.Optimizer): The optimizer to schedule.
        warmup_steps (int): Number of steps for linear warmup.
        max_steps (int): Total number of training steps.
        lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate after decay.

    Returns:
        LambdaLR: The constructed learning rate scheduler.
    """
    assert max_steps > warmup_steps, "max_steps must exceed warmup_steps"
    
    # Guard against warmup_steps == 0, which would cause division by zero
    warmup_steps = max(warmup_steps, 1)
    
    # `LambdaLR` multiplies the base LR by the value returned from
    # `lr_lambda`. We want the final LR to be `min_lr`, so express the
    # floor as a fraction of the base LR.
    lr, min_lr = float(lr), float(min_lr)
    floor = min_lr / lr
    
    def lr_lambda(step: int) -> float:
        step = step + 1
        
        # Ramp linearly from 0 (at step 0) up to 1 (at step == warmup_steps).
        if step < warmup_steps:
            return step / warmup_steps
        
        # `progress` goes from 0 at the end of warmup to 1 at `max_steps`.
        # Clamp to 1 so that steps beyond `max_steps` don't extrapolate the cosine past its minimum (which would make the LR go back up)
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        progress = min(progress, 1)
        
        cosine_decay = 0.5 * (1 + cos(pi * progress))
        
        # Interpolate between the floor (min_lr/lr) and 1.0 so that:
        #   progress=0 -> 1.0        (LR == lr)
        #   progress=1 -> floor      (LR == min_lr)
        return floor + (1.0 - floor) * cosine_decay
    
    return LambdaLR(opt, lr_lambda= lr_lambda)
        
        
    
    