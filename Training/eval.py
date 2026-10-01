#type: ignore

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(1, str(ROOT))

import math
import torch, torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from Training.utils import to_device
from Training.metrics import MetricTracker, CaseTracker

# Fixed prompts so samples are comparable from one eval to the next
PHASE3_SYSTEM = ("You are Nova AI, a helpful AI assistant. Answer clearly, "
                 "accurately, and concisely. Admit when you are unsure.")

DEFAULT_PROMPTS = {
    1: ["The history of the city", "In mathematics, a prime number"],
    2: ["The history of the city", "def fibonacci(n):"],
    3: [f"<bos><system>{PHASE3_SYSTEM}</system><user>What is the capital of France?</user><assistant>"],
}

# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

def chunk_has_targets(labels: Tensor, chunk_size: int = 1024) -> list[bool]:
    """Which sequence chunks of `labels` contain at least one target.

    Call it on the CPU copy of the labels: asking a GPU tensor makes the CPU wait
    for every queued kernel, which stalls the launch pipeline.
    """
    return [bool((labels[:, start:start + chunk_size] != -100).any()) for start in range(0, labels.size(1), chunk_size)]

def chunked_lm_loss(model: nn.Module, hidden: Tensor, labels: Tensor, chunk_size: int = 1024, want_acc: bool = False, active_chunks: list[bool] | None = None) -> tuple[Tensor, int, int]:
    """Training loss from the final hidden states, one sequence chunk at a time.

    Full logits for a 150k vocab are huge (micro 4 x 4096 tokens is ~4.9 GB in
    bf16, plus fp32 copies for backward). Here each chunk's logits are built by
    lm_head inside torch.utils.checkpoint, so only one chunk's logits exist at a
    time and backward recomputes them instead of storing them.

    Args:
        model: NovaLM (uses model.lm_head).
        hidden: (batch_size, seq_len, embed_dim) from model(..., return_hidden=True).
        labels: (batch_size, seq_len) shifted labels, -100 where there is no target.
        want_acc: also count correct argmax predictions (costs one extra
            no-grad pass over lm_head, so only ask on logging steps).
        active_chunks: chunk_has_targets() of the CPU labels; chunks marked False
            are skipped. If None it is computed from `labels` (a GPU sync).

    Returns:
        (loss_sum, correct, n_acc): summed cross-entropy (with grad) over all
        non-ignored targets, and the accuracy counts (0, 0 if not want_acc).
    """
    weight, bias = model.lm_head.weight, model.lm_head.bias

    def chunk_ce(h: Tensor, lbl: Tensor) -> Tensor:
        logits = F.linear(h, weight, bias).float()
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), lbl.reshape(-1),
                               ignore_index=-100, reduction="sum")

    if active_chunks is None:
        active_chunks = chunk_has_targets(labels, chunk_size)

    loss_sum = hidden.new_zeros((), dtype=torch.float32)
    # accuracy counts stay on the GPU and are read back once at the end
    correct = hidden.new_zeros((), dtype=torch.long)
    n_acc = hidden.new_zeros((), dtype=torch.long)
    for start, active in zip(range(0, hidden.size(1), chunk_size), active_chunks):
        if not active:
            continue    # nothing to learn in this chunk (e.g. Phase 3 prompt, padding)
        h = hidden[:, start:start + chunk_size]
        lbl = labels[:, start:start + chunk_size]
        loss_sum = loss_sum + checkpoint(chunk_ce, h, lbl, use_reentrant=False)
        if want_acc:
            with torch.no_grad():
                pred = F.linear(h, weight, bias).argmax(dim=-1)
                valid = lbl != -100
                correct += ((pred == lbl) & valid).sum()
                n_acc += valid.sum()
    if not want_acc:
        return loss_sum, 0, 0
    correct, n_acc = torch.stack((correct, n_acc)).tolist()
    return loss_sum, correct, n_acc

@torch.no_grad()
def chunked_token_loss(model: nn.Module, hidden: Tensor, labels: Tensor, chunk_size: int = 1024) -> tuple[Tensor, int, int]:
    """Eval-side twin of chunked_lm_loss: per-token loss from final hidden states.
 
    Only one (B, chunk, vocab) slab of logits exists at a time, so eval memory
    matches training instead of materializing the full (B, T, 150k) tensor.
    No grad, so no checkpointing needed.
 
    Args:
        model: NovaLM (uses model.lm_head).
        hidden: (batch_size, seq_len, embed_dim) from model(..., return_hidden=True).
        labels: (batch_size, seq_len) shifted labels, -100 where there is no target.
 
    Returns:
        (tok_loss, correct, n_acc): tok_loss is (B, T) fp32, 0 where labels == -100
        (what CaseTracker expects); correct / n_acc are argmax accuracy counts.
    """
    weight, bias = model.lm_head.weight, model.lm_head.bias
    batch_size, seq_len, _ = hidden.shape
    tok_loss = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=hidden.device)
    correct, n_acc = 0, 0
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        lbl = labels[:, start:end]
        valid = lbl != -100
        if not valid.any():
            continue    # prompt-only / padding chunk: loss stays 0
        logits = F.linear(hidden[:, start:end], weight, bias).float()
        tok_loss[:, start:end] = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), lbl.reshape(-1),
            ignore_index=-100, reduction="none",
        ).view(batch_size, end - start)
        pred = logits.argmax(dim=-1)
        correct += int((pred[valid] == lbl[valid]).sum().item())
        n_acc += int(valid.sum().item())
        del logits, pred
    return tok_loss, correct, n_acc

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, max_batches: int, amp_dtype: torch.dtype, device: torch.device, phase: int) -> dict:
    """Score up to `max_batches` batches of a non-shuffled eval DataLoader.

    Build a fresh DataLoader (set_dataloader(phase, "test", ...)) for every
    eval so each one scores the same batches and the numbers are comparable.
    
    Args:
        model: The model to evaluate.
        dl: The DataLoader to evaluate on.
        max_batches: The maximum number of batches to evaluate.
        amp_dtype: The AMP dtype to use.
        device: The device to use.
        phase: The phase to use.

    Returns:
        {"loss", "perplexity", "accuracy", "batches", "tokens"} plus "cases" in Phase 3.
        loss is token-weighted over all scored batches.
    """
    was_training = model.training
    model.eval()

    tracker = MetricTracker()
    cases = CaseTracker() if phase == 3 else None
    use_amp = device.type == "cuda" and amp_dtype != torch.float32

    n_batches = 0
    pbar = tqdm(dl, total=max_batches, desc="eval", leave=False, position=1, dynamic_ncols=True)
    try:
        for batch in pbar:
            if n_batches >= max_batches:
                break
            batch = to_device(batch, device)
            labels = batch["labels"]

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                hidden, _ = model(batch["input_ids"], None, batch.get("position_ids"), batch.get("attn_mask"),
                                  return_hidden=True)
                tok_loss, correct, n_acc = chunked_token_loss(model, hidden, labels)

            n_target = int((labels != -100).sum().item())
            tracker.update(tok_loss.sum(), None, None, n_target, input_tokens=batch["input_ids"].numel())
            tracker.add_accuracy(correct, n_acc)
            if cases is not None:
                cases.update(tok_loss, labels, batch["conv_bounds"], batch["case_labels"])

            n_batches += 1
            del hidden, tok_loss
    finally:
        pbar.close()
        # restore whatever mode the caller was in, even if eval raised
        model.train(was_training)

    m = tracker.compute()
    loss = m["loss"]
    result = {
        "loss": loss,
        # cap the exponent so an early, untrained model doesn't overflow to inf
        "perplexity": math.exp(min(loss, 20.0)) if not math.isnan(loss) else float("nan"),
        "accuracy": m["acc"],
        "batches": n_batches,
        "tokens": tracker.n_target,
    }
    if cases is not None:
        result["cases"] = cases.compute()
    return result

# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #

@torch.no_grad()
def sample_generations(model: nn.Module, prompts: list[str], device: torch.device, max_new_tokens: int = 48) -> dict[str, str]:
    """Greedy continuations of fixed prompts, for a quick read on quality.

    temperature=0 decodes greedily, so the same weights always give the same
    text and changes between evals come from training, not randomness.
    NovaLM.generate() restores the model's train/eval mode itself.
    """
    was_training = model.training
    samples = {}
    
    for prompt in prompts:
        samples[prompt] = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0, stop_tokens=("<eos>", "</assistant>"), device=device, return_prompt=False)

    return samples
