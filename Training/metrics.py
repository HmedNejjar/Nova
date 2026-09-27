from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(1, str(ROOT))

import json
import math
import time
import torch
from torch import Tensor

import plotly.graph_objects as go

# --------------------------------------------------------------------------- #
# Train-step metrics
# --------------------------------------------------------------------------- #

class MetricTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.loss_sum = 0.0     # sum of per-micro-batch *total* losses
        self.n_target = 0       # total number of target tokens in all micro-batches seen (label != -100)
        self.correct = 0        # total number of correct predictions in all micro-batches seen
        self.acc_n = 0          # positions accuracy was actually measured on
        self.input_tokens = 0   # all prompt tokens (throughput measure)
        self.time_start = time.perf_counter()

    def update(self, loss_sum: Tensor, logits: Tensor | None, labels: Tensor | None, n_target: int, *, update_acc: bool = False, input_tokens: int = 0) -> None:
        """Update metrics with a single micro-batch
        Args:
            loss_sum (Tensor): Total loss for the micro-batch (already summed over tokens)
            logits (Tensor): Model output logits of shape (batch_size, seq_len, vocab_size)
            labels (Tensor): Ground truth labels of shape (batch_size, seq_len)
            n_target (int): Number of target tokens in the micro-batch (label != -100)
            update_acc (bool): Whether to update accuracy metrics
            input_tokens (int): Number of input tokens processed in this micro-batch
        """
        
        self.loss_sum += float(loss_sum.item()) if isinstance(loss_sum, Tensor) else float(loss_sum)
        self.n_target += n_target
        self.input_tokens += input_tokens
        if update_acc:
            # labels are already shifted by the collate (labels[t] = input_ids[t+1]),
            # so logits[:, t] is compared with labels[:, t] directly
            with torch.no_grad():
                if logits is None or labels is None:
                    return
                logits = logits.detach()
                pred = logits.argmax(dim=-1)     # (batch_size, seq_len)
                valid = (labels != -100)
                self.correct += int((pred[valid] == labels[valid]).sum().item())
                self.acc_n += int(valid.sum().item())
                
    def add_accuracy(self, correct: int, n: int) -> None:
        """Add precomputed accuracy counts.

        The chunked loss paths (chunked_lm_loss / chunked_token_loss) never build
        full logits, so they count argmax hits per chunk and report them here
        instead of going through update(..., update_acc=True).
        """
        self.correct += int(correct)
        self.acc_n += int(n)
            
    def compute(self) -> dict:
        """Compute final metrics
        Returns:
            dict: Dictionary of computed metrics
        """
        dt = time.perf_counter() - self.time_start
        return {
                "dt": dt,
                "loss": self.loss_sum / self.n_target if self.n_target else float("nan"),
                "acc": self.correct / self.acc_n if self.acc_n > 0 else None,
                "throughput": self.input_tokens / dt if self.input_tokens > 0 and dt > 0 else 0.0,
                }

# --------------------------------------------------------------------------- #
# Phase 3 per-case loss
# (0 = direct answer, 1 = thinking, 2 = multilingual direct, 3 = identity)
# --------------------------------------------------------------------------- #

CASE_NAMES = {0: "direct", 1: "thinking", 2: "multilingual", 3: "identity"}

class CaseTracker:
    """Token-weighted loss per Phase 3 case, built from per-token losses.

    conv_bounds / case_labels come from the Phase 3 collate as per-sample
    lists: conv_bounds[i] is (k, 2) local (start, end), case_labels[i] is (k,).
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_sum: dict[int, float] = {}
        self.tokens: dict[int, int] = {}

    def update(self, token_loss: Tensor, labels: Tensor, conv_bounds: list, case_labels: list) -> None:
        """
        Args:
            token_loss: (batch_size, seq_len) per-token loss, from cross_entropy(..., reduction="none")
            labels: (batch_size, seq_len) shifted labels, -100 where there is no target
            conv_bounds: per-sample (k, 2) LongTensors of local (start, end)
            case_labels: per-sample (k,) LongTensors, aligned with conv_bounds
        """
        # map every position to its conversation's case on the CPU, then
        # reduce on the device once per case (one sync per case, not per conv)
        case_map = torch.full(labels.shape, -1, dtype=torch.long)
        for i, (bounds, cases) in enumerate(zip(conv_bounds, case_labels)):
            for (start, end), case in zip(bounds.tolist(), cases.tolist()):
                case_map[i, start:end] = case
        case_map = case_map.to(labels.device, non_blocking=True)

        valid = (labels != -100)
        token_loss = token_loss.detach().float()
        for case in case_map.unique().tolist():
            if case < 0:
                continue
            sel = valid & (case_map == case)
            n = int(sel.sum().item())
            if n == 0:
                continue
            self.loss_sum[case] = self.loss_sum.get(case, 0.0) + float(token_loss[sel].sum().item())
            self.tokens[case] = self.tokens.get(case, 0) + n

    def compute(self) -> dict:
        """{case_name: mean loss} for every case seen."""
        return {CASE_NAMES.get(c, str(c)): self.loss_sum[c] / self.tokens[c]
                for c in sorted(self.tokens)}

# --------------------------------------------------------------------------- #
# Plots from the JSONL logs
# --------------------------------------------------------------------------- #

def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL log, skipping blank or partially written lines."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue    # a crash mid-write can leave a truncated last line
    return records

def plot_metrics(metrics_dir: str | Path) -> list[Path]:
    """Write loss / accuracy / lr (and Phase 3 per-case loss) HTML plots
    from train.jsonl and eval.jsonl in `metrics_dir`. Returns written paths."""


    metrics_dir = Path(metrics_dir)
    train = read_jsonl(metrics_dir / "train.jsonl")
    evals = read_jsonl(metrics_dir / "eval.jsonl")

    def series(records: list[dict], key: str) -> tuple[list, list]:
        pts = [(r["step"], r[key]) for r in records
               if r.get(key) is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
        return [p[0] for p in pts], [p[1] for p in pts]

    def figure(title: str, y_title: str, traces: list, log_y: bool = False) -> go.Figure:
        fig = go.Figure(traces)
        fig.update_layout(title=title, xaxis_title="Step", yaxis_title=y_title,
                          hovermode="x unified", autosize=True, height=600)
        if log_y:
            fig.update_yaxes(type="log")
        return fig

    written = []
    figs = {
        "loss": figure("Loss", "Cross-entropy", [
            go.Scatter(x=series(train, "loss")[0], y=series(train, "loss")[1], name="train", mode="lines"),
            go.Scatter(x=series(evals, "loss")[0], y=series(evals, "loss")[1], name="eval", mode="lines+markers"),
        ]),
        "accuracy": figure("Next-token accuracy", "Accuracy", [
            go.Scatter(x=series(train, "acc")[0], y=series(train, "acc")[1], name="train", mode="lines"),
            go.Scatter(x=series(evals, "acc")[0], y=series(evals, "acc")[1], name="eval", mode="lines+markers"),
        ]),
        "lr": figure("Learning rate", "LR", [
            go.Scatter(x=series(train, "lr")[0], y=series(train, "lr")[1], name="lr", mode="lines"),
        ], log_y=True),
    }

    # Phase 3: one line per case, from eval records that carry "cases"
    case_names = sorted({name for r in evals for name in (r.get("cases") or {})})
    if case_names:
        figs["cases"] = figure("Eval loss per case", "Cross-entropy", [
            go.Scatter(x=[r["step"] for r in evals if name in (r.get("cases") or {})],
                       y=[r["cases"][name] for r in evals if name in (r.get("cases") or {})],
                       name=name, mode="lines+markers")
            for name in case_names
        ])

    for name, fig in figs.items():
        out = metrics_dir / f"{name}.html"
        fig.write_html(str(out), include_plotlyjs="cdn")
        written.append(out)
    return written
