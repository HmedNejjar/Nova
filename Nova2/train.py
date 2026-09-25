import argparse
from pathlib import Path
import yaml
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from .dataset import TokenShardDataset
from .model import Nova2Config, Nova2ForCausalLM

def main():
    ap = argparse.ArgumentParser(description="Nova 2.0 Phase 3 pretraining")
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f: cfg = yaml.safe_load(f)
    m, t, d = cfg["Model"], cfg["Train"], cfg["Datasets"]["Phase_3"]
    seq_len = int(d.get("seq_len", 4096))
    mc = Nova2Config(vocab_size=int(cfg["Tokenizer"]["vocab_size"]), embed_dim=int(m["embed_dim"]),
        num_layers=int(m["num_layers"]), num_heads=int(m["num_heads"]), hidden_dim=int(m["hidden_dim"]),
        num_kv_heads=int(m["num_kv_heads"]), head_dim=int(m["head_dim"]), max_seq_len=int(m["max_seq_len"]),
        weight_tying=bool(m["weight_tying"]), bias=bool(m["bias"]), dropout=float(m["dropout"]),
        eps=float(m["eps"]), rope_base=float(m["rope_base"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    model = Nova2ForCausalLM(mc).to(device)
    print(f"parameters={model.num_parameters():,} seq_len={seq_len} device={device}")
    train_ds = TokenShardDataset(d["train"], seq_len); test_ds = TokenShardDataset(d["test"], seq_len)
    workers = int(t.get("num_workers", 4)); bs = int(t["batch_size"])
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=device.type=="cuda", drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=device.type=="cuda")
    opt = AdamW(model.parameters(), lr=float(m["learning_rate"]), weight_decay=float(t.get("weight_decay", 0.1)), betas=(0.9, 0.95), fused=device.type=="cuda")
    accum = int(t.get("gradient_accumulation_steps", 1)); scaler = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    if args.resume:
        state = torch.load(args.resume, map_location=device); model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
    save = Path(m["savepath"]); save.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(t["epochs"])):
        model.train(); opt.zero_grad(set_to_none=True); running = 0.0
        for step, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=="cuda"):
                loss = model(x, labels=y)["loss"] / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            running += loss.item() * accum
            if step % 50 == 0: print(f"epoch={epoch+1} step={step} loss={loss.item()*accum:.4f}")
        ckpt = {"model": model.state_dict(), "optimizer": opt.state_dict(), "config": mc.__dict__, "epoch": epoch+1}
        torch.save(ckpt, save.with_suffix(".pt"))
        model.eval(); total = n = 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=="cuda"):
                    total += model(x, labels=y)["loss"].item()
                n += 1
        print(f"epoch={epoch+1} train_loss={running/max(1,len(train_dl)):.4f} eval_loss={total/max(1,n):.4f}")

if __name__ == "__main__": main()
