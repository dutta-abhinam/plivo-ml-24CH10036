"""Trainer for the 2,000-step speedrun. A superset of the starter: the mandated
command still works unchanged --

    python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt

-- but every questionable baseline choice is now a flag so runs can change ONE
thing at a time (see RUNLOG.md). Improvements over the baseline trainer:
  * AdamW with decoupled weight decay (baseline: plain Adam, no decay)
  * linear warmup + cosine decay to min_lr (baseline: constant LR)
  * gradient clipping (baseline: none)
  * weight decay only on 2-D weights; none on biases / LayerNorm / embeddings
  * optional end-of-run dev bpb readout + a runs.csv row, so the RUNLOG numbers
    come straight from the official scorer, not from eyeballing train loss.

HARD CAPS (asserted below): <=2000 steps, <=2,000,000 params, corpus only,
pure PyTorch/stdlib.
"""
import argparse
import csv
import math
import os
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def lr_at(step, steps, base_lr, min_lr, warmup, schedule):
    if warmup > 0 and step <= warmup:
        return base_lr * step / warmup
    if schedule == "constant":
        return base_lr
    prog = (step - warmup) / max(1, steps - warmup)          # cosine to min_lr
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


class Lion(torch.optim.Optimizer):
    """Minimal Lion (EvoLved Sign Momentum, Chen et al. 2023). Sign-based
    update -> effective step size ~lr regardless of gradient scale, so it needs
    a markedly smaller LR than Adam. Included as the Phase-3 'new optimizer'
    experiment; pure PyTorch, no dependencies."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                update = m.mul(b1).add_(p.grad, alpha=1 - b1).sign_()
                p.add_(update, alpha=-g["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)
        return loss


def make_optimizer(model, lr, wd, opt_name):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and "emb" not in n:       # linear weights only
            decay.append(p)
        else:                                     # biases, LayerNorm, embeddings
            no_decay.append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]
    if opt_name == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=(0.9, 0.999))
    if opt_name == "lion":
        return Lion(groups, lr=lr, betas=(0.9, 0.99))
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


@torch.no_grad()
def dev_bpb(model, cfg, tok, dev_path):
    import evaluate
    text = open(dev_path, encoding="utf-8").read()
    model.eval()
    bpb, _, _ = evaluate.bits_per_byte(model, cfg, tok, text)
    model.train()
    return bpb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=None)   # default lr/10
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=0.0)
    ap.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    ap.add_argument("--opt", choices=["adam", "adamw", "lion"], default="adam")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=160)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--tie", type=int, default=0)           # 0/1 weight tying
    ap.add_argument("--init_std", type=float, default=0.05)
    ap.add_argument("--pos", choices=["learned", "rope"], default="learned")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--dev", default=None)                  # eval bpb at the end
    ap.add_argument("--tag", default="")                   # label for runs.csv
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n_bytes = len(text.encode("utf-8"))
    print(f"corpus: {n_bytes:,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size}, {n_bytes/len(ids):.2f} bytes/token)")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    cfg.block_size = args.block
    cfg.n_layer, cfg.n_head, cfg.n_embd = args.n_layer, args.n_head, args.n_embd
    cfg.dropout = args.dropout
    cfg.tie_weights = bool(args.tie)
    cfg.init_std = args.init_std
    cfg.pos_type = args.pos
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params  (tie={cfg.tie_weights} pos={cfg.pos_type} "
          f"L{cfg.n_layer} H{cfg.n_head} E{cfg.n_embd} blk{cfg.block_size})")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params (got {n:,})"

    min_lr = args.min_lr if args.min_lr is not None else args.lr / 10
    opt = make_optimizer(model, args.lr, args.wd, args.opt)

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps, args.lr, min_lr, args.warmup, args.schedule)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        losses.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr:.2e}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)")

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    dt = time.time() - t0
    print(f"saved {args.out}  ({dt:.0f}s total)")

    if args.dev:
        bpb = dev_bpb(model, cfg, tok, args.dev)
        print(f"DEV bpb = {bpb:.4f}")
        row = {"tag": args.tag, "bpb": round(bpb, 4), "params": n,
               "steps": args.steps, "vocab": cfg.vocab_size,
               "batch": args.batch, "block": cfg.block_size,
               "n_layer": cfg.n_layer, "n_head": cfg.n_head,
               "n_embd": cfg.n_embd, "lr": args.lr, "min_lr": min_lr,
               "warmup": args.warmup, "wd": args.wd, "grad_clip": args.grad_clip,
               "schedule": args.schedule, "opt": args.opt,
               "tie": int(cfg.tie_weights), "init_std": cfg.init_std,
               "pos": cfg.pos_type, "dropout": args.dropout,
               "seed": args.seed, "sec": round(dt)}
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.csv")
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    main()
