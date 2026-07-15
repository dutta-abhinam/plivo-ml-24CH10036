"""A small GPT in plain PyTorch, kept a strict superset of the starter so the
official evaluate.py (`from model import GPT, Config`) still rebuilds it from a
checkpoint's saved config dict.

Changes vs. the mediocre baseline (all opt-in via Config, all defended in
RUNLOG.md):
  * GPT-2 style init: N(0, init_std) with the residual projections scaled by
    1/sqrt(2*n_layer) so the residual stream variance does not grow with depth
    (baseline used one flat std=0.05 for every weight).
  * weight tying (head.weight = tok_emb.weight) -- frees vocab*n_embd params
    and couples the input/output representations, which helps a tiny,
    data-starved model.
  * optional rotary position embeddings (pos_type="rope"): zero extra params,
    relative positions instead of a learned absolute table.
Only PyTorch is used; nothing here needs numpy or any custom kernel.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 256      # overwritten by the tokenizer's real vocab
    block_size = 128
    n_layer = 4
    n_head = 4
    n_embd = 160
    dropout = 0.0
    tie_weights = True    # baseline shipped False; tying is a clear win here
    init_std = 0.02       # baseline used 0.05 flat
    pos_type = "learned"  # "learned" | "rope"


# ---------------------------------------------------------------------------
# rotary position embeddings (parameter-free)
# ---------------------------------------------------------------------------
def _build_rope_cache(seq_len, head_dim, base=10000.0):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, theta)                 # (T, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim)
    T = x.shape[-2]
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.pos_type = getattr(cfg, "pos_type", "learned")
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        if self.pos_type == "rope":
            cos, sin = _build_rope_cache(cfg.block_size, self.head_dim)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.pos_type == "rope":
            q = _apply_rope(q, self.rope_cos, self.rope_sin)
            k = _apply_rope(k, self.rope_cos, self.rope_sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pos_type = getattr(cfg, "pos_type", "learned")
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if self.pos_type == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if getattr(cfg, "tie_weights", False):
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)
        # residual-projection scaling (GPT-2): damp the two adds per block
        std = getattr(cfg, "init_std", 0.02)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                if p.dim() == 2:                 # attn-out + mlp-out projections
                    nn.init.normal_(p, mean=0.0,
                                    std=std / math.sqrt(2 * cfg.n_layer))

    def _init(self, m):
        std = getattr(self.cfg, "init_std", 0.02)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos_type == "learned":
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        # count each tensor once so tied weights are not double counted
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total
