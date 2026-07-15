# RUNLOG — 2,000-step LLM speedrun

Score = **bits per byte (bpb)** on held-out text, lower is better. All numbers
below are the official `evaluate.py` scorer run on `data/dev_eval.txt`. Every
run is 2,000 steps on CPU. `runs.csv` holds the machine-written record for each
row (I never hand-typed a bpb number into this log).

Corpus facts that drove the plan: 7.32 MB, ~86% of *characters* are ASCII but
~33% of *bytes* are Devanagari, because every Hindi code point is 3 UTF-8 bytes.
At batch 8 × block 128 the baseline sees ~2.05 M byte-tokens in 2,000 steps —
under 30% of the corpus. So the two biggest structural problems are (1) the
tokenizer wastes the step budget on 3-byte Devanagari characters and (2) the
model is data-starved.

## 0–10 min: what is wrong with the baseline (`starter/`)

Read `train.py` / `model.py` before touching anything. Questionable choices,
in rough order of how much I expected each to matter:

1. **Byte tokenizer (vocab 256).** 3 tokens per Hindi char; the effective
   context in *characters* is tiny and the model re-predicts deterministic
   UTF-8 continuation bytes. Biggest lever.
2. **Constant LR, no schedule, no warmup.** For a 2,000-step run you want to
   anneal; a flat 3e-4 both wastes early steps and never settles.
3. **Plain Adam, no weight decay.** No regularisation on a model that will see
   <1 epoch.
4. **No gradient clipping.** Occasional large steps on a small batch.
5. **Flat init std=0.05 for every weight.** Too large, and it ignores residual
   growth with depth — the residual stream variance compounds over 4 blocks.
6. **`tie_weights=False`.** Wastes `vocab×n_embd` params and decouples input and
   output token representations, which a tiny model can't afford.
7. **LR 3e-4 is conservative** for a ~1.3M-param model at this width.

## Baseline

| tag | change | dev bpb |
|-----|--------|---------|
| baseline (starter, unmodified) | — | **2.3718** |

`python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt`
then `python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt`
→ `{"bpb": 2.3718, "n_params": 1339840, "steps": 2000}`.

## Phase 1 — trainer recipe (byte tokenizer, one change at a time)

Cumulative ladder: each run adds ONE thing to the previous. Byte tokenizer
throughout (`TOKMODE=byte`) so this isolates the *trainer* from the tokenizer
change in Phase 2. Batch 8, block 128, L4/H4/E160 (baseline shape).

| tag | change (vs previous) | dev bpb | Δ |
|-----|----------------------|---------|----|
| baseline | starter as shipped | 2.3718 | — |
| r0b | GPT-2 residual-scaled init (still std 0.05) | 2.3290 | −0.043 |
| r1 | LR 3e-4 → 1e-3 | 2.2474 | −0.082 |
| **r2** | **+ cosine decay + warmup 100** | **2.2148** | **−0.033** |
| r3 | + AdamW + weight decay 0.1 | 2.2447 | +0.030 ✗ |
| r4 | + grad-clip 1.0 | 2.2415 | −0.003 |
| r5 | + init std 0.05 → 0.02 | 2.4138 | +0.172 ✗ |
| r6 | + weight tying (on the r5 base) | 2.4602 | +0.046 ✗ |

**r0b — hypothesis:** the flat std=0.05 init ignores residual growth over depth.
Scaling the two residual output projections by 1/√(2·n_layer) should steady the
residual stream. **Result:** 2.3718 → 2.3290. Confirmed; kept for every later run.

**r1 — hypothesis:** 3e-4 is too timid for a 1.3M model. **Result:** biggest
single win, −0.082. The baseline was simply under-trained by a low LR.

**r2 — hypothesis:** a constant LR wastes a fixed-length run; warm up then
cosine-anneal to lr/10. **Result:** −0.033, best of the phase. This is the
recipe I carry forward.

**r3 — hypothesis (LOST):** AdamW + wd 0.1 is the "modern default", should help.
**Result:** +0.030, a regression. **Why:** at <1 epoch the model is *under*-fit,
not over-fit; weight decay pulls weights toward zero and fights the little
signal 2,000 steps buy. (The adamw betas 0.9/0.95 also shorten the 2nd-moment
memory, adding variance on noisy batch-8 CPU steps.) Regularisation is the wrong
medicine when you are underfitting → reverted to plain Adam, no decay.

**r4 — hypothesis:** grad-clip 1.0 as cheap insurance. **Result:** neutral
(−0.003); no instability to clip here. Kept only as a safety net.

**r5 — hypothesis (LOST):** init std 0.02 is textbook GPT-2. **Result:** +0.172,
a large regression. **Why:** smaller init = smaller initial logits and signal;
with only 2,000 steps there is no time to grow representations from a tiny
start. The "asymptotically nicer" init loses at a short horizon; 0.05 gives a
head start. Reverted to 0.05.

**r6 — hypothesis:** tie input/output embeddings. **Result:** +0.046 *but built
on the already-bad r5 base*, so it is confounded — re-tested cleanly as r7 in
Phase 2.

**Phase 1 conclusion.** Best recipe = **Adam, LR 1e-3, cosine + warmup 100,
init 0.05, residual-scaled projections** (r2 = 2.2148, −6.6% vs baseline). The
two "obvious" modern tweaks (AdamW+wd, small init) both *hurt* in this
<1-epoch regime — the single most useful thing Phase 1 taught me.

## Phase 2 — tokenizer + capacity allocation (the headline change)

All Phase-2 runs use the r2 recipe (Adam, lr 1e-3, cosine+warmup 100, init 0.05).
`r7` first re-tests tying cleanly; then the BPE runs switch on the byte-level
BPE tokenizer (trained only on the corpus), weight tying (needed to fit the
larger vocab), and RoPE (parameter-free positions, so a 256-token block costs
nothing). The design question: given ~2M params, spend them on **vocab
(compression)**, **depth**, or **width**?

| tag | config | vocab | bytes/tok | params | dev bpb |
|-----|--------|-------|-----------|--------|---------|
| r7_tie_on_r2 | byte, +tie on the good r2 base | 256 | 1.00 | 1.30M | 2.2164 |
| b1_v2048_L3 | BPE, E192 L3 rope blk256 | 2048 | 3.36 | 1.73M | 1.7540 |
| b2_v1024_L4 | BPE, E192 **L4** rope blk256 | 1024 | 2.81 | 1.98M | 1.7595 |
| **b3_v3072_L3** | BPE, E192 L3 rope blk256 | **3072** | 3.71 | 1.92M | **1.7416** |

**r7 — hypothesis:** tying was blamed in r6, but that was on the bad init-0.02
base. Re-test on r2. **Result:** 2.2148 → 2.2164, i.e. **neutral** (+0.0016,
noise). Conclusion: tying is ~free here, so I can use it to afford a larger BPE
vocab. r6's regression was the init, not the tying.

**b1 — hypothesis (the big bet):** a byte tokenizer spends 3 tokens per Hindi
char; BPE compressing ~3.4x lets 2,000 steps cover ~a full epoch and shortens
the prediction chain. **Result:** 2.2148 → **1.7540**, −0.46 bpb (−21%). By far
the largest single win of the whole exercise — the tokenizer was the lever, as
the corpus byte-composition predicted.

**b2 — hypothesis:** spend the budget on **depth** (L4) instead of vocab; use
the freed params from a smaller 1024 vocab. **Result:** 1.7595, slightly *worse*
than b1. A 4th layer did not pay for the lost compression at this scale.

**b3 — hypothesis:** push the other way — more **compression** (vocab 3072,
3.71 bytes/tok), stay at L3. **Result:** **1.7416**, best so far. Within L3,
more vocab keeps helping; combined with b2 this says **compression > depth** for
this corpus/budget. (Vocab 4096 at E192/L3 would be ~2.12M params — over the
cap — so 3072 is near the sweet spot without shrinking the model.)

**Phase 2 conclusion.** Best = **BPE vocab 3072, E192, L3, H4, RoPE, block 256,
tied, r2 recipe → 1.7416 dev bpb** (1.92M params). This is the final
architecture. Set as the default `bpe.json`.

## Phase 3 — ambitious experiment (a new optimizer) + final

**Hypothesis:** Lion (sign-momentum, Chen et al. 2023) is a fashionable Adam
replacement — try swapping it in on the best config and see if it beats Adam.
Implemented in ~15 lines of pure PyTorch (`Lion` in `train.py`).

| tag | optimizer | lr | dev bpb | Δ vs Adam |
|-----|-----------|-----|---------|-----------|
| **final** | **Adam** | **1e-3** | **1.7416** | — |
| lion_lr1e-3 | Lion | 1e-3 (Adam's LR) | 2.0007 | +0.259 ✗ |
| lion_lr2e-4 | Lion | 2e-4 (the fix) | 1.9167 | +0.175 |

**lion_lr1e-3 — LOST.** At Adam's LR, Lion scored 2.0007 vs Adam's 1.7416.
**Why:** Lion's update is `sign(β1·m + (1−β1)·g)` — a fixed step of magnitude
≈lr per coordinate, independent of the gradient's scale. An LR tuned for Adam's
adaptive (per-coordinate, gradient-scaled) steps is far too large for a raw sign
step, so training is effectively over-shooting.

**lion_lr2e-4 — the fix, partially.** Following that diagnosis I dropped the LR
5× to 2e-4. bpb improved 2.0007 → 1.9167, confirming the mechanism (Lion needs a
much smaller LR). But it *still* trailed Adam. **Why Lion loses here even tuned:**
its documented wins come at large batch size and long schedules where its
implicit regularisation and cheaper state pay off; in a 2,000-step, batch-8 CPU
run neither applies, and Adam's per-coordinate second moment simply fits a tiny
model faster with so few updates. **Concluded:** keep Adam; Lion is the wrong
tool for this regime. (I did not spend the remaining budget chasing a Lion +
higher-warmup + weight-decay combo — the gap to Adam was too large to be worth
the runs.)

## Final checkpoint

`ckpt.pt` is the **final** row above: the best config, retrained from the same
seed, saved with its step count.

| | value |
|-|-------|
| dev bpb | **1.7416** |
| params | 1,924,800 (< 2,000,000 cap) |
| steps | 2000 (= cap) |
| tokenizer | byte-level BPE, vocab 3072, lossless |
| model | GPT, E192 / L3 / H4, RoPE, block 256, tied |
| recipe | Adam, lr 1e-3 cosine + 100 warmup, init 0.05, grad-clip 1.0 |

Verified: `python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt`
→ `{"bpb": 1.7416, "n_params": 1924800, "steps": 2000}`, and the same command
runs on an arbitrary mixed English/Hindi/emoji file (byte-fallback path).

## Overall

| milestone | dev bpb | cumulative Δ |
|-----------|---------|--------------|
| baseline (starter) | 2.3718 | — |
| best trainer recipe (r2, byte tok) | 2.2148 | −6.6% |
| **+ BPE tokenizer & capacity (final)** | **1.7416** | **−26.6%** |

**What I'd try next with more budget:** a clean RoPE-vs-learned ablation at BPE
scale; a small batch-size sweep (16/24) since bigger batches might let the LR go
higher; and a vocab-4096 run at a slightly narrower `n_embd` to keep under the
cap and test whether compression is still improving past 3072.

## Reproduce
```
python tokenizer.py --corpus ../data/train_corpus.txt --vocab 3072 --out bpe.json
python train.py --data ../data/train_corpus.txt --dev ../data/dev_eval.txt \
  --steps 2000 --batch 8 --pos rope --block 256 --n_embd 192 --n_layer 3 \
  --n_head 4 --tie 1 --init_std 0.05 --opt adam --lr 1e-3 --schedule cosine \
  --warmup 100 --out ckpt.pt
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```


