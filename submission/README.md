# 2,000-Step LLM Speedrun — submission

**Result: 1.7416 dev bits-per-byte** (baseline 2.3718, **−26.6%**),
1,924,800 params (< 2M cap), 2,000 steps (= cap), CPU-only, corpus-only.

## Score it (the exact graded command)
```
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
# -> {"bpb": 1.7416, "n_params": 1924800, "steps": 2000, ...}
```

## Files
| file | what |
|------|------|
| `ckpt.pt` | final checkpoint (config + step count + loss curve) |
| `model.py` | GPT: E192 / L3 / H4, RoPE, weight tying, GPT-2 residual-scaled init |
| `tokenizer.py` | byte-level **BPE** (vocab 3072), lossless, byte fallback; `bpe.json` = trained merges |
| `train.py` | trainer: AdamW/Adam/**Lion**, warmup+cosine, grad-clip, dev-bpb readout |
| `evaluate.py` | official scorer, **unchanged** from the starter |
| `RUNLOG.md` | every run: hypothesis → change → dev bpb before/after → conclusion |
| `NOTES.md` | best config and why, in ≤10 sentences |
| `SUMMARY.html` | one-page summary of the above + the machine-vs-human split |
| `runs.csv` | machine-written record of all ~15 runs |
| `experiments/` | sweep scripts, per-phase logs, vocab variants, baseline ckpt |

## One-line story
The corpus is ~33% Devanagari *by bytes*, so a byte tokenizer wastes the step
budget; a corpus-trained BPE tokenizer (3.7 bytes/token) is the biggest lever
(−0.46 bpb). A higher LR + cosine schedule and spending the param budget on
vocabulary (compression) rather than depth do the rest. The measured ladder also
showed the "obvious" AdamW+weight-decay and 0.02-init *hurt* in this <1-epoch
regime, and an ambitious Lion-optimizer swap lost to Adam — both documented and
diagnosed in `RUNLOG.md`.
