#!/usr/bin/env bash
# Phase 2. Recipe is the empirically-best point from the ladder (= r2):
#   Adam (NOT AdamW), lr 1e-3, cosine + warmup 100, init_std 0.05, no wd/clip.
# The ladder showed AdamW+wd0.1 and init0.02 both REGRESS in this <1-epoch
# regime, so we do NOT carry them forward. First a clean tie ablation on the
# byte base, then the BPE depth-vs-vocab trade-off under the 2M cap (rope => a
# 256-token block costs no params).
set -e
cd "$(dirname "$0")"
D=../data/train_corpus.txt
V=../data/dev_eval.txt
R2="--lr 1e-3 --schedule cosine --warmup 100 --opt adam --init_std 0.05"
COMMON="--data $D --dev $V --steps 2000 --batch 8 --log_every 500"

echo "=== r7_tie_on_r2 (clean tie ablation on the GOOD base, byte tok) ==="
TOKMODE=byte python train.py $COMMON $R2 --n_embd 160 --n_layer 4 --n_head 4 \
  --block 128 --pos learned --tie 1 --tag r7_tie_on_r2 --out /tmp/p.pt 2>&1 | grep -E "DEV"

BPE="$R2 --pos rope --block 256 --n_embd 192 --n_head 4 --tie 1"

echo "=== b1_v2048_L3 ==="
BPE_JSON=bpe.json      python train.py $COMMON $BPE --n_layer 3 --tag b1_v2048_L3 --out /tmp/p.pt 2>&1 | grep -E "corpus|model:|DEV"

echo "=== b2_v1024_L4 ==="
BPE_JSON=bpe_1024.json python train.py $COMMON $BPE --n_layer 4 --tag b2_v1024_L4 --out /tmp/p.pt 2>&1 | grep -E "corpus|model:|DEV"

echo "=== b3_v3072_L3 ==="
BPE_JSON=bpe_3072.json python train.py $COMMON $BPE --n_layer 3 --tag b3_v3072_L3 --out /tmp/p.pt 2>&1 | grep -E "corpus|model:|DEV"

echo "=== phase 2 done ==="
