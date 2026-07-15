#!/usr/bin/env bash
# Phase 3: final checkpoint (best config) + the ambitious "new optimizer"
# experiment (Lion). Default bpe.json is the vocab-3072 winner.
# Best config: Adam, lr 1e-3, cosine+warmup100, init0.05, tie, rope, block256,
# E192, L3  -> b3 = 1.7416 dev bpb.
set -e
cd "$(dirname "$0")"
D=../data/train_corpus.txt
V=../data/dev_eval.txt
CFG="--data $D --dev $V --steps 2000 --batch 8 --pos rope --block 256 --n_embd 192 --n_head 4 --n_layer 3 --tie 1 --init_std 0.05 --warmup 100 --schedule cosine --log_every 500"

echo "=== FINAL (best config -> ckpt.pt) ==="
python train.py $CFG --opt adam --lr 1e-3 --seed 1337 --tag final --out ckpt.pt 2>&1 | grep -E "corpus|model:|DEV|saved"

echo "=== lion_lr1e-3 (ambitious: Lion at Adam's LR, expect it to LOSE) ==="
python train.py $CFG --opt lion --lr 1e-3 --seed 1337 --tag lion_lr1e-3 --out /tmp/l.pt 2>&1 | grep -E "DEV"

echo "=== lion_lr2e-4 (the FIX: Lion needs a much smaller LR) ==="
python train.py $CFG --opt lion --lr 2e-4 --seed 1337 --tag lion_lr2e-4 --out /tmp/l.pt 2>&1 | grep -E "DEV"

echo "=== phase 3 done ==="
