#!/usr/bin/env bash
# Byte-tokenizer recipe ablation (cumulative, one change at a time).
# TOKMODE=byte forces the raw byte tokenizer so we isolate the TRAINER recipe
# from the tokenizer change that comes later.
set -e
cd "$(dirname "$0")"
export TOKMODE=byte
D=../data/train_corpus.txt
V=../data/dev_eval.txt
COMMON="--data $D --dev $V --steps 2000 --batch 8 --block 128 --log_every 500"

echo "=== r0b_control (reproduce baseline via new harness) ==="
python train.py $COMMON --tag r0b_control --lr 3e-4 --schedule constant --opt adam --init_std 0.05 --tie 0 --out /tmp/r.pt 2>&1 | grep -E "corpus|model:|DEV"

echo "=== r1_lr1e-3 ==="
python train.py $COMMON --tag r1_lr1e-3 --lr 1e-3 --schedule constant --opt adam --init_std 0.05 --tie 0 --out /tmp/r.pt 2>&1 | grep -E "DEV"

echo "=== r2_cosine_warmup ==="
python train.py $COMMON --tag r2_cosine --lr 1e-3 --schedule cosine --warmup 100 --opt adam --init_std 0.05 --tie 0 --out /tmp/r.pt 2>&1 | grep -E "DEV"

echo "=== r3_adamw_wd ==="
python train.py $COMMON --tag r3_adamw_wd --lr 1e-3 --schedule cosine --warmup 100 --opt adamw --wd 0.1 --init_std 0.05 --tie 0 --out /tmp/r.pt 2>&1 | grep -E "DEV"

echo "=== r4_gradclip ==="
python train.py $COMMON --tag r4_gradclip --lr 1e-3 --schedule cosine --warmup 100 --opt adamw --wd 0.1 --grad_clip 1.0 --init_std 0.05 --tie 0 --out /tmp/r.pt 2>&1 | grep -E "DEV"

echo "=== r5_init02 ==="
python train.py $COMMON --tag r5_init02 --lr 1e-3 --schedule cosine --warmup 100 --opt adamw --wd 0.1 --grad_clip 1.0 --init_std 0.02 --tie 0 --out /tmp/r.pt 2>&1 | grep -E "DEV"

echo "=== r6_tie ==="
python train.py $COMMON --tag r6_tie --lr 1e-3 --schedule cosine --warmup 100 --opt adamw --wd 0.1 --grad_clip 1.0 --init_std 0.02 --tie 1 --out /tmp/r.pt 2>&1 | grep -E "DEV"

echo "=== recipe sweep done ==="
