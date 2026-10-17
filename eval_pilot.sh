#!/bin/sh
# Evaluate the base model, then every pilot checkpoint once training (tmux session "cyclediff") finishes
export CUDA_VISIBLE_DEVICES=5 PYTHONPATH=.
.venv/bin/python -u src/eval/evaluate.py --out runs/eval/base > runs/eval-base.log 2>&1
while tmux has-session -t cyclediff 2>/dev/null; do sleep 60; done
for lam in 0 1; do for s in 250 500 750 1000; do
  .venv/bin/python -u src/eval/evaluate.py --unet runs/pilot-p2s-lam$lam/unet-$s --out runs/eval/p2s-lam$lam-$s > runs/eval-p2s-lam$lam-$s.log 2>&1
done; done
