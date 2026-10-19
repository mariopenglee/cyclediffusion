#!/bin/sh
# LoRA (rank 8, lr 1e-4) pilot over the denoise weight; each run is evaluated at every checkpoint
export CUDA_VISIBLE_DEVICES=5 PYTHONPATH=.
for lam in 0 0.1 1; do
  .venv/bin/python -u src/training/train_cyclediffusion.py --lora-rank 8 --lr 1e-4 --denoise-weight $lam \
    --max-steps 1000 --save-every 250 --out runs/lora-p2s-lam$lam > runs/lora-p2s-lam$lam.log 2>&1
  for s in 250 500 750 1000; do
    .venv/bin/python -u src/eval/evaluate.py --unet runs/lora-p2s-lam$lam/unet-$s --out runs/eval/lora-p2s-lam$lam-$s > runs/eval-lora-p2s-lam$lam-$s.log 2>&1
  done
done
