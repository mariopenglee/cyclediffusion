# BLIP-large lambda=0 moved to GPU 6 (was queued behind lambda=0.1 on GPU 5); base-v2 runs alongside training
export CUDA_VISIBLE_DEVICES=6 PYTHONPATH=.
PY=.venv/bin/python
ev() { $PY -u src/eval/evaluate.py "$@" > runs/eval-$(basename "${@: -1}").log 2>&1; }
ev --out runs/eval/base-v2 &
$PY -u src/training/train_cyclediffusion.py --lora-rank 8 --lr 1e-4 --max-steps 1000 --save-every 250 --denoise-weight 0 \
  --captioner Salesforce/blip-image-captioning-large --out runs/lora-blip-lam0 > runs/lora-blip-lam0.log 2>&1
for s in 250 500 750 1000; do ev --unet runs/lora-blip-lam0/unet-$s --out runs/eval/lora-blip-lam0-$s; done
wait
