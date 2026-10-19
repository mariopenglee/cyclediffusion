# Two pipelines sharing GPU 5 (at most one process each):
#  text: can the Pix2Struct (TextCaps) reward teach the generator to render a quoted word? held-out words test letters vs memorization
#  blip: same LoRA setup with a captioner that actually captions (BLIP-large) as the reward, lambda 0.1 then 0
export CUDA_VISIBLE_DEVICES=5 PYTHONPATH=.
PY=.venv/bin/python
LORA="--lora-rank 8 --lr 1e-4 --max-steps 1000 --save-every 250"
ev() { $PY -u src/eval/evaluate.py "$@" > runs/eval-$(basename "${@: -1}").log 2>&1; }

text() {
  ev --prompts data/text_prompts_heldout.txt --n 120 --out runs/eval/base-text-heldout
  ev --prompts data/text_prompts_train.txt --n 120 --out runs/eval/base-text-train
  $PY -u src/training/train_cyclediffusion.py $LORA --denoise-weight 0.1 --prompts data/text_prompts_train.txt \
    --out runs/text-p2s-lam0.1 > runs/text-p2s-lam0.1.log 2>&1
  for s in 250 500 750 1000; do
    ev --unet runs/text-p2s-lam0.1/unet-$s --prompts data/text_prompts_heldout.txt --n 120 --out runs/eval/text-p2s-lam0.1-$s-heldout
    ev --unet runs/text-p2s-lam0.1/unet-$s --prompts data/text_prompts_train.txt --n 120 --out runs/eval/text-p2s-lam0.1-$s-train
  done
}

blip() {
  for lam in 0.1 0; do
    $PY -u src/training/train_cyclediffusion.py $LORA --denoise-weight $lam --captioner Salesforce/blip-image-captioning-large \
      --out runs/lora-blip-lam$lam > runs/lora-blip-lam$lam.log 2>&1
    for s in 250 500 750 1000; do ev --unet runs/lora-blip-lam$lam/unet-$s --out runs/eval/lora-blip-lam$lam-$s; done
  done
  ev --out runs/eval/base-v2  # base on COCO with the new GIT / BLIP-large columns
}

text & blip & wait
