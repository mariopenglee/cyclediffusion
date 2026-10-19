# usage: bash wordid_sweep.sh GPU RUN... ; RUN = base | STEP | STEP-train
export CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=.; shift
for r in "$@"; do
  s=${r%-train}; unet=""; [ "$s" != base ] && unet="--unet runs/text-p2s-lam0.1/unet-$s"
  prompts=data/text_prompts_heldout.txt; [ "$r" != "$s" ] && prompts=data/text_prompts_train.txt
  .venv/bin/python -u src/eval/word_id.py $unet --prompts $prompts --out runs/wordid/$r > runs/wordid-$r.log 2>&1
done
