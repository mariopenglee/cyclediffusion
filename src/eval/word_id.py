"""
Who can read the rendered word? For prompts "a sign that says 'word'", ask each reader to pick the word out of the
prompt file's vocabulary: captioners by lowest NLL of the prompt with each candidate word swapped in, Pix2Struct also
by greedy caption, EasyOCR by exact match. Text the reward captioner reads but OCR does not is the hack we look for.
Same seeds as evaluate.py, so the images match its sample grids.
"""
import argparse
import json
import os
import re

import easyocr
import torch
from diffusers import UNet2DConditionModel
from transformers import AutoModelForCausalLM, AutoProcessor

from models.cyclediffusion import BlipCaptioner, Captioner, Diffuser
from src.eval.evaluate import git_nll, word_scores

WORD = re.compile(r"'([^']+)'")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unet", default=None)
    p.add_argument("--diffuser", default="OFA-Sys/small-stable-diffusion-v0")
    p.add_argument("--prompts", default="data/text_prompts_heldout.txt")
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    device = "cuda"

    diffuser = Diffuser(args.diffuser)
    if args.unet:
        diffuser.unet = UNet2DConditionModel.from_pretrained(args.unet)
    diffuser.to(device).eval()
    p2s, blip_large = Captioner().to(device), BlipCaptioner().to(device)
    git = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco").to(device).eval()
    git_proc = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    readers = {"p2s": p2s, "blipL": blip_large, "git": lambda im, c: git_nll(git, git_proc, im, c)}

    prompts = open(args.prompts).read().splitlines()
    vocab = sorted({WORD.search(c).group(1) for c in prompts})
    prompts = prompts[: args.n]

    rows = []
    with torch.no_grad():
        for i in range(0, len(prompts), args.batch_size):
            caps = prompts[i: i + args.batch_size]
            images = diffuser.sample(caps, steps=25, guidance=7.5, generator=torch.Generator().manual_seed(args.seed + i))
            flat, mask = p2s.flatten_patches(images)
            greedy = p2s.tokenizer.batch_decode(p2s.model.generate(flattened_patches=flat, attention_mask=mask, max_new_tokens=30),
                                                skip_special_tokens=True)
            for img, cap, g in zip(images, caps, greedy):
                word = WORD.search(cap).group(1)
                cands = [WORD.sub(f"'{w}'", cap) for w in vocab]
                row = dict(caption=cap, word=word, p2s_greedy=g, p2s_greedy_hit=word in g.lower())
                for name, fn in readers.items():
                    nll = fn(img[None].expand(len(cands), -1, -1, -1), cands)
                    row[f"{name}_rank"] = int((nll < nll[vocab.index(word)]).sum())  # 0 = picked the right word
                arr = (img.permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
                texts = [t for _, t, _ in reader.readtext(arr)]
                row["ocr_text"], row["ocr_hit"] = " | ".join(texts), word_scores(word, texts)[0]
                rows.append(row)
            print(f"{len(rows)}/{len(prompts)}", flush=True)

    summary = {f"{k}_top1": sum(r[f"{k}_rank"] == 0 for r in rows) / len(rows) for k in readers}
    summary |= {k: sum(r[k] for r in rows) / len(rows) for k in ["p2s_greedy_hit", "ocr_hit"]}
    summary["p2s_top1_not_ocr"] = sum(r["p2s_rank"] == 0 and not r["ocr_hit"] for r in rows) / len(rows)
    summary["chance"] = 1 / len(vocab)
    print(json.dumps(summary, indent=1))
    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(args=vars(args), vocab=vocab, summary=summary, rows=rows), open(f"{args.out}/word_id.json", "w"), indent=1)


if __name__ == "__main__":
    main()
