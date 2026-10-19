"""
Score a UNet checkpoint on held-out COCO prompts: the training reward (Pix2Struct NLL) next to held-out
evaluators (BLIP and GIT NLL, CLIPScore) and an OCR check for the "write the caption into the image" hack.
With --prompts of the form "a sign that says 'word'", also scores whether the quoted word is legibly rendered.
"""
import argparse
import difflib
import json
import os
import re

import easyocr
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import UNet2DConditionModel
from torchvision.utils import save_image
from transformers import (AutoModelForCausalLM, AutoProcessor, BlipForConditionalGeneration, BlipProcessor, CLIPModel,
                          CLIPProcessor)

from models.cyclediffusion import BlipCaptioner, Captioner, Diffuser


def blip_nll(model, processor, images, captions):
    """Per-caption mean token NLL under BLIP. images: (B, 3, H, W) in [0, 1]"""
    inputs = processor(images=list(images.float().cpu()), text=captions, return_tensors="pt", padding=True,
                       do_rescale=False).to(images.device)
    input_ids = inputs.input_ids.clone()
    input_ids[:, 0] = model.config.text_config.bos_token_id  # BLIP decodes from [DEC], not [CLS]
    logits = model(pixel_values=inputs.pixel_values, input_ids=input_ids, attention_mask=inputs.attention_mask).logits
    labels = input_ids[:, 1:].masked_fill(inputs.attention_mask[:, 1:] == 0, -100)
    nll = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
    return nll.sum(1) / (labels != -100).sum(1)


def git_nll(model, processor, images, captions):
    """Per-caption mean token NLL under GIT (image tokens are prepended, so only the trailing text logits count)."""
    inputs = processor(images=list(images.float().cpu()), text=captions, return_tensors="pt", padding=True,
                       do_rescale=False).to(images.device)
    logits = model(**inputs).logits[:, -inputs.input_ids.shape[1]:]
    labels = inputs.input_ids[:, 1:].masked_fill(inputs.attention_mask[:, 1:] == 0, -100)
    nll = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
    return nll.sum(1) / (labels != -100).sum(1)


def clip_score(model, processor, images, captions):
    inputs = processor(images=list(images.float().cpu()), text=captions, return_tensors="pt", padding=True,
                       truncation=True, do_rescale=False).to(images.device)
    out = model(**inputs)
    return 100 * F.cosine_similarity(out.image_embeds, out.text_embeds).clamp(min=0)


def word_scores(target, texts):
    """(exact, similarity) of a quoted target word against OCR strings; similarity is the best difflib ratio."""
    norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
    target, texts = norm(target), [norm(t) for t in texts]
    return target in "".join(texts), max([difflib.SequenceMatcher(None, target, t).ratio() for t in texts], default=0.0)


def ocr_stats(reader, images, captions):
    """
    (has_text, text_matches_caption, word_match, word_sim) per image; words shorter than 3 chars are ignored for the
    first two. The last two score the 'quoted' word in the caption (None when there is none), over all detections.
    """
    has_text, matches, word_match, word_sim = [], [], [], []
    for img, cap in zip(images, captions):
        arr = (img.permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
        dets = reader.readtext(arr)
        words = {w.lower() for _, text, conf in dets if conf > 0.5 for w in text.split() if len(w) >= 3}
        has_text.append(bool(words))
        matches.append(bool(words & {w.lower().strip(".,") for w in cap.split()}))
        quoted = re.search(r"'([^']+)'", cap)
        exact, sim = word_scores(quoted.group(1), [t for _, t, _ in dets]) if quoted else (None, None)
        word_match.append(exact)
        word_sim.append(sim)
    return has_text, matches, word_match, word_sim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unet", default=None, help="checkpoint dir from training; omit for the base model")
    p.add_argument("--diffuser", default="OFA-Sys/small-stable-diffusion-v0")
    p.add_argument("--prompts", default=None, help="text file, one prompt per line, instead of COCO validation captions")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output dir for metrics.json and a sample grid")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    diffuser = Diffuser(args.diffuser)
    if args.unet:
        diffuser.unet = UNet2DConditionModel.from_pretrained(args.unet)
    diffuser.to(device).eval()
    captioner = Captioner().to(device)
    blip_large = BlipCaptioner().to(device)  # the alternative training reward
    blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device).eval()
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    git = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco").to(device).eval()
    git_proc = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    reader = easyocr.Reader(["en"], gpu=device == "cuda", verbose=False)

    if args.prompts:
        captions = open(args.prompts).read().splitlines()[: args.n]
    else:
        # one caption per image (the split has 5 captions per image, consecutive)
        stream = load_dataset("jxie/coco_captions", split="validation", streaming=True).select_columns(["caption"])
        captions = [r["caption"] for r in stream.take(5 * args.n)][::5]
    args.n = len(captions)

    rows = []
    with torch.no_grad():
        for i in range(0, args.n, args.batch_size):
            caps = captions[i: i + args.batch_size]
            images = diffuser.sample(caps, size=args.size, steps=args.steps, guidance=args.guidance,
                                     generator=torch.Generator().manual_seed(args.seed + i))
            ocr = ocr_stats(reader, images, caps)
            for c, a, bl, b, g, s, t, m, wm, ws in zip(caps, captioner(images, caps), blip_large(images, caps),
                                                       blip_nll(blip, blip_proc, images, caps), git_nll(git, git_proc, images, caps),
                                                   clip_score(clip, clip_proc, images, caps), *ocr):
                rows.append(dict(caption=c, p2s_nll=a.item(), blipL_nll=bl.item(), blip_nll=b.item(), git_nll=g.item(), clip=s.item(),
                                 ocr_text=t, ocr_match=m, word_match=wm, word_sim=ws))
            if i == 0:
                os.makedirs(args.out, exist_ok=True)
                save_image(images.float(), os.path.join(args.out, "samples.png"))
            print(f"{i + len(caps)}/{args.n}", flush=True)

    keys = ["p2s_nll", "blipL_nll", "blip_nll", "git_nll", "clip", "ocr_text", "ocr_match", "word_match", "word_sim"]
    summary = {k: float(np.mean(v)) for k in keys if (v := [r[k] for r in rows if r[k] is not None])}
    print(json.dumps(summary, indent=1))
    json.dump(dict(args=vars(args), summary=summary, rows=rows), open(os.path.join(args.out, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
