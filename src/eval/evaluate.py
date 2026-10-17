"""
Score a UNet checkpoint on held-out COCO prompts: the training reward (Pix2Struct NLL) next to held-out
evaluators (BLIP NLL, CLIPScore) and an OCR check for the "write the caption into the image" hack.
"""
import argparse
import json
import os

import easyocr
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import UNet2DConditionModel
from torchvision.utils import save_image
from transformers import BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor

from models.cyclediffusion import Captioner, Diffuser


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


def clip_score(model, processor, images, captions):
    inputs = processor(images=list(images.float().cpu()), text=captions, return_tensors="pt", padding=True,
                       truncation=True, do_rescale=False).to(images.device)
    out = model(**inputs)
    return 100 * F.cosine_similarity(out.image_embeds, out.text_embeds).clamp(min=0)


def ocr_stats(reader, images, captions):
    """(has_text, text_matches_caption) per image; words shorter than 3 chars are ignored."""
    has_text, matches = [], []
    for img, cap in zip(images, captions):
        arr = (img.permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
        words = {w.lower() for _, text, conf in reader.readtext(arr) if conf > 0.5 for w in text.split() if len(w) >= 3}
        has_text.append(bool(words))
        matches.append(bool(words & {w.lower().strip(".,") for w in cap.split()}))
    return has_text, matches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unet", default=None, help="checkpoint dir from training; omit for the base model")
    p.add_argument("--diffuser", default="OFA-Sys/small-stable-diffusion-v0")
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
    blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device).eval()
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    reader = easyocr.Reader(["en"], gpu=device == "cuda", verbose=False)

    # one caption per image (the split has 5 captions per image, consecutive)
    stream = load_dataset("jxie/coco_captions", split="validation", streaming=True).select_columns(["caption"])
    captions = [r["caption"] for r in stream.take(5 * args.n)][::5]

    rows = []
    with torch.no_grad():
        for i in range(0, args.n, args.batch_size):
            caps = captions[i: i + args.batch_size]
            images = diffuser.sample(caps, size=args.size, steps=args.steps, guidance=args.guidance,
                                     generator=torch.Generator().manual_seed(args.seed + i))
            has_text, matches = ocr_stats(reader, images, caps)
            for c, a, b, s, t, m in zip(caps, captioner(images, caps), blip_nll(blip, blip_proc, images, caps),
                                        clip_score(clip, clip_proc, images, caps), has_text, matches):
                rows.append(dict(caption=c, p2s_nll=a.item(), blip_nll=b.item(), clip=s.item(), ocr_text=t, ocr_match=m))
            if i == 0:
                os.makedirs(args.out, exist_ok=True)
                save_image(images.float(), os.path.join(args.out, "samples.png"))
            print(f"{i + len(caps)}/{args.n}", flush=True)

    summary = {k: float(np.mean([r[k] for r in rows])) for k in ["p2s_nll", "blip_nll", "clip", "ocr_text", "ocr_match"]}
    print(json.dumps(summary, indent=1))
    json.dump(dict(args=vars(args), summary=summary, rows=rows), open(os.path.join(args.out, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
