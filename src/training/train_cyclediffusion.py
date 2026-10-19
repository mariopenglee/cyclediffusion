import argparse
import copy
import os
import random

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image

from models.cyclediffusion import CycleDiffusionModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--diffuser", default="OFA-Sys/small-stable-diffusion-v0")
    p.add_argument("--captioner", default="google/pix2struct-textcaps-base")
    p.add_argument("--dataset", default="jxie/coco_captions")
    p.add_argument("--image-col", default="image")
    p.add_argument("--text-col", default="caption")
    p.add_argument("--prompts", default=None, help="text file, one prompt per line: cycle-loss prompts are drawn from it "
                                                   "instead of the dataset captions (which still drive the denoising loss)")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lora-rank", type=int, default=0, help="train LoRA adapters on the UNet attention projections; 0 = full UNet")
    p.add_argument("--max-steps", type=int, default=2000, help="optimizer steps")
    p.add_argument("--denoise-weight", type=float, default=1.0, help="lambda on the denoising regularizer; 0 = pure cycle loss")
    p.add_argument("--sample-steps", type=int, default=20)
    p.add_argument("--grad-steps", type=int, default=1, help="K: number of final denoising steps to backprop through")
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--out", default="runs/cyclediff")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = CycleDiffusionModel(args.diffuser, args.captioner)
    model.diffuser.to(device)
    model.captioner.to(device)
    model.diffuser.unet.enable_gradient_checkpointing()
    if args.lora_rank:
        model.diffuser.unet = get_peft_model(model.diffuser.unet, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank, target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    model.diffuser.unet.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    tf = transforms.Compose([
        transforms.Resize(args.size), transforms.CenterCrop(args.size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])
    ds = load_dataset(args.dataset, split="train", streaming=True).shuffle(seed=args.seed, buffer_size=10_000)
    collate = lambda b: ([x[args.text_col] for x in b], torch.stack([tf(x[args.image_col].convert("RGB")) for x in b]))
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)

    prompts = open(args.prompts).read().splitlines() if args.prompts else None
    rng = random.Random(args.seed)

    writer = SummaryWriter(args.out)
    os.makedirs(args.out, exist_ok=True)
    step, micro = 0, 0
    for captions, images in loader:
        prompts_batch = rng.choices(prompts, k=len(captions)) if prompts else captions
        with torch.autocast(device, dtype=dtype, enabled=device == "cuda"):
            cycle_loss, denoise_loss, generated = model(
                prompts_batch, images.to(device) if args.denoise_weight > 0 else None, captions,
                size=args.size, steps=args.sample_steps, grad_steps=args.grad_steps, guidance=args.guidance,
            )
            loss = cycle_loss + args.denoise_weight * denoise_loss
        (loss / args.accumulation_steps).backward()
        micro += 1
        if micro % args.accumulation_steps:
            continue

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        writer.add_scalar("loss/cycle", cycle_loss.item(), step)
        writer.add_scalar("loss/denoise", denoise_loss.item(), step)
        print(f"step {step} cycle {cycle_loss.item():.4f} denoise {denoise_loss.item():.4f}", flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            unet = model.diffuser.unet
            # LoRA checkpoints are saved merged so evaluate.py loads them like a full UNet
            (copy.deepcopy(unet).merge_and_unload() if args.lora_rank else unet).save_pretrained(os.path.join(args.out, f"unet-{step}"))
            save_image(generated.detach().float(), os.path.join(args.out, f"samples-{step}.png"))
        if step == args.max_steps:
            break
    writer.close()


if __name__ == "__main__":
    main()
