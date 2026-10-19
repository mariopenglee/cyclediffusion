import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import (AutoProcessor, BlipForConditionalGeneration, BlipProcessor, CLIPTextModel, CLIPTokenizer,
                          Pix2StructForConditionalGeneration)


class Captioner(nn.Module):
    """
    Frozen Pix2Struct captioner used as a differentiable critic: given an image and a caption,
    returns the per-caption mean token NLL of that caption. Gradients flow back into the image.
    """
    def __init__(self, model_id="google/pix2struct-textcaps-base", max_patches=1024, patch_size=16, max_length=48):
        super().__init__()
        self.model = Pix2StructForConditionalGeneration.from_pretrained(model_id).requires_grad_(False).eval()
        self.tokenizer = AutoProcessor.from_pretrained(model_id).tokenizer
        self.max_patches, self.patch_size, self.max_length = max_patches, patch_size, max_length

    def flatten_patches(self, images):
        """
        Differentiable, batched version of Pix2StructImageProcessor (normalize + extract_flattened_patches).

        Args:
            images (torch.Tensor): (B, C, H, W) in [0, 1]

        Returns:
            torch.Tensor: (B, max_patches, 2 + patch_size * patch_size * C)
            torch.Tensor: attention mask (B, max_patches)
        """
        B, C, H, W = images.shape
        p, n = self.patch_size, self.max_patches
        # per-image standardization, with the processor's std floor
        mean = images.mean(dim=(1, 2, 3), keepdim=True)
        std = images.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp(min=1 / math.sqrt(C * H * W))
        images = (images - mean) / std

        scale = math.sqrt(n * (p / H) * (p / W))
        rows = max(min(math.floor(scale * H / p), n), 1)
        cols = max(min(math.floor(scale * W / p), n), 1)
        images = F.interpolate(images, size=(rows * p, cols * p), mode="bilinear", align_corners=False, antialias=True)

        # (B, C*p*p, L) -> (B, L, p, p, C) -> (B, L, p*p*C), matching the processor's (row, col, channel) order
        patches = F.unfold(images, p, stride=p).view(B, C, p, p, rows * cols)
        patches = patches.permute(0, 4, 2, 3, 1).reshape(B, rows * cols, p * p * C)

        # 1-indexed row/col ids; 0 means padding
        row_ids = torch.arange(rows, device=images.device).repeat_interleave(cols) + 1
        col_ids = torch.arange(cols, device=images.device).repeat(rows) + 1
        ids = torch.stack([row_ids, col_ids], -1).to(patches.dtype).expand(B, -1, -1)

        flattened = F.pad(torch.cat([ids, patches], -1), (0, 0, 0, n - rows * cols))
        attention_mask = torch.zeros(B, n, device=images.device)
        attention_mask[:, : rows * cols] = 1
        return flattened, attention_mask

    def forward(self, images, captions):
        labels = self.tokenizer(captions, padding="max_length", truncation=True, max_length=self.max_length,
                                return_tensors="pt").input_ids.to(images.device)
        labels[labels == self.tokenizer.pad_token_id] = -100
        flattened, attention_mask = self.flatten_patches(images)
        logits = self.model(flattened_patches=flattened, attention_mask=attention_mask, labels=labels).logits
        nll = F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
        return nll.sum(1) / (labels != -100).sum(1)


class BlipCaptioner(nn.Module):
    """
    Frozen BLIP captioner as a differentiable critic, same interface as Captioner: per-caption mean token NLL.
    """
    def __init__(self, model_id="Salesforce/blip-image-captioning-large", max_length=48):
        super().__init__()
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).requires_grad_(False).eval()
        processor = BlipProcessor.from_pretrained(model_id)
        self.tokenizer, self.max_length = processor.tokenizer, max_length
        self.size = processor.image_processor.size["height"]
        self.register_buffer("mean", torch.tensor(processor.image_processor.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(processor.image_processor.image_std).view(1, 3, 1, 1), persistent=False)

    def forward(self, images, captions):
        pixels = F.interpolate(images, size=(self.size, self.size), mode="bicubic", align_corners=False, antialias=True)
        pixels = (pixels - self.mean) / self.std
        tok = self.tokenizer(captions, padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt").to(images.device)
        input_ids = tok.input_ids.clone()
        input_ids[:, 0] = self.model.config.text_config.bos_token_id  # BLIP decodes from [DEC], not [CLS]
        logits = self.model(pixel_values=pixels, input_ids=input_ids, attention_mask=tok.attention_mask).logits
        labels = input_ids[:, 1:].masked_fill(tok.attention_mask[:, 1:] == 0, -100)
        nll = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
        return nll.sum(1) / (labels != -100).sum(1)


class Diffuser(nn.Module):
    """
    Stable Diffusion with a trainable UNet; text encoder and VAE are frozen.
    """
    def __init__(self, model_id="OFA-Sys/small-stable-diffusion-v0"):
        super().__init__()
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").requires_grad_(False)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").requires_grad_(False)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

    def encode_text(self, captions):
        ids = self.tokenizer(captions, padding="max_length", max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids
        return self.text_encoder(ids.to(self.unet.device))[0]

    def denoise_loss(self, captions, images):
        """Standard epsilon-prediction loss on real (caption, image) pairs. images: (B, 3, H, W) in [-1, 1]"""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
            cond = self.encode_text(captions)
        noise = torch.randn_like(latents)
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (len(latents),), device=latents.device)
        pred = self.unet(self.scheduler.add_noise(latents, noise, t), t, cond).sample
        return F.mse_loss(pred.float(), noise.float())

    def sample(self, captions, size=512, steps=20, grad_steps=1, guidance=7.5, generator=None):
        """
        DDIM sampling with classifier-free guidance. Only the last `grad_steps` UNet calls and the VAE decode
        are differentiable (truncated backprop, as in DRaFT-K). Returns images (B, 3, size, size) in [0, 1].
        """
        B, device = len(captions), self.unet.device
        with torch.no_grad():
            cond = torch.cat([self.encode_text([""] * B), self.encode_text(captions)])
        shape = (B, self.unet.config.in_channels, size // 8, size // 8)
        latents = torch.randn(shape, generator=generator).to(device)

        self.scheduler.set_timesteps(steps, device=device)
        for i, t in enumerate(self.scheduler.timesteps):
            with torch.set_grad_enabled(torch.is_grad_enabled() and i >= steps - grad_steps):
                eps_uncond, eps_cond = self.unet(torch.cat([latents] * 2), t, cond).sample.chunk(2)
                eps = eps_uncond + guidance * (eps_cond - eps_uncond)
                latents = self.scheduler.step(eps, t, latents).prev_sample
        images = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        return (images / 2 + 0.5).clamp(0, 1)


class CycleDiffusionModel(nn.Module):
    """
    text -> image -> text cycle: the diffuser generates an image from a caption, the frozen captioner scores
    how likely the original caption is given that image, and that NLL is backpropagated into the UNet.
    The standard denoising loss on real pairs acts as a regularizer against reward hacking.
    """
    def __init__(self, diffuser_id="OFA-Sys/small-stable-diffusion-v0", captioner_id="google/pix2struct-textcaps-base"):
        super().__init__()
        self.diffuser = Diffuser(diffuser_id)
        self.captioner = (BlipCaptioner if "blip" in captioner_id.lower() else Captioner)(captioner_id)

    def forward(self, captions, images=None, image_captions=None, **sample_kwargs):
        """image_captions: captions of `images` for the denoising loss, when they differ from the cycle prompts"""
        generated = self.diffuser.sample(captions, **sample_kwargs)
        cycle_loss = self.captioner(generated, captions).mean()
        denoise_loss = (self.diffuser.denoise_loss(image_captions or captions, images) if images is not None
                        else cycle_loss.new_zeros(()))
        return cycle_loss, denoise_loss, generated

    def parameters(self, recurse: bool = True):
        return self.diffuser.unet.parameters(recurse)
