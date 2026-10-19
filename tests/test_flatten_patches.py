"""The differentiable patcher must match Pix2Struct's own processor, otherwise the captioner sees off-distribution input."""
import numpy as np
import torch
from transformers import AutoProcessor

from models.cyclediffusion import Captioner


def test_matches_hf_processor():
    proc = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
    cap = Captioner.__new__(Captioner)
    cap.max_patches, cap.patch_size = 1024, 16
    for h, w in [(512, 512), (256, 384)]:
        img = (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)
        ref = proc(images=img, return_tensors="pt", max_patches=1024)
        ours, mask = cap.flatten_patches(torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255)
        assert torch.equal(mask, ref.attention_mask.float())
        assert torch.allclose(ours, ref.flattened_patches, atol=1e-4), (ours - ref.flattened_patches).abs().max()


if __name__ == "__main__":
    test_matches_hf_processor()
    print("ok")
