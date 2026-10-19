"""BlipCaptioner's differentiable preprocessing must score like BLIP's own processor (evaluate.blip_nll)."""
import importlib.util

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from models.cyclediffusion import BlipCaptioner

spec = importlib.util.spec_from_file_location("evaluate", "src/eval/evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def test_matches_processor_nll():
    model_id = "Salesforce/blip-image-captioning-base"
    cap = BlipCaptioner(model_id)
    images = torch.rand(2, 3, 512, 512, generator=torch.Generator().manual_seed(0))
    captions = ["a dog on a couch", "a sign that says 'coffee'"]
    with torch.no_grad():
        ours = cap(images, captions)
        ref = evaluate.blip_nll(cap.model, BlipProcessor.from_pretrained(model_id), images, captions)
    assert torch.allclose(ours, ref, atol=0.05), (ours, ref)


def test_word_scores():
    assert evaluate.word_scores("coffee", ["COFFEE!", "shop"]) == (True, 1.0)
    exact, sim = evaluate.word_scores("coffee", ["cofee"])
    assert not exact and 0.8 < sim < 1
    assert evaluate.word_scores("coffee", []) == (False, 0.0)


if __name__ == "__main__":
    test_word_scores()
    test_matches_processor_nll()
    print("ok")
