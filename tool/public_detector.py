#!/usr/bin/env python3
"""Wrapper for public pre-trained AI detectors."""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS = {
    "roberta-openai": ("roberta-base-openai-detector", 0),      # label 0 = fake
    "chatgpt-detector": ("Hello-SimpleAI/chatgpt-detector-roberta", 1),  # label 1 = fake
}

_cache = {}

def _get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_public_detector(name: str = "roberta-openai"):
    """Load a public detector by short name."""
    if name in _cache:
        return _cache[name]
    if name not in MODELS:
        raise ValueError(f"Unknown detector {name}. Choose from {list(MODELS.keys())}")
    model_name, ai_label = MODELS[name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = _get_device()
    model.to(device)
    model.eval()
    _cache[name] = (tokenizer, model, device, ai_label)
    return _cache[name]

def predict_ai_probability(text: str, name: str = "roberta-openai") -> float:
    """Return probability that text is AI-generated."""
    tokenizer, model, device, ai_label = load_public_detector(name)
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            return float(probs[0][ai_label].cpu())
    except Exception as e:
        raise RuntimeError(f"Public detector inference failed: {e}")

def predict_ensemble(text: str) -> dict:
    """Return probabilities from both public detectors."""
    return {
        "roberta_openai": predict_ai_probability(text, "roberta-openai"),
        "chatgpt_detector": predict_ai_probability(text, "chatgpt-detector"),
    }
