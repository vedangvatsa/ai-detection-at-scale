#!/usr/bin/env python3
"""Wrapper for the fine-tuned TuringBench RoBERTa-large detector."""
import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DEFAULT_MODEL_DIR = os.path.join(PROJECT_DIR, 'models', 'turingbench_roberta_large')

_model = None
_tokenizer = None
_device = None


def _get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def load_model(model_dir: str = DEFAULT_MODEL_DIR):
    """Load the fine-tuned TuringBench model lazily."""
    global _model, _tokenizer, _device
    if _model is None:
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Fine-tuned TuringBench model not found at {model_dir}. "
                "Train it with scripts/33_finetune_turingbench.py or download it from Kaggle."
            )
        _device = _get_device()
        _tokenizer = AutoTokenizer.from_pretrained(model_dir)
        _model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        _model.to(_device)
        _model.eval()
    return _tokenizer, _model, _device


def predict_ai_probability(text: str, model_dir: str = DEFAULT_MODEL_DIR) -> float:
    """Return the probability that `text` is AI-generated."""
    tokenizer, model, device = load_model(model_dir)
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=256,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # label 1 = AI
        return float(probs[0][1].cpu())


def predict_batch(texts: list[str], model_dir: str = DEFAULT_MODEL_DIR, batch_size: int = 32) -> list[float]:
    """Return AI probabilities for a list of texts."""
    tokenizer, model, device = load_model(model_dir)
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            results.extend(probs[:, 1].cpu().tolist())
    return results


def classify(text: str, threshold: float = 0.5, model_dir: str = DEFAULT_MODEL_DIR) -> dict:
    """Return label, probability, and confidence for a single text."""
    prob = predict_ai_probability(text, model_dir=model_dir)
    label = 'AI' if prob >= threshold else 'Human'
    confidence = max(prob, 1 - prob)
    return {
        'text': text,
        'ai_probability': prob,
        'label': label,
        'confidence': confidence,
    }


def _read_input(path_or_text: str) -> str:
    if os.path.isfile(path_or_text):
        with open(path_or_text, 'r', encoding='utf-8') as f:
            return f.read()
    return path_or_text


def main():
    parser = argparse.ArgumentParser(
        description='Detect AI-generated text using the fine-tuned TuringBench RoBERTa-large model.'
    )
    parser.add_argument(
        'input',
        help='Text to classify, or path to a .txt file. Use - to read from stdin.',
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default=DEFAULT_MODEL_DIR,
        help='Path to the fine-tuned model directory.',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Probability threshold for AI classification.',
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Treat input as a file with one text per line and classify all lines.',
    )
    args = parser.parse_args()

    if args.input == '-':
        raw = sys.stdin.read()
    else:
        raw = _read_input(args.input)

    if args.batch:
        texts = [line.strip() for line in raw.splitlines() if line.strip()]
        probs = predict_batch(texts, model_dir=args.model_dir)
        for text, prob in zip(texts, probs):
            label = 'AI' if prob >= args.threshold else 'Human'
            print(f'{prob:.4f}\t{label}\t{text[:120]}')
    else:
        result = classify(raw, threshold=args.threshold, model_dir=args.model_dir)
        print(f"Label:        {result['label']}")
        print(f"AI probability: {result['ai_probability']:.4f}")
        print(f"Confidence:   {result['confidence']:.4f}")


if __name__ == '__main__':
    main()
