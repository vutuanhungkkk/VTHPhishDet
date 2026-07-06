from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from roberta_pipeline import MODEL_DIR, predict_email


def load_model(model_dir: Path = MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser(description="Infer phishing probability using the trained RoBERTa model.")
    parser.add_argument("--text", help="Email text to classify.")
    parser.add_argument("--file", help="Text file containing email content.")
    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("Provide either --text or --file")

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = args.text

    model, tokenizer, device = load_model()
    label, probs = predict_email(text, model, tokenizer, device)
    print(f"Prediction : {label}")
    print(f"Safe prob  : {float(probs[0]):.4f}")
    print(f"Phish prob : {float(probs[1]):.4f}")


if __name__ == "__main__":
    main()
