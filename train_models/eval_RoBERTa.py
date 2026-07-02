from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification

from roberta_pipeline import (
    balance_data,
    MODEL_DIR,
    RESULTS_DIR,
    build_dataloaders,
    build_tokenizer,
    clean_data,
    evaluate_model,
    load_raw_data,
    maybe_save_plots,
    split_data,
    write_metrics_summary,
)


def main():
    df = clean_data(load_raw_data())
    df_balanced = balance_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_balanced)
    tokenizer = build_tokenizer()
    _, _, test_loader = build_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

    metrics = evaluate_model(model, test_loader, device)
    print(metrics["classification_report"])
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    maybe_save_plots({"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}, metrics, RESULTS_DIR)
    write_metrics_summary(metrics, RESULTS_DIR / "metrics_summary.txt", len(X_train), len(X_val), len(X_test))


if __name__ == "__main__":
    main()
