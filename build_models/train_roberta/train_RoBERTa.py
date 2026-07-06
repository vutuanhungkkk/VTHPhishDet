from __future__ import annotations

from pathlib import Path

import torch

print("Loading libraries and initialising... Please wait.")

from roberta_pipeline import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_DIR,
    RESULTS_DIR,
    STATE_DICT_PATH,
    WEIGHT_DECAY,
    balance_data,
    build_dataloaders,
    build_model,
    build_tokenizer,
    clean_data,
    eval_epoch,
    evaluate_model,
    get_device,
    load_raw_data,
    maybe_save_plots,
    save_model_artifacts,
    split_data,
    train_epoch,
    write_metrics_summary,
)
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def main():
    device = get_device()
    print(f"Device: {device}")

    df = load_raw_data()
    print(f"Shape after loading: {df.shape}")

    df = clean_data(df)
    print(f"Shape after cleaning: {df.shape}")
    print(df["label"].value_counts())

    df_balanced = balance_data(df)
    print(f"Balanced dataset shape: {df_balanced.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_balanced)
    print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")

    tokenizer = build_tokenizer()
    train_loader, val_loader, test_loader = build_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, batch_size=BATCH_SIZE
    )

    model = build_model().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), STATE_DICT_PATH)
            print(f"Best checkpoint saved -> {STATE_DICT_PATH}")

    if STATE_DICT_PATH.exists():
        model.load_state_dict(torch.load(STATE_DICT_PATH, map_location=device))

    save_model_artifacts(model, tokenizer, MODEL_DIR, STATE_DICT_PATH)
    metrics = evaluate_model(model, test_loader, device)
    print(metrics["classification_report"])

    maybe_save_plots(history, metrics, RESULTS_DIR)
    write_metrics_summary(metrics, RESULTS_DIR / "metrics_summary.txt", len(X_train), len(X_val), len(X_test))

    print("Training complete.")
    print(f"Model dir : {MODEL_DIR}")
    print(f"State dict : {STATE_DICT_PATH}")


if __name__ == "__main__":
    main()
