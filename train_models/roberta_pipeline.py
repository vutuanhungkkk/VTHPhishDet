from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Phishing_Email.csv"
MODEL_DIR = BASE_DIR / "models" / "RoBERTa_model"
STATE_DICT_PATH = BASE_DIR / "models" / "best_roberta_phishing.pt"
RESULTS_DIR = BASE_DIR / "results" / "RoBERTa"

MODEL_NAME = "roberta-base"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LABEL_MAP = {"Safe Email": 0, "Phishing Email": 1}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_raw_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path, engine="python", on_bad_lines="skip")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.rename(columns={"Email Text": "text", "Email Type": "label"}).copy()
    frame = frame.dropna(subset=["text", "label"])
    frame = frame[frame["text"].astype(str).str.strip() != ""]
    frame = frame[frame["text"].astype(str).str.lower() != "empty"]
    frame["label"] = frame["label"].astype(str).str.strip()
    frame["label_enc"] = frame["label"].map(LABEL_MAP)
    frame = frame.dropna(subset=["label_enc"])
    frame["label_enc"] = frame["label_enc"].astype(int)
    return frame.reset_index(drop=True)


def balance_data(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    safe = df[df["label_enc"] == 0]
    phishing = df[df["label_enc"] == 1]
    min_count = min(len(safe), len(phishing))
    safe_balanced = safe.sample(n=min_count, random_state=seed)
    phishing_balanced = phishing.sample(n=min_count, random_state=seed)
    balanced = pd.concat([safe_balanced, phishing_balanced])
    return balanced.sample(frac=1, random_state=seed).reset_index(drop=True)


def split_data(df_balanced: pd.DataFrame, seed: int = 42):
    X = df_balanced["text"].astype(str).tolist()
    y = df_balanced["label_enc"].astype(int).tolist()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = MAX_LEN):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_tokenizer(model_name: str = MODEL_NAME):
    return RobertaTokenizer.from_pretrained(model_name)


def build_dataloaders(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    tokenizer,
    batch_size: int = BATCH_SIZE,
):
    train_dataset = EmailDataset(X_train, y_train, tokenizer)
    val_dataset = EmailDataset(X_val, y_val, tokenizer)
    test_dataset = EmailDataset(X_test, y_test, tokenizer)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def build_model(model_name: str = MODEL_NAME, num_labels: int = 2):
    return RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def eval_epoch(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = outputs.logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].tolist())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs),
        "classification_report": classification_report(
            all_labels,
            all_preds,
            target_names=["Safe Email", "Phishing Email"],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
    }
    return metrics


def predict_email(text: str, model, tokenizer, device, max_len: int = MAX_LEN):
    model.eval()
    enc = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())

    label_map_inv = {0: "Safe Email", 1: "Phishing Email"}
    return label_map_inv[pred], probs


def save_model_artifacts(model, tokenizer, save_dir: Path = MODEL_DIR, state_dict_path: Path = STATE_DICT_PATH):
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save(model.state_dict(), state_dict_path)


def write_metrics_summary(metrics: dict, output_path: Path, train_size: int, val_size: int, test_size: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("RoBERTa Phishing Email Classifier - Results\n")
        handle.write("=" * 50 + "\n")
        handle.write(f"Accuracy  : {metrics['accuracy']:.4f}\n")
        handle.write(f"Precision : {metrics['precision']:.4f}\n")
        handle.write(f"Recall    : {metrics['recall']:.4f}\n")
        handle.write(f"F1 Score  : {metrics['f1']:.4f}\n")
        handle.write(f"ROC-AUC   : {metrics['roc_auc']:.4f}\n")
        handle.write("=" * 50 + "\n")
        handle.write(f"Train size: {train_size}\n")
        handle.write(f"Val size  : {val_size}\n")
        handle.write(f"Test size : {test_size}\n")


def maybe_save_plots(history: dict, metrics: dict, output_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(history["val_loss"], label="Val Loss", marker="o")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train Acc", marker="o")
    ax2.plot(history["val_acc"], label="Val Acc", marker="o")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    cm = metrics["confusion_matrix"]
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Safe", "Phishing"], yticklabels=["Safe", "Phishing"])
    plt.title("Confusion Matrix - Test Set")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    fpr = None
    tpr = None
    try:
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(metrics["labels"], metrics["probs"])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier")
        plt.fill_between(fpr, tpr, alpha=0.1, color="darkorange")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - RoBERTa Phishing Detector")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png", dpi=150)
        plt.close()
    except Exception:
        pass
