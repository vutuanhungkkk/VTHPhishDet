from __future__ import annotations

import math
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tldextract
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
from urllib.parse import urlparse
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Malicious URLs dataset" / "malicious_phish.csv"
MODEL_PATH = BASE_DIR.parent / "models" / "xgboost_phishing_model.pkl"
RESULTS_DIR = BASE_DIR.parent / "results" / "XGBoost"

FEATURE_COLUMNS = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_digits",
    "ratio_digits",
    "num_special_chars",
    "num_dots",
    "num_hyphens",
    "num_subdomains",
    "has_ip",
    "has_at",
    "is_shortened",
    "has_suspicious_words",
    "url_entropy",
    "domain_length",
    "tld_length",
]


def load_raw_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if 'type' in frame.columns:
        frame['label'] = frame['type'].map(lambda x: 0 if str(x).strip().lower() == 'benign' else 1)
    else:
        frame = frame.drop(columns=["Unnamed: 0", "result"], errors="ignore")
        if 'label' in frame.columns and frame['label'].dtype == object:
            frame["label"] = frame["label"].map({"benign": 0, "malicious": 1})
            
    frame = frame.dropna(subset=["url", "label"])
    frame["url"] = frame["url"].astype(str)
    frame["label"] = frame["label"].astype(int)
    return frame.reset_index(drop=True)


def entropy(s: str) -> float:
    if not s:
        return 0.0
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
    return -sum(p * math.log2(p) for p in prob)


def extract_features(url: str) -> dict:
    try:
        url = str(url)
        if not url.startswith("http"):
            url = "http://" + url

        parsed = urlparse(url)
        ext = tldextract.extract(url)

        url_len = len(url)
        return {
            "url_length": url_len,
            "hostname_length": len(parsed.netloc),
            "path_length": len(parsed.path),
            "num_digits": sum(c.isdigit() for c in url),
            "ratio_digits": sum(c.isdigit() for c in url) / max(url_len, 1),
            "num_special_chars": sum(c in "@#$%^&*()-_=+[]{}|;:,.<>?" for c in url),
            "num_dots": url.count("."),
            "num_hyphens": url.count("-"),
            "num_subdomains": len(ext.subdomain.split(".")) if ext.subdomain else 0,
            "has_ip": int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))),
            "has_at": int("@" in url),
            "is_shortened": int(any(short in ext.domain.lower() for short in ["bit", "t", "tinyurl", "goo", "ow", "is", "buff"])),
            "has_suspicious_words": int(any(w in url.lower() for w in ["login", "verify", "bank", "secure", "paypal", "amazon", "admin", "update", "free", "bonus", "account", "webscr", "ebank", "service"])),
            "url_entropy": entropy(url),
            "domain_length": len(ext.domain),
            "tld_length": len(ext.suffix),
        }
    except Exception:
        return {k: 0 for k in FEATURE_COLUMNS}


def build_feature_matrix(df: pd.DataFrame):
    feature_list = df["url"].apply(extract_features)
    X = pd.DataFrame(feature_list.tolist())
    y = df["label"].astype(int)
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def build_model(scale_pos_weight: float):
    return XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )


def evaluate_model(model, X_test, y_test, threshold: float = 0.7):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_probs),
        "classification_report": classification_report(y_test, y_pred, target_names=["benign", "malicious"], zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_probs": y_probs,
    }
    return metrics


def predict_url(url: str, model) -> dict:
    features = extract_features(url)
    df = pd.DataFrame([features])
    prob = float(model.predict_proba(df)[0][1])
    label = "Phishing" if prob > 0.7 else "Benign"
    return {"label": label, "probability": prob, "features": features}


def save_model(model, model_path: Path = MODEL_PATH):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)


def load_model(model_path: Path = MODEL_PATH):
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def write_metrics_summary(metrics: dict, output_path: Path, train_size: int, test_size: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("XGBoost URL Classifier - Results\n")
        handle.write("=" * 50 + "\n")
        handle.write(f"Accuracy  : {metrics['accuracy']:.4f}\n")
        handle.write(f"Precision : {metrics['precision']:.4f}\n")
        handle.write(f"Recall    : {metrics['recall']:.4f}\n")
        handle.write(f"F1 Score  : {metrics['f1']:.4f}\n")
        handle.write(f"ROC-AUC   : {metrics['roc_auc']:.4f}\n")
        handle.write("=" * 50 + "\n")
        handle.write(f"Train size: {train_size}\n")
        handle.write(f"Test size : {test_size}\n")


def maybe_save_confusion_matrix(metrics: dict, output_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
