import pandas as pd
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROBERTA_DIR = BASE_DIR.parent / "train_roberta"

# Add the roberta directory to path so we can import from roberta_pipeline
sys.path.append(str(ROBERTA_DIR))
try:
    from roberta_pipeline import load_raw_data, clean_data, balance_data, split_data
except ImportError:
    print("Error: Could not import roberta_pipeline. Make sure you are in the correct directory.")
    sys.exit(1)

OUT_TRAIN_PATH = BASE_DIR / "data" / "train.jsonl"
OUT_TEST_PATH = BASE_DIR / "data" / "test.jsonl"

# Dùng toàn bộ 60k samples cho tập train như người dùng yêu cầu
LIMIT_TRAIN_SAMPLES = None 

def save_to_jsonl(X, y, filepath, instruction="Analyze the following email content and determine if it is a 'Safe Email' or a 'Phishing Email'."):
    records = []
    for text, label in zip(X, y):
        # label=0 is Safe, label=1 is Phishing
        out_str = "Safe Email" if label == 0 else "Phishing Email"
        records.append({
            "instruction": instruction,
            "input": str(text).strip(),
            "output": out_str
        })
        
    print(f"Saving {len(records)} samples to {filepath}...")
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def process_data():
    print("Loading and splitting data identical to RoBERTa pipeline...")
    
    # Run the exact same steps as RoBERTa
    df = load_raw_data()
    df_clean = clean_data(df)
    df_balanced = balance_data(df_clean)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_balanced)
    
    print(f"RoBERTa original split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    if LIMIT_TRAIN_SAMPLES and LIMIT_TRAIN_SAMPLES < len(X_train):
        print(f"Limiting train samples to {LIMIT_TRAIN_SAMPLES} to save time...")
        # Since X_train and y_train are lists, we zip, sample, and unzip
        import random
        combined = list(zip(X_train, y_train))
        random.seed(42)
        sampled = random.sample(combined, LIMIT_TRAIN_SAMPLES)
        X_train_final, y_train_final = zip(*sampled)
    else:
        X_train_final, y_train_final = X_train, y_train

    print(f"SLM Final Split -> Train: {len(X_train_final)}, Test: {len(X_test)}")
    
    # Save test set (12852 samples, identical to RoBERTa's test set)
    save_to_jsonl(X_test, y_test, OUT_TEST_PATH)
    
    # Save train set
    save_to_jsonl(X_train_final, y_train_final, OUT_TRAIN_PATH)
    
    print("Done!")

if __name__ == "__main__":
    process_data()
