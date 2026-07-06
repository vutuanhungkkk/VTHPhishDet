import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Malicious URLs dataset" / "malicious_phish.csv"
if not DATA_PATH.exists():
    # Fallback
    DATA_PATH = BASE_DIR / "data" / "Malicious URLs dataset" / "malicious_phish.csv"

def main():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Raw data stats
    raw_total = len(df)
    
    # Cleaning as in the training pipeline
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
    frame = frame.reset_index(drop=True)

    clean_total = len(frame)
    label_counts = frame["label"].value_counts().to_dict()
    
    # Format the data statistics
    output_lines = [
        "XGBoost Training Data Statistics",
        "================================",
        f"Raw data total rows: {raw_total}",
        f"Cleaned data total rows: {clean_total}",
        "",
        "Label Distribution (0: Benign, 1: Malicious):",
        f"  Benign (0): {label_counts.get(0, 0)} ({(label_counts.get(0, 0)/clean_total)*100:.2f}%)",
        f"  Malicious (1): {label_counts.get(1, 0)} ({(label_counts.get(1, 0)/clean_total)*100:.2f}%)",
        "",
        "Columns after cleaning:",
        ", ".join(frame.columns.tolist()),
        "",
        "Sample Data Format (first 5 rows):",
        frame.head(5).to_string(index=False)
    ]

    out_file = BASE_DIR / "xgboost_data_stats.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    
    print(f"Successfully wrote statistics to {out_file}")

if __name__ == "__main__":
    main()
