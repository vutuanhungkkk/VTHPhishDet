import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "The Biggest Spam Ham Phish Email Dataset" / "df.csv"
if not DATA_PATH.exists():
    # Fallback
    DATA_PATH = BASE_DIR / "data" / "The Biggest Spam Ham Phish Email Dataset" / "df.csv"

LABEL_MAP = {"Safe Email": 0, "Phishing Email": 1}

def main():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH, engine="python", on_bad_lines="skip")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    raw_total = len(df)
    
    # Cleaning as in the training pipeline
    rename_cols = {}
    if "Email Text" in df.columns: rename_cols["Email Text"] = "text"
    if "Email Type" in df.columns: rename_cols["Email Type"] = "label"
    frame = df.rename(columns=rename_cols).copy()
    
    frame = frame.dropna(subset=["text", "label"])
    frame = frame[frame["text"].astype(str).str.strip() != ""]
    frame = frame[frame["text"].astype(str).str.lower() != "empty"]
    
    import re
    def clean_text(text):
        text = str(text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'(?i)(Message-ID|Date|From|To|Subject|MIME-Version|Content-Type|Content-Transfer-Encoding|Bcc):.*?(?=\n|$)', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    from tqdm import tqdm
    tqdm.pandas(desc="Cleaning text (Regex)")
    frame["text"] = frame["text"].progress_apply(clean_text)
    frame = frame[frame["text"].str.strip() != ""]
    
    frame["label_str"] = frame["label"].astype(str).str.strip()
    def map_label(val):
        if val in ["0", "1", "0.0", "1.0"]: return int(float(val))
        return LABEL_MAP.get(val, None)
        
    tqdm.pandas(desc="Mapping labels")
    frame["label_enc"] = frame["label_str"].progress_apply(map_label)
    frame = frame.dropna(subset=["label_enc"])
    frame["label_enc"] = frame["label_enc"].astype(int)
    frame = frame.reset_index(drop=True)

    clean_total = len(frame)
    label_counts = frame["label_enc"].value_counts().to_dict()

    # Balancing as in the training pipeline
    safe = frame[frame["label_enc"] == 0]
    phishing = frame[frame["label_enc"] == 1]
    min_count = min(len(safe), len(phishing), 50000)
    safe_balanced = safe.sample(n=min_count, random_state=42)
    phishing_balanced = phishing.sample(n=min_count, random_state=42)
    balanced = pd.concat([safe_balanced, phishing_balanced])
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    balanced_total = len(balanced)
    balanced_counts = balanced["label_enc"].value_counts().to_dict()

    # Format the data statistics
    output_lines = [
        "RoBERTa Training Data Statistics",
        "================================",
        f"Raw data total rows: {raw_total}",
        f"Cleaned data total rows: {clean_total}",
        "",
        "Cleaned Label Distribution (0: Safe Email, 1: Phishing Email):",
        f"  Safe Email (0): {label_counts.get(0, 0)} ({(label_counts.get(0, 0)/clean_total)*100:.2f}%)",
        f"  Phishing Email (1): {label_counts.get(1, 0)} ({(label_counts.get(1, 0)/clean_total)*100:.2f}%)",
        "",
        f"Balanced data total rows: {balanced_total}",
        "Balanced Label Distribution:",
        f"  Safe Email (0): {balanced_counts.get(0, 0)} ({(balanced_counts.get(0, 0)/balanced_total)*100:.2f}%)",
        f"  Phishing Email (1): {balanced_counts.get(1, 0)} ({(balanced_counts.get(1, 0)/balanced_total)*100:.2f}%)",
        "",
        "Columns after cleaning:",
        ", ".join(balanced.columns.tolist()),
        "",
        "Sample Data Format (first 3 rows of balanced data):",
        balanced.head(3).to_string(index=False)
    ]

    out_file = BASE_DIR / "roberta_data_stats.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    
    print(f"Successfully wrote statistics to {out_file}")

if __name__ == "__main__":
    main()
