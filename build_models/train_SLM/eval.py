import os
from pathlib import Path
import ssl
import certifi
def _patched_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    self.load_verify_locations(cafile=certifi.where())
ssl.SSLContext.load_default_certs = _patched_load_default_certs
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import json

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "test.jsonl"
ADAPTER_DIR = BASE_DIR.parent / "models" / "SLM_adapter"
RESULTS_DIR = BASE_DIR.parent / "results" / "SLM"

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

def evaluate():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True
    
    print("Loading model and tokenizer with LoRA adapters...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = str(ADAPTER_DIR), # Load the LoRA adapter
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
    except Exception as e:
        print(f"Failed to load adapter from {ADAPTER_DIR}: {e}")
        print("Please make sure you have run train.py first.")
        return
        
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
    print(f"Loading dataset from {DATA_PATH}...")
    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")
    
    # Evaluate on the entire test set (12852 samples identical to RoBERTa)
    test_dataset = dataset
    
    y_true = []
    y_pred = []
    
    print("Running evaluation...")
    for item in tqdm(test_dataset):
        instruction = item["instruction"]
        input_text = item["input"]
        actual_output = item["output"]
        
        prompt = alpaca_prompt.format(instruction, input_text)
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens = 10, use_cache = True, pad_token_id=tokenizer.eos_token_id)
        
        # Decode and parse out the generated response
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
        # The output includes the prompt, so we split to get the response part
        response = decoded.split("### Response:\n")[-1].strip()
        
        # Heuristic parsing of response
        if "Phishing Email" in response or "Phishing" in response:
            pred = "Phishing Email"
        elif "Safe Email" in response or "Safe" in response:
            pred = "Safe Email"
        else:
            pred = "Unknown"
            
        y_true.append(actual_output.strip())
        y_pred.append(pred)
        
    print("\n--- Evaluation Results ---")
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)
    
    # Save results
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    with open(RESULTS_DIR / "eval_report.json", "w") as f:
        json.dump(report_dict, f, indent=4)
        
    print(f"Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    evaluate()

