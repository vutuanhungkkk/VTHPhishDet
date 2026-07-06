import os
from pathlib import Path
from unsloth import FastLanguageModel
import torch

BASE_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = BASE_DIR.parent / "models" / "SLM_adapter"

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

def predict_email(email_content):
    max_seq_length = 512
    dtype = None
    load_in_4bit = True

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = str(ADAPTER_DIR),
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
    except Exception as e:
        print(f"Failed to load adapter from {ADAPTER_DIR}: {e}")
        return

    FastLanguageModel.for_inference(model)
    
    instruction = "Analyze the following email content and determine if it is a 'Safe Email' or a 'Phishing Email'."
    
    prompt = alpaca_prompt.format(instruction, email_content)
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens = 10, use_cache = True, pad_token_id=tokenizer.eos_token_id)
    
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
    response = decoded.split("### Response:\\n")[-1].strip()
    
    return response

if __name__ == "__main__":
    print("SLM Phishing Detection Inference")
    print("================================")
    print("Enter email content below (Type 'exit' to quit):")
    
    while True:
        try:
            content = input("\nEmail Content> ")
            if content.strip().lower() == "exit":
                break
            
            if not content.strip():
                continue
                
            print("\nAnalyzing...")
            result = predict_email(content)
            print(f"Prediction: {result}")
            
        except KeyboardInterrupt:
            break

