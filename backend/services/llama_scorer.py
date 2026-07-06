from unsloth import FastLanguageModel
import torch
import os
from config import LLAMA_MODEL_PATH

_model = None
_tokenizer = None

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

def load_llama_model():
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    print(f"Loading Llama model from {LLAMA_MODEL_PATH}...")
    max_seq_length = 512
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = LLAMA_MODEL_PATH,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    
    _model = model
    _tokenizer = tokenizer
    print("Llama model loaded successfully.")
    return _model, _tokenizer

def score_email_llama(text: str) -> dict:
    if not text or not text.strip():
        return {"email_score": None, "error": "Empty input"}

    try:
        model, tokenizer = load_llama_model()
    except Exception as e:
        print(f"Error loading Llama model: {e}")
        return {"email_score": None, "error": f"Failed to load Llama model: {e}"}
        
    instruction = "Analyze the following email content and determine if it is a 'Safe Email' or a 'Phishing Email'."
    prompt = alpaca_prompt.format(instruction, text)
    
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens = 10, use_cache = True, pad_token_id=tokenizer.eos_token_id)
        
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
    response = decoded.split("### Response:\n")[-1].strip()
    
    # Map text response to scores
    if "Phishing Email" in response or "phishing" in response.lower():
        label = "phishing"
        phishing_score = 0.95
        safe_score = 0.05
    else:
        label = "safe"
        phishing_score = 0.05
        safe_score = 0.95
        
    confidence = phishing_score if label == "phishing" else safe_score
    
    return {
        "email_score": round(phishing_score, 4),
        "safe_score": round(safe_score, 4),
        "label": label,
        "confidence": round(confidence, 4)
    }
