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
from trl import SFTTrainer
from transformers import TrainingArguments

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.jsonl"
OUTPUT_DIR = BASE_DIR.parent / "models" / "SLM_adapter"

# Define the Prompt Format
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = "" # Will be set by model
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

def train():
    max_seq_length = 2048 # Increased sequence length to accommodate longer emails
    dtype = None # Auto detection
    load_in_4bit = True # Use 4bit quantization to reduce memory usage

    print("Loading model and tokenizer...")
    # Using local Llama-3.2-3B-Instruct-bnb-4bit checkpoint
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = str(BASE_DIR / "checkpoints"),
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    
    global EOS_TOKEN
    EOS_TOKEN = tokenizer.eos_token
    
    print("Setting up PEFT / LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8, # Rank 8 is good for general tasks and keeps memory low
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    print(f"Loading dataset from {DATA_PATH}...")
    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    print("Configuring SFTTrainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training faster for short sequences
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            num_train_epochs = 1, # Full training for 1 epoch
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            save_strategy = "no",
        ),
    )

    print("Starting training...")
    trainer_stats = trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Training finished and LoRA adapter saved.")

if __name__ == "__main__":
    train()

