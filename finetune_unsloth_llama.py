"""
Fine-tuning LLaMA 3.2 (3B) using Unsloth + TRL SFTTrainer on the FineTome-100k dataset.

This script performs LoRA-based supervised fine-tuning of a 4-bit quantized 
LLaMA 3.2 model using HuggingFace's TRL and Unsloth. It uses the FineTome-100k dataset, 
formatted in ShareGPT style, for training.

It also includes evaluation:
1. Splits off 5% for validation
2. Tracks validation loss/perplexity during training
3. Compares base vs LoRA perplexity after training
"""

import os
#os.environ["HF_DATASETS_NUM_PROC"] = "2"

import math
import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


# ------------------- Load Pretrained LLaMA Model -------------------
model_name = "unsloth/Llama-3.2-3B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    load_in_4bit=True,
)

# ------------------- Add LoRA Adapters -------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# ------------------- Load + Preprocess Dataset -------------------
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = standardize_sharegpt(dataset)

# Train/validation split (95/5)
dataset_split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

# Map conversations -> "text"
def format_examples(examples):
    return {
        "text": [
            tokenizer.apply_chat_template(convo, tokenize=False)
            for convo in examples["conversations"]
        ]
    }

train_dataset = train_dataset.map(format_examples, batched=True, num_proc=2)
eval_dataset = eval_dataset.map(format_examples, batched=True, num_proc=2)

# ------------------- Training Configuration -------------------
training_args = SFTConfig(
    output_dir="outputs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=2000,
    learning_rate=1e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",          # or "steps" if your build supports it
    eval_steps=500,                 # only if using "steps"
    load_best_model_at_end=True,   
    dataset_text_field="text",
    dataset_num_proc=2,
    max_length=2048,
    remove_unused_columns=False,
)

# ------------------- Initialize Trainer -------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)

# ------------------- Train -------------------
trainer.train()

# ------------------- Save LoRA Model -------------------
model.save_pretrained("finetuned_model")