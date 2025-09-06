"""
Evaluate Base vs LoRA-finetuned LLaMA 3.2 (3B) on FineTome-100k (perplexity).
Works with 4-bit quantized models (no Trainer). Masks padding from loss.
Adds rich logging + tqdm progress bars.
"""

import os, math, time, torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from tqdm.auto import tqdm

# ------------------- Config -------------------
os.environ["HF_DATASETS_NUM_PROC"] = "2"
MODEL_NAME   = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_DIR  = "finetuned_model"     # where you saved model.save_pretrained(...)
MAX_LEN      = 2048
BATCH_SIZE   = 4
NUM_WORKERS  = 2
EVAL_FRACTION = 0.05                 # 5% validation split
MAX_SAMPLES  = None                  # e.g. 1000 for a quick run; or None for full eval
LOG_EVERY    = 50                    # update tqdm postfix every N batches

torch.set_grad_enabled(False)

# ------------------- Load tokenizer -------------------
print(">> Loading tokenizer…")
_, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_LEN,
    load_in_4bit=True,   # only to get tokenizer quickly
)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------- Load & preprocess dataset -------------------
print(">> Loading dataset (FineTome-100k)…")
ds = load_dataset("mlabonne/FineTome-100k", split="train")
ds = standardize_sharegpt(ds)

print(f">> Splitting eval fraction: {EVAL_FRACTION*100:.1f}%")
split = ds.train_test_split(test_size=EVAL_FRACTION, seed=42)
eval_ds = split["test"]
if MAX_SAMPLES is not None:
    eval_ds = eval_ds.select(range(min(MAX_SAMPLES, len(eval_ds))))
print(f">> Eval examples: {len(eval_ds):,}")

def to_text(examples):
    return {
        "text": [
            tokenizer.apply_chat_template(conv, tokenize=False)
            for conv in examples["conversations"]
        ]
    }

print(">> Formatting with chat template…")
eval_ds = eval_ds.map(to_text, batched=True, num_proc=2)

# ------------------- Collate (tokenize + pad + mask pads in labels) -------------------
def make_collate(tokenizer, max_len):
    def collate(batch):
        texts = [ex["text"] for ex in batch]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        labels = enc["input_ids"].clone()
        # mask out padding so it doesn't affect loss
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc
    return collate

collate_fn = make_collate(tokenizer, MAX_LEN)


def perplexity(model, dataset, label=""):
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    print(f">> Evaluating [{label}] … batches: {len(loader)}, bs={BATCH_SIZE}, max_len={MAX_LEN}")

    total_loss, steps = 0.0, 0
    total_tokens = 0

    pbar = tqdm(loader, desc=f"Eval {label}", dynamic_ncols=True)
    with torch.inference_mode():
        for i, batch in enumerate(pbar, 1):
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = float(out.loss)
            total_loss += loss
            steps += 1
            total_tokens += int(batch["attention_mask"].sum().item())
            if i % LOG_EVERY == 0:
                avg = total_loss / steps
                pbar.set_postfix(loss=f"{avg:.4f}", ppl=f"{math.exp(avg):.2f}", toks=total_tokens)

    avg_loss = total_loss / max(steps, 1)
    ppl = math.exp(avg_loss)
    print(f"\n[{label}] Summary")
    print(f"  • Batches:      {steps:,}")
    print(f"  • Tokens:       {total_tokens:,}")
    print(f"  • Avg loss:     {avg_loss:.4f}")
    print(f"  • Perplexity:   {ppl:.3f}\n")
    return ppl

print("\n=== Perplexity Comparison ===")

# Base (no LoRA)
print(">> Loading BASE model…")
base_model, _ = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_LEN,
    load_in_4bit=True,
)
base_ppl = perplexity(base_model, eval_ds, label="Original Instruct Model")

# LoRA-finetuned (load base then attach adapter)
print(">> Loading LoRA model and attaching adapters…")
lora_model, _ = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_LEN,
    load_in_4bit=True,
)
lora_model.load_adapter(ADAPTER_DIR)
lora_ppl = perplexity(lora_model, eval_ds, label="LoRA Fine-tuned Model")

print("=== Final ===")
print(f"Base PPL: {base_ppl:.3f} | LoRA PPL: {lora_ppl:.3f} | Δ: {base_ppl - lora_ppl:+.3f}")