# 🦥 Unsloth Fine-Tuning: LLaMA 3.2B on FineTome-100k

This project demonstrates how to fine-tune a 4-bit quantized [LLaMA 3.2B Instruct](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct) model using the [Unsloth](https://github.com/unslothai/unsloth) framework, [LoRA adapters](https://arxiv.org/abs/2106.09685), and Hugging Face's [TRL](https://github.com/huggingface/trl) library.  

We use the `FineTome-100k` dataset (ShareGPT-style dialogues) for efficient **supervised fine-tuning (SFT)** with LoRA on a single NVIDIA A100 GPU.

---

## 🔑 Key Features

✅ **4-bit Quantized Fine-Tuning**  
Memory-efficient fine-tuning using Unsloth with `bitsandbytes`.

✅ **LoRA Adapter Integration**  
Parameter-efficient training (~0.75% of total parameters updated).

✅ **Unsloth + TRL (SFTTrainer)**  
Fast supervised fine-tuning with built-in evaluation and checkpointing.

✅ **Real Chat-Based Dataset (FineTome-100k)**  
ShareGPT-style multi-turn dialogues, formatted with LLaMA-3 chat template.

✅ **Built-in Evaluation**  
Tracks validation loss/perplexity during training and compares **Base vs Fine-tuned** perplexity after training.

✅ **GPU-Friendly**  
Supports gradient checkpointing, smart VRAM offload, `bf16/fp16` auto-detection.

---

## 📁 Project Structure

```
unsloth-finetuning-llama3/
├── finetune_unsloth_llama.py.py      # Main training script (LoRA fine-tuning with Unsloth + TRL)
├── evaluate.py                       # Standalone evaluation script (Base vs LoRA perplexity)
├── requirements.txt                  # Dependencies
├── README.md                         # Project description (you are here)
├── .gitignore                        
```

---

## 📦 Requirements

- Python ≥ 3.10
- CUDA-enabled GPU (A100 40GB recommended, V100 also works)
- PyTorch ≥ 2.0 with CUDA support

## 🚀 Quickstart

### 1. Clone this Repo

```bash
git clone https://github.com/yourusername/unsloth-finetuning-llama3.git
cd unsloth-finetuning-llama3
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Fine-Tuning

```bash
python finetune_unsloth_llama.py
```

This will:
- Download the `unsloth/Llama-3.2-3B-Instruct` base model  
- Preprocess the `mlabonne/FineTome-100k` dataset  
- Split 95/5 for training/validation  
- Attach LoRA adapters  
- Train for **2000 steps** with evaluation every 500 steps  
- Save adapters to `finetuned_model/`

### 4. Evaluation

Run:

```bash
python evaluate.py
```

This compares **Base vs LoRA-finetuned** model perplexity on the validation set with progress bars and summaries.

Example output:

```
=== Perplexity Comparison ===
Original Instruct Model Perplexity: 4.347
LoRA Fine-tuned Model Perplexity: 2.161
=== Final ===
Base PPL: 4.347 | LoRA PPL: 2.161 | Δ: -2.186
```

---

## 🧠 Model Details

| Item                     | Value                                     |
|--------------------------|-------------------------------------------|
| Base Model               | `unsloth/Llama-3.2-3B-Instruct` (4-bit)   |
| Params Trained (LoRA)    | ~24M (0.75% of 3.2B)                      |
| Dataset                  | `mlabonne/FineTome-100k` (ShareGPT style) |
| Training Steps           | 2000                                      |
| Effective Batch Size     | 16 (4 × grad acc 4 × 1 GPU)               |
| Validation Split         | 5%                                        |
| Eval Metric              | Perplexity (Base vs LoRA)                 |
| Hardware                 | NVIDIA A100 40GB                          |
| Final Perplexity (LoRA)  | ~2.16 (↓ from ~4.35 base)                 |

---

## 🛠️ Notes

- `unsloth_compiled_cache/`, `outputs/`, and `finetuned_model/` are ignored in `.gitignore`.  
- Training hyperparameters (learning rate, steps, batch size) can be tuned in `finetune.py`.  
- `evaluate.py` uses a **custom collate function** for correct padding/masking in 4-bit inference.  
- Future extensions: add generation-based evaluation (win-rate comparison, qualitative outputs).  
