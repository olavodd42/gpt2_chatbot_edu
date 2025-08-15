import sys
import os
import yaml
import math
import bitsandbytes as bnb
from pathlib import Path
from datasets import load_dataset

# --- sys.path ---
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from src.modeling import load_model, create_data_collator, tokenize_dataset
from src.train_loop import create_trainer, train, evaluate, save_model_tokenizer, log_samples_wandb
from src.chat_loop import format_prompt

# --- 1) Carregar configs ---
with open("../configs/train_small.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

# --- 2) Carregar dataset ---
dataset = load_dataset(
    "json",
    data_files={
        "train": params["data"]["train_path"],
        "val":   params["data"]["val_path"],
    }
)

tokenizer, model = load_model()


def probe_quant(model):
    classes = set()
    for m in model.modules():
        mod = type(m).__module__
        if "bitsandbytes" in mod:
            classes.add(type(m).__name__)
    print("Camadas bnb presentes:", classes)

probe_quant(model)

total = 0
trainable = 0
for n, p in model.named_parameters():
    num = p.numel()
    total += num
    if p.requires_grad:
        trainable += num
print(f"Total params: {total/1e6:.1f}M | Trainable (LoRA): {trainable/1e6:.2f}M | Trainable rate: {trainable/total*100:.2f}%")

tokenized_dataset = tokenize_dataset(tokenizer, dataset, block_size=params["data"]["block_size"])

ex = tokenized_dataset["train"][0]
print("Keys:", ex.keys())
print("input_ids len:", len(ex["input_ids"]), "| attn len:", len(ex["attention_mask"]))


# --- 3) Criar e avaliar trainer modelo base ---
base_trainer = create_trainer(model, tokenized_dataset, tokenizer=tokenizer)
base_metrics = evaluate(base_trainer)
base_ppl = math.exp(base_metrics["eval_loss"]) if "eval_loss" in base_metrics else float("nan")
print(f"[BASE] eval_loss={base_metrics.get('eval_loss'):.4f} | ppl={base_ppl:.2f}")

# --- 4) Treinar ---
trainer = train(model, dataset, tokenizer=tokenizer)


# --- 5) Avaliar ---
final_metrics = evaluate(trainer)

final_ppl = math.exp(final_metrics["eval_loss"]) if "eval_loss" in final_metrics else float("nan")
print(f"[FINAL] eval_loss={final_metrics.get('eval_loss'):.4f} | ppl={final_ppl:.2f}")


# --- 6) Logar samples no W&B ---
sample_prompts = [
    format_prompt("What is the capital of France?"),
    format_prompt("Name three benefits of exercise."),
    format_prompt("Write a short poem about the ocean."),
]
log_samples_wandb(trainer, tokenizer, sample_prompts)

# --- 7) Salvar modelo/tokenizer ---
save_model_tokenizer(trainer, tokenizer)