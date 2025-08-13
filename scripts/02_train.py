import sys
import os
import yaml
import math
import bitsandbytes as bnb
from pathlib import Path
from datasets import load_dataset

# Adiciona o diretório raiz do projeto ao path (prioriza no início para evitar conflitos com pacotes instalados chamados 'src')
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from src.modeling import load_model, create_data_collator, tokenize_dataset
from src.train_loop import ModelTrainer

dataset = load_dataset(
    "json",
    data_files={
        "train": "../data/processed/train.jsonl",
        "val":   "../data/processed/val.jsonl",
        "test":   "../data/processed/test.jsonl"
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

# após carregar:
probe_quant(model)

total = 0
trainable = 0
for n, p in model.named_parameters():
    num = p.numel()
    total += num
    if p.requires_grad:
        trainable += num
print(f"Total params: {total/1e6:.1f}M | Trainable (LoRA): {trainable/1e6:.2f}M | Trainable rate: {trainable/total*100:.2f}%")

tokenized_dataset = tokenize_dataset(tokenizer, dataset)

example = tokenized_dataset["train"][0]
print("Keys:", example.keys())
print("input_ids:", example["input_ids"])
print("attention_mask:", example["attention_mask"])

trainer = ModelTrainer(model, tokenized_dataset, tokenizer)
base_metrics = trainer.evaluate()
trainer.train()
metrics = trainer.evaluate()

print("Base Model Eval loss:", base_metrics["eval_loss"])
print("Base Model Perplexity:", math.exp(base_metrics["eval_loss"]))
print("Trained Model Eval loss:", metrics["eval_loss"])
print("Trained Model Perplexity:", math.exp(metrics["eval_loss"]))