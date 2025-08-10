import sys
import os
from pathlib import Path

# Adiciona o diretório raiz do projeto ao path (prioriza no início para evitar conflitos com pacotes instalados chamados 'src')
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from src.modeling import load_model

tokenizer, model = load_model()

total = 0
trainable = 0
for n, p in model.named_parameters():
    num = p.numel()
    total += num
    if p.requires_grad:
        trainable += num
print(f"Total params: {total/1e6:.1f}M | Trainable (LoRA): {trainable/1e6:.2f}M | Trainable rate: {trainable/total*100:.2f}%")