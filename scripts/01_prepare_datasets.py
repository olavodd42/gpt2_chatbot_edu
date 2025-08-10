import sys
import os
from pathlib import Path

# Adiciona o diretório raiz do projeto ao path (prioriza no início para evitar conflitos com pacotes instalados chamados 'src')
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_prep import load_dataset_hf, convert_dataset, concatenate_datasets_
from src.templates import json_to_text

# Carrega datasets do Hugging Face
alpaca = load_dataset_hf("tatsu-lab/alpaca", sample_len=10)
dolly = load_dataset_hf("databricks/databricks-dolly-15k", sample_len=10)
sciq = load_dataset_hf("sciq", sample_len=10)

# Converte datasets no formato de chat
alpaca = convert_dataset(alpaca, name="alpaca")
dolly = convert_dataset(dolly, name="dolly")
sciq = convert_dataset(sciq, name="sciq")

# Concatena datasets
splits = concatenate_datasets_([alpaca, dolly, sciq])

Path("../data/processed").mkdir(parents=True, exist_ok=True)
splits["train"].to_json("../data/processed/train.jsonl", orient="records", lines=True, force_ascii=False)
splits["val"].to_json("../data/processed/val.jsonl", orient="records", lines=True, force_ascii=False)
splits["test"].to_json("../data/processed/test.jsonl", orient="records", lines=True, force_ascii=False)

print(json_to_text("train.jsonl"))
