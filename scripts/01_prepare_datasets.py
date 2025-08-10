import sys
import os
from pathlib import Path

# Adiciona o diretório raiz do projeto ao path (prioriza no início para evitar conflitos com pacotes instalados chamados 'src')
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_prep import load_dataset_hf, convert_dataset, concatenate_datasets_, clean_dataset
from src.templates import json_to_text

# Carrega datasets do Hugging Face
alpaca = load_dataset_hf("tatsu-lab/alpaca", sample_len=10)
dolly = load_dataset_hf("databricks/databricks-dolly-15k", sample_len=10)
sciq = load_dataset_hf("sciq", sample_len=10)

# Converte datasets no formato de chat
alpaca = convert_dataset(alpaca, name="alpaca")
dolly = convert_dataset(dolly, name="dolly")
sciq = convert_dataset(sciq, name="sciq")

# Concatena e duvude datasets
splits = concatenate_datasets_([alpaca, dolly, sciq])
splits = clean_dataset(splits)
# Converte datasets em JSON
Path("../data/interim").mkdir(parents=True, exist_ok=True)
splits["train"].to_json("../data/interim/train.jsonl", orient="records", lines=True, force_ascii=False)
splits["val"].to_json("../data/interim/val.jsonl", orient="records", lines=True, force_ascii=False)
splits["test"].to_json("../data/interim/test.jsonl", orient="records", lines=True, force_ascii=False)

# Converte JSON no formato de texto padrão do GPT-2
txt_train = json_to_text("train.jsonl")
txt_val = json_to_text("val.jsonl")
txt_test = json_to_text("test.jsonl")

Path("../data/processed").mkdir(parents=True, exist_ok=True)
with open("../data/processed/train.txt", 'w') as f:
    f.write(txt_train)

with open("../data/processed/val.txt", 'w') as f:
    f.write(txt_val)

with open("../data/processed/test.txt", 'w') as f:
    f.write(txt_test)
