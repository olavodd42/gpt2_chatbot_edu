import sys
import os
from pathlib import Path

# Adiciona o diretório raiz do projeto ao path (prioriza no início para evitar conflitos com pacotes instalados chamados 'src')
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_prep import load_dataset_hf, convert_dataset, concatenate_datasets_, clean_dataset, save_text_jsonl
from src.templates import json_to_text, dialogue_to_text


def as_text_dataset(ds):
    # converte cada exemplo {"dialogue": [...]} -> {"text": "..."}
    return ds.map(lambda ex: {"text": dialogue_to_text(ex["dialogue"])}, remove_columns=ds.column_names)

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

for k in ["train", "val", "test"]:
    splits[k] = splits[k].filter(lambda ex: len(ex["dialogue"]) > 0)

train_txt = as_text_dataset(splits["train"])
val_txt   = as_text_dataset(splits["val"])
test_txt  = as_text_dataset(splits["test"])

# Converte datasets em JSON
train_path = save_text_jsonl(train_txt, "train.jsonl")
val_path   = save_text_jsonl(val_txt,   "val.jsonl")
test_path  = save_text_jsonl(test_txt,  "test.jsonl")

# # Converte JSON no formato de texto padrão do GPT-2
# txt_train = json_to_text("train.jsonl")
# txt_val = json_to_text("val.jsonl")
# txt_test = json_to_text("test.jsonl")

print("Salvos em:", train_path, val_path, test_path)
