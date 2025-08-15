import sys
import os
from pathlib import Path

# Adiciona o diretório raiz do projeto ao path (prioriza no início para evitar conflitos com pacotes instalados chamados 'src')
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_prep import (
    load_dataset_hf, convert_dataset, concatenate_datasets_, clean_dataset,
    save_text_jsonl, filter_dataset, balance_by_targets, split_keep_ratios)
from src.templates import as_text

from datasets import load_dataset

# DEBUG
# ds = load_dataset("google-research-datasets/natural_questions", split="train[:200]")
# mapped = ds.map(nq_to_dialog, remove_columns=ds.column_names)
# print(mapped[0]["dialogue"])  # deve ter user/assistant


# 1) Carrega datasets do Hugging Face
alpaca = load_dataset_hf("tatsu-lab/alpaca")
dolly = load_dataset_hf("databricks/databricks-dolly-15k")
sciq = load_dataset_hf("sciq")
nq = load_dataset_hf("google-research-datasets/natural_questions")

# Converte datasets no formato de chat
alpaca = convert_dataset(alpaca, name="alpaca")
dolly = convert_dataset(dolly, name="dolly")
sciq = convert_dataset(sciq, name="sciq")
nq = convert_dataset(nq, name="natural-questions")

# 2) Filtrar diálogos vazios (alguns exemplos podem virar [])
for ds_name, ds in [("alpaca", alpaca), ("dolly", dolly), ("sciq", sciq), ("nq", nq)]:
    locals()[ds_name] = ds.filter(lambda ex: len(ex["dialogue"]) == 2)


TOTAL = 100000

targets = {
    "alpaca": int(0.25 * TOTAL),
    "dolly":  int(0.25 * TOTAL),
    "sciq":   int(0.20 * TOTAL),
    "nq":     int(0.30 * TOTAL),
}

balanced = balance_by_targets(
    {"alpaca": alpaca, "dolly": dolly, "sciq": sciq, "nq": nq},
    targets,
    seed=42
)
splits = split_keep_ratios(balanced, val=0.02, test=0.02, seed=42)
splits = clean_dataset(splits)

for k in ["train", "val", "test"]:
    splits[k] = filter_dataset(splits[k], column="dialogue")

train_txt = as_text(splits["train"])
val_txt   = as_text(splits["val"])
test_txt  = as_text(splits["test"])

# Converte datasets em JSON
train_path = save_text_jsonl(train_txt, "train.jsonl")
val_path   = save_text_jsonl(val_txt,   "val.jsonl")
test_path  = save_text_jsonl(test_txt,  "test.jsonl")


print("Salvos em:", train_path, val_path, test_path)

from collections import Counter

def count_source(ds):
    # 'source' foi adicionada no balanceamento, preservada no split
    return Counter(ds["source"]) if "source" in ds.column_names else {}

print("train:", count_source(splits["train"]))
print("val:  ", count_source(splits["val"]))
print("test: ", count_source(splits["test"]))