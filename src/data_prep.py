import sys
from pathlib import Path

# Adiciona o diretório raiz do projeto ao path (prioriza no início para evitar conflitos com pacotes instalados chamados 'src')
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import re
import json
from bs4 import BeautifulSoup
from datasets import load_dataset, concatenate_datasets
from pathlib import Path
from src.utils import map_dataset

def load_dataset_hf(dataset_name, sample_len=None):
    return load_dataset(dataset_name, split=f"train[:{sample_len}]" if sample_len is not None else "train")

def _norm_space(s: str) -> str:
    # remove espaços duplicados e aparas
    return re.sub(r"\s+", " ", s).strip()

def _clean(s):
    if not s: return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_sciq_answer(ca, sup):
    ca, sup = _clean(ca), _clean(sup)
    if not ca: 
        return sup
    if not sup: 
        return ca
    # evita duplicar se sup já começa com a resposta
    if sup.lower().startswith(ca.lower()):
        return sup
    return f"{ca}. {sup}"

def to_dialog(example, src):
    if src == "alpaca":
        return {"dialogue":[
            {"role":"user","content":example["instruction"] + ("" if not example["input"] else " " + example["input"])},
            {"role":"assistant","content":example["output"]}
        ]}
    elif src == "dolly":
        return {"dialogue":[
            {"role":"user","content":example["instruction"]},
            {"role":"assistant","content":example["response"]}
        ]}
    elif src == "sciq":
        answer = build_sciq_answer(example.get("correct_answer"), example.get("support"))
        return {"dialogue":[
            {"role":"user","content": _clean(example["question"])},
            {"role":"assistant","content": answer}
        ]}

def convert_dataset(dataset, name):
    # Remove todas as colunas originais e mantém apenas a coluna 'dialogue' criada pelo mapeamento
    mapped = map_dataset(dataset, to_dialog, args=[name])
    keep = ["dialogue"]
    drop = [c for c in mapped.column_names if c not in keep]
    return mapped.remove_columns(drop)


def split_dataset(dataset, val_size=0.025, test_size=0.025, seed=42):
    # Separa dataset em train e val+test
    temp_size = val_size + test_size
    split1 = dataset.train_test_split(test_size=temp_size, seed=seed)
    
    train_ds = split1["train"]
    temp_ds = split1["test"]

    # Separa temp_ds em val e test datasets
    val_ratio = val_size/temp_size
    split2 = temp_ds.train_test_split(test_size=1-val_ratio, seed=seed)

    val_ds = split2["train"]
    test_ds = split2["test"]

    return {"train": train_ds, "val": val_ds, "test": test_ds}

def concatenate_datasets_(datasets, val_size=.025, test_size=.025, seed=42, force_ascii=False):
    full = concatenate_datasets(datasets)
    full.shuffle(seed=seed).to_json("../data/raw/train_edu.jsonl", orient="records", lines=True, force_ascii=force_ascii)
    return split_dataset(full, val_size=val_size, test_size=test_size, seed=seed)

def clean_text(text):
    if not text:
        return ""
    text = BeautifulSoup(text, "lxml").get_text()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return text.strip()

def clean_dialogue(example):
    cleaned_dialogue = []

    for msg in example["dialogue"]:
        role = msg["role"].strip()
        text = clean_text(msg["content"])
        if not text:
            continue
        cleaned_dialogue.append({"role": role, "content": text})

    return {"dialogue": cleaned_dialogue}

def clean_dataset(dataset):
    return {
        "train": map_dataset(dataset["train"], clean_dialogue),
        "val": map_dataset(dataset["val"], clean_dialogue),
        "test": map_dataset(dataset["test"], clean_dialogue)
    }