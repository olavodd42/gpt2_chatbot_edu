import sys
from pathlib import Path

# Adiciona o diretório raiz do projeto ao path (prioriza no início para evitar conflitos com pacotes instalados chamados 'src')
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import re
import json
from bs4 import BeautifulSoup
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from pathlib import Path
from src.utils import map_dataset, _first, _as_str

from typing import Optional, Union

def load_dataset_hf(dataset_name: str, sample_len: Optional[int] = None) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
    return load_dataset(dataset_name, split=f"train[:{sample_len}]" if sample_len is not None else "train")

def _norm_space(s: str) -> str:
    # remove espaços duplicados e aparas
    return re.sub(r"\s+", " ", s).strip()

_WS = re.compile(r"\s+")

def _norm(s):
    s = _as_str(s)
    if not s:
        return ""
    try:
        s = BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
    except Exception:
        s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return _WS.sub(" ", s).strip()

def _clean(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        # tenta converter em string; se não der, retorna vazio
        try:
            s = str(s)
        except Exception:
            return ""
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

def extract_answer_from_annotations(annotations: dict) -> str:
    if not annotations:
        return ""

    # às vezes vem como lista de anotações
    if isinstance(annotations, list) and len(annotations) > 0:
        annotations = annotations[0]

    sa = annotations.get("short_answers")
    if isinstance(sa, list) and len(sa) > 0:
        sa0 = sa[0]
        # 'text' pode ser str, lista de str, lista de listas...
        txt = sa0.get("text")
        txt = _as_str(txt)        # desempacota iterativamente
        if isinstance(txt, str) and txt.strip():
            return _norm(txt)

    # sem texto para long_answer (offsets), não há o que extrair aqui
    return ""

def nq_to_dialog(example):
    """
    Converte um exemplo do NQ (com 'question' + 'annotations' e/ou 'answers')
    para {"dialogue":[{"role":"user","content":...}, {"role":"assistant","content":...}]}.
    É tolerante a múltiplos esquemas.
    """
    q = _norm(example.get("question", ""))
    annotations = example.get("annotations")
    ans = extract_answer_from_annotations(example.get("annotations"))

    if not q or not ans:
        return {"dialogue": []}
    return {
        "dialogue": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": ans}
        ]
    }


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

    elif src == "natural-questions":
        return nq_to_dialog(example)
    
def add_source_column(ds, source_name: str) -> Dataset:
    return ds.map(lambda _: {"source": source_name})


def _take(ds: Dataset, n: int, seed: int = 42) -> Dataset:
    n = min(n, len(ds))
    return ds.shuffle(seed=seed).select(range(n))

def balance_by_targets(dsets: dict, targets: dict, seed: int = 42) -> Dataset:
    balanced = []
    for name, ds in dsets.items():
        ds = add_source_column(ds, name)
        tgt = targets.get(name, len(ds))
        if len(ds) > tgt:
            ds = _take(ds, tgt, seed)
        balanced.append(ds)
    return concatenate_datasets(balanced).shuffle(seed=seed)

def split_keep_ratios(ds: Dataset, val: float = 0.02, test: float = 0.02, seed: int = 42) -> DatasetDict:
    tmp_size = val + test
    spl1 = ds.train_test_split(test_size=tmp_size, seed=seed)
    tmp  = spl1["test"].train_test_split(test_size=test/(val + test), seed=seed)
    return DatasetDict({"train": spl1["train"], "val": tmp["train"], "test": tmp["test"]})

def convert_dataset(dataset, name):
    # Remove todas as colunas originais e mantém apenas a coluna 'dialogue' criada pelo mapeamento
    mapped = dataset.map(lambda ex: to_dialog(ex, name))
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
    full.shuffle(seed=seed).to_json("../data/raw/train_edu.jsonl", lines=True, force_ascii=force_ascii)
    return split_dataset(full, val_size=val_size, test_size=test_size, seed=seed)

def filter_dataset(dataset, column):
    return dataset.filter(lambda ex: len(ex[column]) > 0)

def _ensure_text(x):
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return " ".join(map(str, x))
    return str(x)


def strip_html_safe(s: str) -> str:
    # tolerant parser
    try:
        return BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
    except Exception:
        # fallback if parser fails
        return re.sub(r"<[^>]+>", " ", s or "")

def clean_text(text):
    text = _ensure_text(text)
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", " ", text)  # remove control chars
    text = strip_html_safe(text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return text.strip()


def clean_dialogue(example):
    cleaned_dialogue = []
    for msg in example.get("dialogue", []):
        role = _ensure_text(msg.get("role", "")).strip().lower()
        text = clean_text(msg.get("content", ""))
        if not text:
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        cleaned_dialogue.append({"role": role, "content": text})
    return {"dialogue": cleaned_dialogue} if cleaned_dialogue else {"dialogue": []}

def clean_dataset(dataset):
    splits = {
        "train": map_dataset(dataset["train"], clean_dialogue),
        "val": map_dataset(dataset["val"], clean_dialogue),
        "test": map_dataset(dataset["test"], clean_dialogue)
    }

    for k in ["train", "val", "test"]:
        splits[k] = splits[k].filter(
            lambda ex: isinstance(ex.get("dialogue"), list) and any(m.get("content") for m in ex["dialogue"])
        )
        
    return splits

def save_text_jsonl(data, fname):
    """
    Salva em JSON Lines compatível com load_dataset("json").
    Aceita tanto datasets.Dataset (HF) quanto pandas.DataFrame.
    """
    out_dir = Path("../data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname

    # Dataset HF
    if data.__class__.__name__ == "Dataset":
        data.to_json(str(out_path), lines=True, force_ascii=False)
        return str(out_path)

    # Fallback: pandas DataFrame
    if hasattr(data, "to_json"):
        data.to_json(str(out_path), orient="records", lines=True, force_ascii=False)
        return str(out_path)

    raise TypeError("save_text_jsonl: objeto 'data' não é Dataset HF nem DataFrame.")