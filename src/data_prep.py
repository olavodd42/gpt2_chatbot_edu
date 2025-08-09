from datasets import load_dataset, concatenate_datasets
from pathlib import Path

def load_dataset_hf(dataset_name, sample_len=None):
    return load_dataset(dataset_name, split=f"train[:{sample_len}]" if sample_len is not None else "train")


def to_dialog(example, src):
    if src=="alpaca":
        return {"dialogue":[
            {"role":"user","content":example["instruction"] + ("" if not example["input"] else " " + example["input"])},
            {"role":"assistant","content":example["output"]}
        ]}
    elif src=="dolly":
        return {"dialogue":[
            {"role":"user","content":example["instruction"]},
            {"role":"assistant","content":example["response"]}
        ]}
    elif src=="sciq":
        return {"dialogue":[
            {"role":"user","content":example["question"]},
            {"role":"assistant","content":example["correct_answer"] + example["support"]}
        ]}

def convert_dataset(dataset, name):
    # Remove todas as colunas originais e mantém apenas a coluna 'dialogue' criada pelo mapeamento
    return dataset.map(lambda ex: to_dialog(ex, name), remove_columns=dataset.column_names)

def concatenate_datasets_(datasets, seed=42, force_ascii=False):
    full = concatenate_datasets(datasets)
    full.shuffle(seed=seed).to_json("../data/train_edu.jsonl", orient="records", lines=True, force_ascii=force_ascii)
    return full

def split_dataset(dataset, val_size=0.1, test_size=0.1, seed=42):
    # Separa dataset em train e val+test
    temp_size = val_size + test_size
    split1 = dataset.train_test_split(test_size=temp_size, seed=seed)

    train_ds = split1["train"]
    temp_ds = split1["test"]

    # Separa temp_ds em val e test datasets
    val_ratio = val_size/temp_size
    split2 = temp_ds.train_test_split(test_size=1-val_ratio, seed=seed)

    val_ds = temp_ds["train"]
    test_ds = temp_ds["test"]

    return {"train": train_ds, "val": val_ds, "test": test_ds}