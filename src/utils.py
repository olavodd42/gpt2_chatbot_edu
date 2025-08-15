import yaml
import os
from datasets import Dataset

def load_params(filename):
    try:
        with open(os.path.join(f"../configs/{filename}"), 'r') as f:
            config_params = yaml.safe_load(f)

        return config_params
    except FileNotFoundError:
        print("Error: config.yaml not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")

def map_dataset(dataset, function, args=None, remove_columns=None):
    if args is None:
        return dataset.map(function, remove_columns=remove_columns)
    if isinstance(args, dict):
        return dataset.map(function, fn_kwargs=args, remove_columns=remove_columns)
    if isinstance(args, (list, tuple)):
        return dataset.map(lambda ex: function(ex, *args), remove_columns=remove_columns)
    # se args for um único valor escalar:
    return dataset.map(lambda ex: function(ex, args), remove_columns=remove_columns)

def _as_float(d, k):
    v = d[k]
    try:
        return float(v)
    except Exception:
        raise TypeError(f"Campo '{k}' deve ser float, veio: {v!r} ({type(v)})")

def _as_int(d, k):
    v = d[k]
    try:
        return int(v)
    except Exception:
        raise TypeError(f"Campo '{k}' deve ser int, veio: {v!r} ({type(v)})")

def _as_bool(d, k):
    v = d[k]
    if isinstance(v, bool): 
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1","true","yes","y","on")
    raise TypeError(f"Campo '{k}' deve ser bool, veio: {v!r} ({type(v)})")


def _first(x):
    return x[0] if isinstance(x, (list, tuple)) and x else x

def _as_str(x):
    visited = 0
    while True:
        if x is None:
            return ""
        if isinstance(x, (list, tuple)) and x:
            x = x[0]
        elif isinstance(x, dict):
            if "text" in x:
                x = x["text"]
            elif "texts" in x:
                x = x["texts"]
            else:
                return str(x)
        else:
            break
        visited += 1
        if visited > 50:
            return str(x)
    return str(x) if not isinstance(x, str) else x
