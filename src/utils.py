import yaml
import os

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