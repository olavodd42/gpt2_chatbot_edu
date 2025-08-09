import sys
import os
from pathlib import Path

# Adiciona o diretório raiz do projeto ao path (prioriza no início para evitar conflitos com pacotes instalados chamados 'src')
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_prep import load_dataset_hf, convert_dataset, concatenate_datasets_

# Carrega datasets do Hugging Face
alpaca = load_dataset_hf("tatsu-lab/alpaca", sample_len=500)
dolly = load_dataset_hf("databricks/databricks-dolly-15k", sample_len=500)
sciq = load_dataset_hf("sciq", sample_len=500)  # sample

# Converte datasets no formato de chat
alpaca = convert_dataset(alpaca, name="alpaca")
dolly = convert_dataset(dolly, name="dolly")
sciq = convert_dataset(sciq, name="sciq")

# Concatena datasets
dataset = concatenate_datasets_([alpaca, dolly, sciq])

# Exibir informações sobre o dataset
print(f"Dataset total: {len(dataset)} exemplos")
print(f"Colunas disponíveis: {dataset.column_names}")
print("\n" + "="*50)
print("PRIMEIROS 3 EXEMPLOS DO DATASET:")
print("="*50)

# Exibir os primeiros 3 exemplos
for i in range(min(3, len(dataset))):
    print(f"\n--- Exemplo {i+1} ---")
    for column in dataset.column_names:
        value = dataset[i][column]
        if isinstance(value, str) and len(value) > 200:
            # Truncar textos muito longos
            print(f"{column}: {value[:200]}...")
        else:
            print(f"{column}: {value}")


