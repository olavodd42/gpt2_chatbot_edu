
import sys
from pathlib import Path

# Adiciona o diretório raiz do projeto ao path (prioriza no início para evitar conflitos com pacotes instalados chamados 'src')
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
from src.utils import load_params, map_dataset
from peft import LoraConfig, get_peft_model, TaskType

try:
    params = load_params('train_small.yaml')
except ValueError:
    print("Failed to load parameters from 'train_small.yaml'. Please check the file and its contents.")
    
model_params = params["model"]
lora_params = params["lora"]
SPECIALS = model_params["add_special_tokens"]
BLOCK_SIZE = 512

def load_tokenizer(model_name="gpt-2"):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIALS})
    return tokenizer

def config_lora_model(model):
    lora_cfg = LoraConfig(
        r=lora_params['r'],
        lora_alpha=lora_params["alpha"],
        lora_dropout=lora_params["dropout"],
        task_type=lora_params["task_type"],
        bias=lora_params["bias"],
        target_modules=lora_params["target_modules"]
    )

    return get_peft_model(model, lora_cfg)

def load_model():
    bnb_cfg = BitsAndBytesConfig(
        load_in_8bit=model_params["load_in_8bit"],     # força 8-bit
        load_in_4bit=False     # garante que 4-bit NÃO será usado
    )
    tokenizer = load_tokenizer(model_params["base_ckpt"])
    print("model_params:", model_params)
    print("bnb:", {"load_in_8bit": True, "load_in_4bit": False, "device_map": "auto"})
    model = AutoModelForCausalLM.from_pretrained(
        model_params["base_ckpt"],
        quantization_config=bnb_cfg,           # para 8-bit
        device_map=model_params["device_map"]
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = model_params["cache"]

    if model_params["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    if lora_params["enabled"]:
        model = config_lora_model(model)

    return tokenizer, model

def create_data_collator(tokenizer):
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    return collator


def tokenize_dataset(tokenizer, dataset, block_size=512):
    # 1) tira espaços e linhas vazias
    def strip_fn(ex):
        txt = ex["text"] if ex["text"] is not None else ""
        return {"text": txt.strip()}
    dataset = dataset.map(strip_fn)

    dataset = dataset.filter(lambda ex: len(ex["text"]) > 0)

    # 2) tokeniza em lote
    def tok_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=block_size,
            return_attention_mask=True,
        )
    tokenized = dataset.map(tok_batch, batched=True, remove_columns=["text"])

    # 3) remove quaisquer resíduos vazios (defensivo)
    tokenized = tokenized.filter(lambda ex: len(ex["input_ids"]) > 0)

    return tokenized