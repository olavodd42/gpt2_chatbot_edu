
import sys
from pathlib import Path

# Adiciona o diretório raiz do projeto ao path (prioriza no início para evitar conflitos com pacotes instalados chamados 'src')
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import load_params
from peft import LoraConfig, get_peft_model, TaskType

try:
    params = load_params('train_small.yaml')
except ValueError:
    print("Failed to load parameters from 'train_small.yaml'. Please check the file and its contents.")
    
model_params = params["model"]
lora_params = params["lora"]
SPECIALS = model_params["add_special_tokens"]

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
    tokenizer = load_tokenizer(model_params["base_ckpt"])
    model = AutoModelForCausalLM.from_pretrained(
        model_params["base_ckpt"],
        load_in_8bit=model_params['load_in_8bit'],           # para 8-bit
        device_map=model_params["device_map"]
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = model_params["cache"]

    if model_params["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    if lora_params["enabled"]:
        model = config_lora_model(model)

    return tokenizer, model
