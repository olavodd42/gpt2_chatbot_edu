
import sys
import yaml
from pathlib import Path

# Adiciona o diretório raiz do projeto ao path (prioriza no início para evitar conflitos com pacotes instalados chamados 'src')
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
from src.utils import load_params, map_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


# --- carregamento do YAML ---
with open('../configs/train_small.yaml', 'r') as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

model_params = params["model"]
lora_params  = params["lora"]
SPECIALS     = model_params["add_special_tokens"]
BLOCK_SIZE   = 512

# -------- tokenizer --------
def load_tokenizer(model_name: str = "gpt2"):
    tok = AutoTokenizer.from_pretrained(model_name)  # <- usa o argumento
    # GPT-2 não tem pad; emparelha com eos
    tok.pad_token = tok.eos_token
    if SPECIALS:
        tok.add_special_tokens({"additional_special_tokens": SPECIALS})
    return tok


# -------- LoRA --------
def config_lora_model(model):
    lora_cfg = LoraConfig(
        r=lora_params['r'],
        lora_alpha=lora_params["alpha"],
        lora_dropout=lora_params["dropout"],
        task_type=TaskType[lora_params["task_type"]],
        bias=lora_params["bias"],
        target_modules=lora_params["target_modules"],  # ["c_attn","c_proj"] p/ GPT-2
    )
    return get_peft_model(model, lora_cfg)


# -------- modelo --------
def load_model():
    # 8-bit somente (evita confusão com 4-bit)
    bnb_cfg = BitsAndBytesConfig(
        load_in_8bit = bool(model_params["load_in_8bit"]),
        load_in_4bit = False
    )

    tokenizer = load_tokenizer(model_params["base_ckpt"])
    print("model_params:", model_params)
    print("bnb(load_in_8bit, load_in_4bit):", bnb_cfg.load_in_8bit, bnb_cfg.load_in_4bit)

    model = AutoModelForCausalLM.from_pretrained(
        model_params["base_ckpt"],
        quantization_config=bnb_cfg,
        device_map=model_params["device_map"],  # "auto"
        low_cpu_mem_usage=True,
        trust_remote_code=False
    )

    # redimensiona embeddings APÓS adicionar tokens especiais
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # espelha pad/eos no config (útil na geração e em alguns collators)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache    = bool(model_params["cache"])  # tipa corretamente

    # prepare k-bit + gradient checkpointing (não precisa ligar manualmente antes)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(model_params["gradient_checkpointing"])
    )

    # aplica LoRA se habilitado
    if lora_params["enabled"]:
        model = config_lora_model(model)
        # sanidade: imprime somente para debug
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    return tokenizer, model

# -------- collator --------
def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )


# -------- tokenização do dataset de texto --------
def tokenize_dataset(tokenizer, dataset, block_size: int = 512):
    # 1) aparas/linhas vazias
    def strip_fn(ex):
        txt = ex.get("text") or ""
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

    # 3) filtro defensivo
    tokenized = tokenized.filter(lambda ex: len(ex["input_ids"]) > 0)
    return tokenized