import sys
import torch
import bitsandbytes as bnb
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- PATHS ---
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from src.modeling import create_data_collator, load_model
from src.utils import _as_bool,_as_float,_as_int

ADAPTER_DIR = (ROOT_DIR / "experiments" / "checkpoints" / "adapter").as_posix()
BASE_CKPT   = "gpt2"
SPECIALS    = ["<|user|>", "<|assistant|>"]

# -------- utils de carregamento --------
def load_tokenizer_safe():
    """
    Tenta carregar o tokenizer salvo junto aos adapters; se não existir,
    cai para o tokenizer do base e injeta os tokens especiais.
    """
    try:
        tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    except Exception:
        tok = AutoTokenizer.from_pretrained(BASE_CKPT)
        tok.pad_token = tok.eos_token
        tok.add_special_tokens({"additional_special_tokens": SPECIALS})
    # Garantia: pad sempre definido
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_model_for_inference():
    """
    Carrega o base em 8-bit, alinha vocabulário com o tokenizer e acopla o LoRA.
    """
    tokenizer = load_tokenizer_safe()

    # 8-bit
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_CKPT,
        quantization_config=bnb_cfg,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False
    )

    # espelha pad/eos no config
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.eos_token_id = tokenizer.eos_token_id
    base.config.use_cache    = True  # gerar com cache ajuda a performance

    # redimensiona embeddings p/ bater com tokenizer (após adicionar especiais)
    vocab_len = len(tokenizer)
    if base.get_input_embeddings().num_embeddings != vocab_len:
        base.resize_token_embeddings(vocab_len, mean_resizing=False)

    # acopla LoRA
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    return tokenizer, model

# -------- template de prompt --------
def format_prompt(user_text: str) -> str:
    # mesmo template usado no treino
    return f"<|user|> {user_text}\n<|assistant|> "

# -------- chat unitário --------
@torch.no_grad()
def chat(
    model,
    tokenizer,
    user_text: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    **gen_kwargs
):
    # Evita duplicar args
    for k in ["max_new_tokens", "temperature", "top_p", "repetition_penalty", "do_sample"]:
        gen_kwargs.pop(k, None)

    prompt = format_prompt(user_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        **gen_kwargs
    )

    # só a parte gerada (pós-prompt)
    gen_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # limpeza defensiva: se o modelo "ecoar" um novo <|user|>, corta
    cut = text.split("<|user|>", 1)[0].strip()
    return cut

# -------- base sem LoRA (para comparação) --------
def load_base_only():
    tok = AutoTokenizer.from_pretrained(BASE_CKPT)
    tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": SPECIALS})

    base = AutoModelForCausalLM.from_pretrained(
        BASE_CKPT,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # alinhar vocabulário
    vocab_len = len(tok)
    if base.get_input_embeddings().num_embeddings != vocab_len:
        base.resize_token_embeddings(vocab_len, mean_resizing=False)

    base.config.pad_token_id = tok.pad_token_id
    base.config.eos_token_id = tok.eos_token_id
    base.config.use_cache    = True

    base.eval()
    return tok, base

# -------- util para dataset tokenizado de teste --------
def load_tokenized_test(tokenizer, path="../data/processed/test.jsonl", max_len=512):
    ds = load_dataset("json", data_files={"test": path})["test"]
    ds = ds.filter(lambda ex: isinstance(ex.get("text"), str) and len(ex["text"].strip()) > 0)

    def tok_fn(batch):
        return tokenizer(
            batch["text"],
            max_length=max_len,
            truncation=True,
            return_attention_mask=True
        )
    return ds.map(tok_fn, batched=True)

# -------- geração em lote (visualização) --------
@torch.no_grad()
def generate_batch(model, tokenizer, ds, n=5, **gen_kwargs):
    n = min(n, len(ds))
    for i in range(n):
        inp_ids = torch.tensor(ds[i]["input_ids"], device=model.device).unsqueeze(0)
        attn    = torch.tensor(ds[i]["attention_mask"], device=model.device).unsqueeze(0)
        out = model.generate(
            input_ids=inp_ids,
            attention_mask=attn,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            **gen_kwargs
        )
        # exibe prompt + resposta limpa
        prompt_text = tokenizer.decode(inp_ids[0], skip_special_tokens=False)
        full_text   = tokenizer.decode(out[0],     skip_special_tokens=True)
        # tenta extrair só a parte após "<|assistant|>"
        if "<|assistant|>" in prompt_text:
            after = full_text.split("<|assistant|>", 1)[-1]
        else:
            after = full_text
        print("\nPROMPT:")
        print(prompt_text)
        print("\nOUTPUT:")
        print(after.strip())