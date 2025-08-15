import sys
import torch
import re
import yaml
import bitsandbytes as bnb
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    StoppingCriteria, StoppingCriteriaList
    )
from peft import PeftModel

# --- PATHS ---
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from src.modeling import create_data_collator, load_model
from src.utils import _as_bool,_as_float,_as_int

with open("../configs/train_small.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)
project_name = params.get("project_name", "chatbot-edu-gpt2")
seed = params.get("seed", 42)

ADAPTER_DIR = (ROOT_DIR / "experiments" / f"{project_name}" / f"seed{seed}" / "checkpoints" / "adapter").as_posix()
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


# --- 1) StoppingCriteria para parar em sequência de tokens (e.g., "<|user|>")
class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings, prompt_len, max_chars_to_check=300):
        super().__init__()
        self.tok = tokenizer
        self.stop_strings = stop_strings
        self.max_chars_to_check = max_chars_to_check
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0]
        tail_ids = seq[self.prompt_len:]
        if tail_ids.numel() == 0:
            return False
        text = self.tok.decode(tail_ids, skip_special_tokens=False)
        text = text[-self.max_chars_to_check:]
        return any(s in text for s in self.stop_strings)
    
# -------- template de prompt --------
def format_prompt(user_text: str) -> str:
    return f"<|user|> {user_text}\n<|assistant|> "

def _postprocess_response(tokenizer, prompt_ids, full_ids):
    # texto total (prompt + geração)
    full_text  = tokenizer.decode(full_ids, skip_special_tokens=False)
    prompt_txt = tokenizer.decode(prompt_ids, skip_special_tokens=False)

    # só a parte gerada
    gen = full_text[len(prompt_txt):]

    # 1) corte no primeiro marcador de fim (na ordem que preferir)
    for stop in ("<|endoftext|>", "\n<|user|>"):
        if stop in gen:
            gen = gen.split(stop, 1)[0]

    # 2) remova tags e espaços
    gen = gen.replace("<|assistant|>", "").strip()

    # 3) fallback extra: se ainda vier “lixo” multi-parágrafo, retenha só primeiro parágrafo/sentença
    gen = gen.split("\n\n", 1)[0].strip()
    # opcional: corte após primeira sentença longa
    m = re.search(r"(.+?[.!?])(\s|$)", gen)
    if m and len(gen) > 300:  # só se ficou verborrágico
        gen = m.group(1)

    return gen

# -------- chat unitário --------
@torch.no_grad()
def chat(
    model, tokenizer, user_text: str,
    max_new_tokens=48, temperature=0.5, top_p=0.9,
    repetition_penalty=1.12, do_sample=True, **gen_kwargs
):
    # limpar duplicatas
    for k in ["max_new_tokens","temperature","top_p","repetition_penalty","do_sample",
              "eos_token_id","pad_token_id","stopping_criteria","num_return_sequences",
              "return_dict_in_generate","output_scores","no_repeat_ngram_size","bad_words_ids"]:
        gen_kwargs.pop(k, None)

    prompt = format_prompt(user_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    stop_strings = ["<|endoftext|>", "\n<|user|>"]
    stopping = StoppingCriteriaList([StopOnSubstrings(tokenizer, stop_strings, prompt_len)])

    # opcional: impedir que o modelo gere o marcador de usuário
    bad_words_ids = tokenizer(["<|user|>"], add_special_tokens=False).input_ids

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=3,
        bad_words_ids=bad_words_ids,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        early_stopping=True,
        stopping_criteria=stopping,
    )

    return _postprocess_response(tokenizer, inputs["input_ids"][0], out[0])


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

        prompt_len = inp_ids.shape[1]
        stop_strings = ["<|endoftext|>", "\n<|user|>"]
        stopping = StoppingCriteriaList([StopOnSubstrings(tokenizer, stop_strings, prompt_len)])

        out = model.generate(
            input_ids=inp_ids,
            attention_mask=attn,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            stopping_criteria=stopping,
            **gen_kwargs
        )

        prompt_text = tokenizer.decode(inp_ids[0], skip_special_tokens=False)
        gen_only = _postprocess_response(tokenizer, inp_ids[0], out[0])

        print("\nPROMPT:\n", prompt_text)
        print("\nOUTPUT:\n", gen_only)

def assert_vocab_match(model, tokenizer, name=""):
    n_model = model.get_input_embeddings().num_embeddings
    n_tok = len(tokenizer)
    if n_model != n_tok:
        raise ValueError(f"[{name}] vocab mismatch: model={n_model} vs tokenizer={n_tok}")
