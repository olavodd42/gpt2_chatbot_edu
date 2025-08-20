import sys
import torch
from pathlib import Path
from transformers import StoppingCriteriaList

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference_loop import load_model_for_inference, StopOnSubstrings, _postprocess_response
from src.chat_loop import build_prompt, truncate_to_context, assert_lora_attached
from src.templates import USER, ASSISTANT

print("[encodings]", sys.stdout.encoding, sys.stderr.encoding, flush=True)

# ======== Modo de geração e prompt de sistema ========
MODE = "creative"   # "creative" para sampling; "factual" se quiser beam
SYSTEM = "You are an educational assistant. Be concise, factual and avoid rambling."

# ================= Loop interativo =================
tokenizer, model = load_model_for_inference()

assert_lora_attached(model)

# garanta pad/eos alinhados
eos_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id
bad_words_ids = tokenizer(["<|user|>"], add_special_tokens=False).input_ids

max_ctx = getattr(model.config, "n_positions", 1024) or 1024
history = []

print("Digite 'exit' para sair.\n")
while True:
    try:
        u = input("Você: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if u.lower() in {"exit", "quit", "sair"}:
        break
    if not u:
        continue

    # monta prompt e trunca para o contexto
    raw_prompt = build_prompt(history, u)             # <- sem SYSTEM
    truncated_prompt, prompt_ids = truncate_to_context(tokenizer, raw_prompt, max_len=max_ctx)

    # prepara tensores
    inputs = tokenizer(truncated_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    stopping = StoppingCriteriaList([
        StopOnSubstrings(tokenizer, ["<|endoftext|>", "\n<|user|>"], prompt_len=inputs["input_ids"].shape[1])
    ])


    gen_kwargs = {}
    # ======== Perfil de geração (factual vs criativo) ========
    if MODE == "creative":
        # Sampling → NÃO usar num_beams aqui
        gen_kwargs = dict(
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.18,
            no_repeat_ngram_size=4,
            # num_beams REMOVIDO
        )
    else:
        # Beam search → NÃO usar temperature/top_p/top_k
        gen_kwargs = dict(
            do_sample=False,
            num_beams=4,
            length_penalty=1.0,
        )

    # Remova quaisquer chaves com None (proteção geral)
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    bad_words_ids = tokenizer(["<|user|>"], add_special_tokens=False).input_ids

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            stopping_criteria=stopping,
            bad_words_ids=bad_words_ids,
            **gen_kwargs
        )[0]


    resp = _postprocess_response(tokenizer, inputs["input_ids"][0], out_ids)
    print("Bot:", resp)
    history.append((USER, u))
    history.append((ASSISTANT, resp))
