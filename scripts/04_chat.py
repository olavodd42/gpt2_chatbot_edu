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

# ======== Modo de geração e prompt de sistema ========
MODE = "factual"   # "factual" para perguntas objetivas; "creative" para tarefas abertas
SYSTEM = "Você é um assistente educacional, objetivo e factual. Responda de forma correta e concisa."
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
    raw_prompt = SYSTEM + "\n" + build_prompt(history, u)
    truncated_prompt, prompt_ids = truncate_to_context(tokenizer, raw_prompt, max_len=max_ctx)

    # prepara tensores
    inputs = tokenizer(
        truncated_prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).to(model.device)

    # stopping criterias (robusto contra vazamento)
    stopping = StoppingCriteriaList([
        StopOnSubstrings(tokenizer, stop_strings=["<|endoftext|>", "\n<|user|>"],
                            prompt_len=inputs["input_ids"].shape[1])
    ])

    # ======== Perfil de geração (factual vs criativo) ========
    gen_kwargs = {}
    if MODE == "factual":
        # Sem sampling → respostas mais estáveis/objetivas (beam search)
        gen_kwargs.update(dict(
            do_sample=False,
            num_beams=4,
            early_stopping=True,   # aqui faz sentido com beam
        ))
    else:
        # Criativo → sampling
        gen_kwargs.update(dict(
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            # NÃO usar early_stopping aqui (gera aviso e é ignorado)
        ))


    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=64,                # menor → menos “rabo”
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            bad_words_ids=bad_words_ids,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            stopping_criteria=stopping,       # mantém o StopOnSubstrings
            **gen_kwargs                      # ← aplica o perfil de geração
        )[0]

    resp = _postprocess_response(tokenizer, inputs["input_ids"][0], out_ids)
    print("Bot:", resp)
    history.append((USER, u))
    history.append((ASSISTANT, resp))
