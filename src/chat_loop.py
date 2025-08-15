import sys
from pathlib import Path

# --- PATHs do projeto ---
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --- utilidades do projeto ---
from src.templates import USER, ASSISTANT            # "<|user|>", "<|assistant|>"


# ================= Helpers =================
def build_prompt(history, user_msg):
    """
    Constrói o prompt no MESMO formato usado no treino:
    <|user|> ...
    <|assistant|> ...
    """
    lines = []
    for r, m in history:
        if r == USER:
            lines.append(f"{USER} {m}")
        else:
            lines.append(f"{ASSISTANT} {m}")
    # adiciona a nova entrada do usuário e abre a tag do assistente
    lines.append(f"{USER} {user_msg}")
    lines.append(f"{ASSISTANT} ")
    return "\n".join(lines)

def truncate_to_context(tokenizer, text, max_len=1024):
    """
    Trunca o prompt por TOKEN (mantém o final) para caber no contexto do GPT-2 (1024 por padrão).
    """
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if ids.numel() <= max_len:
        return text, ids
    # mantemos só o final
    ids = ids[-max_len:]
    return tokenizer.decode(ids, skip_special_tokens=False), ids.unsqueeze(0)

def assert_lora_attached(m):
    has_peft = hasattr(m, "peft_config") and len(getattr(m, "peft_config", {})) > 0
    if not has_peft:
        print("[AVISO] Nenhum adapter PEFT (LoRA) detectado. Verifique ADAPTER_DIR.")