import json
import os

USER, ASSISTANT = "<|user|>", "<|assistant|>"

def json_to_text(filename):
    with open(os.path.join("/home/olavo-dalberto/gpt2_chatbot_edu/data/interim", filename), 'r') as f:
        data = []
        for line in f:
            data.append(json.loads(line.strip()))

    parts = [""]
    for m in data:
        dialogue = m["dialogue"]
        user_content = dialogue[0]["content"]
        assistant_content = dialogue[1]["content"]
        parts.append(f"{USER}: {user_content.strip()}\n{ASSISTANT}: {assistant_content}")

    return "\n".join(parts) + f"\n{ASSISTANT}: "

def dialogue_to_text(dialogue):
    """
    Converte uma lista de mensagens [{"role": "...", "content": "..."}]
    para uma string única no formato GPT-2.
    """
    parts = []
    for msg in dialogue:
        if msg["role"].lower() == "user":
            parts.append(f"<|user|> {msg['content']}")
        elif msg["role"].lower() == "assistant":
            parts.append(f"<|assistant|> {msg['content']}")
        else:
            parts.append(f"{msg['role']}: {msg['content']}")
    # Fecha com token de fim de texto
    return "\n".join(parts) + "<|endoftext|>"