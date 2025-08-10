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