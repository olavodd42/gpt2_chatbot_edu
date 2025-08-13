import sys
from pathlib import Path
from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chat_loop import load_model_for_inference, load_base_only, chat, load_tokenized_test, generate_batch

def assert_vocab_match(model, tokenizer, name=""):
    n_model = model.get_input_embeddings().num_embeddings
    n_tok = len(tokenizer)
    if n_model != n_tok:
        raise ValueError(f"[{name}] vocab mismatch: model={n_model} vs tokenizer={n_tok}")


# dataset = load_dataset(
#     "json",
#     data_files={
#         "train": "../data/processed/train.jsonl",
#         "val":   "../data/processed/val.jsonl",
#         "test":   "../data/processed/test.jsonl"
#     }
# )

tokenizer, model = load_model_for_inference()
base_tokenizer, base_model = load_base_only()
assert_vocab_match(base_model, base_tokenizer, "BASE")
q = "Make a step by step guide on how to learn transformers models."

print("\n=== BASE ===")
print(chat(base_model, base_tokenizer, q, max_new_tokens=128))

print("\n=== FINE-TUNED (LoRA) ===")
print(chat(model, tokenizer, q, max_new_tokens=128))

ds = load_tokenized_test(tokenizer)

generate_batch(model, tokenizer, ds)