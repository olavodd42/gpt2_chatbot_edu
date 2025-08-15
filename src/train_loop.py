import sys
import os
import yaml
import bitsandbytes as bnb
from pathlib import Path
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from src.modeling import create_data_collator, load_model
from src.utils import _as_bool,_as_float,_as_int

with open("../configs/train_small.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

train_args   = params["train"]
logging_args = params["logging"]

def create_trainer(model, dataset, tokenizer):
    # Tipagem defensiva (yaml -> tipos nativos corretos)
    per_device_bs   = _as_int(train_args, "train_batch_size")
    grad_accum      = _as_int(train_args, "grad_accum_steps")
    eval_steps      = _as_int(train_args, "eval_steps")
    save_steps      = _as_int(train_args, "save_steps")
    num_workers     = _as_int(train_args, "dataloader_num_workers")
    warmup_ratio    = _as_float(train_args, "warmup_ratio")
    learning_rate   = _as_float(train_args, "lr")
    weight_decay    = _as_float(train_args, "wd")
    use_fp16        = _as_bool(train_args, "fp16")
    group_by_length = _as_bool(train_args, "group_by_length")
    logging_steps   = _as_int(logging_args, "logging_steps")
    report_to       = logging_args["report_to"]  # string ("none") ou lista, dependendo do seu YAML
    log_dir         = logging_args["output_dir"]

    print("DEBUG tipos (train):", {
        "train_batch_size": type(per_device_bs).__name__,
        "grad_accum_steps": type(grad_accum).__name__,
        "eval_steps": type(eval_steps).__name__,
        "save_steps": type(save_steps).__name__,
        "dataloader_num_workers": type(num_workers).__name__,
        "warmup_ratio": type(warmup_ratio).__name__,
        "lr": type(learning_rate).__name__,
        "wd": type(weight_decay).__name__,
        "fp16": type(use_fp16).__name__,
        "group_by_length": type(group_by_length).__name__,
    })
    print("DEBUG tipos (logging):", {
        "report_to": type(report_to).__name__,
        "output_dir": type(log_dir).__name__,
        "logging_steps": type(logging_steps).__name__,
    })

    data_collator = create_data_collator(tokenizer)

    args = TrainingArguments(
        output_dir="../experiments",
        seed=42,

        # treino/otimizador
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,

        # avaliação & salvamento
        eval_strategy="steps",   # <- nome correto
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,          # <- para usar early stopping direito
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # desempenho
        fp16=use_fp16,
        dataloader_num_workers=num_workers,
        group_by_length=group_by_length,
        remove_unused_columns=False,

        # logging
        report_to=report_to,
        logging_dir=log_dir,
        logging_steps=logging_steps,
    )

    return Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

def train(model, dataset, tokenizer):
    trainer = create_trainer(model, dataset, tokenizer)
    trainer.train()
    return trainer

def evaluate(trainer):
    return trainer.evaluate()

def save_model_tokenizer(trainer, tokenizer):
    # Salva adapters (como o Trainer já está sobre PeftModel, isso grava os pesos do adapter)
    out = "../experiments/checkpoints/adapter"
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
