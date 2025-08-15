import sys
import os
import yaml
import wandb
import torch
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

train_args   = params.get("train", None)
logging_args = params.get("logging", None)
project_name = params.get("project_name", "chatbot-edu-gpt2")
seed = params.get("seed", 42)

# --- preparar env do W&B ---
if logging_args.get("report_to", "none") == "wandb":
    os.environ["WANDB_PROJECT"] = project_name
    if logging_args.get("entity"):
        os.environ["WANDB_ENTITY"] = logging_args["entity"]
    if logging_args.get("run_name"):
        os.environ["WANDB_NAME"] = logging_args["run_name"]


def create_trainer(model, dataset, tokenizer):
    out_dir = Path("../experiments") / f"{project_name}" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        output_dir=out_dir.as_posix(),
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

        # --- logging / W&B ---
        report_to=logging_args.get("report_to", "none"),   # "wandb"
        logging_dir=Path(logging_args["output_dir"]).as_posix(),
        logging_steps=int(logging_args["logging_steps"]),
        run_name=logging_args.get("run_name", None),       # nome do run no W&B
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

    if logging_args.get("report_to", "none") == "wandb":
        wandb.config.update(params, allow_val_change=True)
    trainer = create_trainer(model, dataset, tokenizer)
    trainer.train()
    trainer.save_state()
    return trainer

def evaluate(trainer):
    return trainer.evaluate()

def save_model_tokenizer(trainer, tokenizer):
    # Salva adapters (como o Trainer já está sobre PeftModel, isso grava os pesos do adapter)
    out = "../experiments/checkpoints/adapter"
    trainer.save_model(out)
    tokenizer.save_pretrained(out)

def log_samples_wandb(trainer, tokenizer, prompts):
    if trainer.args.report_to and "wandb" in trainer.args.report_to:
        rows = []
        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt").to(trainer.model.device)
            out = trainer.model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9)
            gen = tokenizer.decode(out[0], skip_special_tokens=True)
            rows.append({"prompt": p, "generation": gen})
        wandb.log({"samples": wandb.Table(data=rows, columns=["prompt","generation"])})
