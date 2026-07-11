"""
Modern QLoRA sequence-classifier trainer for 70B+ models (Qwen, LLaMA, etc.).
Designed for a single A100 80GB or multi-A100 node.

Usage:
    HF_TOKEN=hf_... python scripts/train_qlora_classifier.py \
        --model_name unsloth/Qwen2.5-72B-Instruct-bnb-4bit \
        --hub_model_id vedangvatsa123/vedang-turingbench-qwen-72b-qlora \
        --output_dir /workspace/models/turingbench_qwen_72b

The script supports resuming from a local checkpoint or from the Hub checkpoint
if the Hub repo already exists.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen3-4B",
                        help="Base model name or HF path (e.g. Qwen/Qwen3-4B)")
    parser.add_argument("--output_dir", type=str, default="./models/turingbench_qlora",
                        help="Directory to save checkpoints and final model")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="HF Hub model ID to push to")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum token length")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="LoRA learning rate")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit training samples for fast testing")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Limit validation samples for fast testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def has_hf_token():
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))


def load_turingbench():
    print("Loading TuringBench dataset...")
    dataset = load_dataset("turingbench/TuringBench", revision="refs/convert/parquet")
    train = dataset["train"].to_pandas()
    val = dataset["validation"].to_pandas()

    for df in (train, val):
        df["label"] = df["label"].apply(lambda x: 0 if str(x).lower() == "human" else 1)
        df["text"] = df["Generation"].astype(str)

    train = train[train["text"].str.len() >= 20].reset_index(drop=True)
    val = val[val["text"].str.len() >= 20].reset_index(drop=True)
    return train[["text", "label"]], val[["text", "label"]]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = np.argmax(logits, axis=-1)
    return {
        "auc": roc_auc_score(labels, probs),
        "accuracy": accuracy_score(labels, preds),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if not has_hf_token():
        print("WARNING: HF_TOKEN not set. Hub push will be disabled.")
    else:
        print("HF_TOKEN detected.")

    train_df, val_df = load_turingbench()
    if args.max_train_samples:
        train_df = train_df.sample(min(args.max_train_samples, len(train_df)), random_state=args.seed).reset_index(drop=True)
    if args.max_val_samples:
        val_df = val_df.sample(min(args.max_val_samples, len(val_df)), random_state=args.seed).reset_index(drop=True)

    print(f"Train rows: {len(train_df)}, labels: {train_df['label'].value_counts().to_dict()}")
    print(f"Validation rows: {len(val_df)}, labels: {val_df['label'].value_counts().to_dict()}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "human", 1: "ai"},
        label2id={"human": 0, "ai": 1},
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(model)

    # Detect target modules by model family
    model_name_lower = args.model_name.lower()
    if "qwen" in model_name_lower:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "llama" in model_name_lower or "mistral" in model_name_lower:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length, padding=False)

    from datasets import Dataset
    train_ds = Dataset.from_pandas(train_df).map(tokenize_function, batched=True, remove_columns=["text"])
    val_ds = Dataset.from_pandas(val_df).map(tokenize_function, batched=True, remove_columns=["text"])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    can_push = args.hub_model_id is not None and has_hf_token()
    if args.hub_model_id and not can_push:
        print(f"WARNING: --hub_model_id={args.hub_model_id} set but no HF_TOKEN. Hub push disabled.")

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    warmup_steps = max(1, len(train_ds) // effective_batch_size // 10)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        report_to=["none"],
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        bf16=True,
        push_to_hub=can_push,
        hub_model_id=args.hub_model_id if can_push else None,
        hub_strategy="checkpoint" if can_push else "every_save",
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Resume from the latest local checkpoint if it exists
    resume_path = None
    checkpoint_dir = Path(args.output_dir)
    if checkpoint_dir.exists():
        checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        if checkpoints:
            resume_path = sorted(checkpoints, key=lambda x: int(x.name.split("-")[-1]))[-1]
            print(f"Resuming from local checkpoint: {resume_path}")

    trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)

    print(f"Saving final model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    eval_result = trainer.evaluate()
    print(eval_result)

    results_df = pd.DataFrame([{
        "model": args.model_name,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "val_auc": eval_result.get("eval_auc", None),
        "val_accuracy": eval_result.get("eval_accuracy", None),
    }])
    results_path = Path(__file__).parent.parent / "results" / "turingbench_qlora.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        existing = pd.read_csv(results_path)
        results_df = pd.concat([existing, results_df], ignore_index=True)
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
