#!/usr/bin/env python3
"""Fine-tune a DeBERTa/RoBERTa model on TuringBench for human-vs-AI detection."""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import roc_auc_score, accuracy_score

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune on TuringBench")
    parser.add_argument("--model_name", type=str, default="roberta-large",
                        help="Base HuggingFace model name")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_DIR, "models", "turingbench_roberta_large"),
                        help="Directory to save fine-tuned model")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum token length")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit training samples for fast testing")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Limit validation samples for fast testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = np.argmax(logits, axis=-1)
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    return {"auc": auc, "accuracy": acc}


def load_turingbench():
    print("Loading TuringBench dataset...")
    tb = load_dataset("turingbench/TuringBench", revision="refs/convert/parquet")
    train = tb['train'].to_pandas()
    val = tb['validation'].to_pandas()

    for df in (train, val):
        df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() == 'human' else 1)
        df['text'] = df['Generation'].astype(str)

    # Drop empty/unlabeled rows
    train = train[train['text'].str.len() >= 20].reset_index(drop=True)
    val = val[val['text'].str.len() >= 20].reset_index(drop=True)

    print(f"Train rows: {len(train)}, labels: {train['label'].value_counts().to_dict()}")
    print(f"Validation rows: {len(val)}, labels: {val['label'].value_counts().to_dict()}")
    return train, val


def make_dataset(df, tokenizer, max_length):
    from datasets import Dataset
    ds = Dataset.from_pandas(df[['text', 'label']])

    def tokenize(example):
        return tokenizer(
            example['text'],
            truncation=True,
            max_length=max_length,
        )

    return ds.map(tokenize, batched=True, remove_columns=['text'])


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_df, val_df = load_turingbench()

    if args.max_train_samples is not None:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)),
                                   random_state=args.seed).reset_index(drop=True)
        print(f"Using {len(train_df)} training samples for fast testing")

    if args.max_val_samples is not None:
        val_df = val_df.sample(n=min(args.max_val_samples, len(val_df)),
                               random_state=args.seed).reset_index(drop=True)
        print(f"Using {len(val_df)} validation samples for fast testing")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "human", 1: "ai"},
        label2id={"human": 0, "ai": 1},
    )
    model.gradient_checkpointing_enable()

    train_ds = make_dataset(train_df, tokenizer, args.max_length)
    val_ds = make_dataset(val_df, tokenizer, args.max_length)
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    # Class weights for imbalance
    human_count = (train_df['label'] == 0).sum()
    ai_count = (train_df['label'] == 1).sum()
    total = human_count + ai_count
    class_weights = torch.tensor([total / (2.0 * human_count), total / (2.0 * ai_count)],
                                  dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    class_weights = class_weights.to(device)
    model = model.to(device)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.float(), labels)
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        report_to=["none"],
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on validation set...")
    eval_result = trainer.evaluate()
    print(eval_result)

    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save result CSV
    results_df = pd.DataFrame([{
        "model": args.model_name,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "val_auc": eval_result.get("eval_auc", None),
        "val_accuracy": eval_result.get("eval_accuracy", None),
    }])
    results_path = os.path.join(PROJECT_DIR, "results", "turingbench_finetuned.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")


if __name__ == '__main__':
    main()
