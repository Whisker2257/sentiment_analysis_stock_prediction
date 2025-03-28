#!/usr/bin/env python3

"""
sentiment_finetuning.py

Script to fine-tune the FinancialBERT model on a labeled finance-sentiment dataset.
"""

import os
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

def custom_preprocess(text: str) -> str:
    """
    Preprocess the tweet text:
      - Replace URLs with [URL]
      - Remove # from hashtags
      - Reduce repeated punctuation
    """
    text = re.sub(r'http\S+|www.\S+', '[URL]', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'([!?.])\1+', r'\1', text)
    return text

def tokenize_function(examples, tokenizer, max_length=128):
    return tokenizer(
        examples['tweet_text'], 
        padding='max_length', 
        truncation=True, 
        max_length=max_length
    )

def compute_metrics(eval_pred):
    """
    Simple accuracy metric.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------
    # 1. Load your labeled dataset
    # ------------------------------
    data_path = os.path.join("data", "sentiment_training_data", "fin_tweets.csv")
    df = pd.read_csv(data_path)

    # df has columns: [tweet_text, label] where label in {bearish, neutral, bullish}
    df['tweet_text'] = df['tweet_text'].apply(custom_preprocess)

    # Mapping to numeric labels
    label2id = {"bearish": 0, "neutral": 1, "bullish": 2}
    df['label_id'] = df['label'].map(label2id)

    # ------------------------------
    # 2. Split & Convert to Dataset
    # ------------------------------
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # ------------------------------
    # 3. Load Tokenizer & Model
    # ------------------------------
    model_name = "ahmedrachid/FinancialBERT-Sentiment-Analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    # ------------------------------
    # 4. Tokenize
    # ------------------------------
    def tokenize_batch(examples):
        return tokenize_function(examples, tokenizer)

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    test_dataset = test_dataset.map(tokenize_batch, batched=True)

    # IMPORTANT: rename 'label_id' -> 'labels'
    train_dataset = train_dataset.rename_column("label_id", "labels")
    test_dataset = test_dataset.rename_column("label_id", "labels")

    # Now set the format so PyTorch Tensors are created with 'input_ids','attention_mask','labels'
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # ------------------------------
    # 5. Set Training Arguments
    # ------------------------------
    training_args = TrainingArguments(
        output_dir=os.path.join("models", "finetuned_finbert"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100
    )

    # ------------------------------
    # 6. Trainer & Fine-tune
    # ------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # ------------------------------
    # 7. Evaluate & Save
    # ------------------------------
    eval_results = trainer.evaluate()
    print("Final Evaluation Results:", eval_results)

    trainer.save_model(os.path.join("models", "finetuned_finbert"))
    tokenizer.save_pretrained(os.path.join("models", "finetuned_finbert"))

if __name__ == "__main__":
    main()
