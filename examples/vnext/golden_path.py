#!/usr/bin/env python
"""
Golden Path: The simplest possible Gradience + HuggingFace integration.

This is the exact code from the README Quick Start.
Run this to see Gradience in action with zero configuration.

Usage:
    python golden_path.py
    
Output:
    ./outputs/run.jsonl - Telemetry file
    ./outputs/adapter/  - LoRA adapter (if using PEFT)
    
After running:
    gradience monitor outputs/run.jsonl
    gradience audit --peft-dir outputs/adapter
"""

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from gradience.vnext.integrations.hf import GradienceCallback
import torch

# Create a tiny synthetic dataset
def create_tiny_dataset():
    texts = ["Hello world", "Gradience rocks", "LoRA is cool"] * 10
    labels = [0, 1, 0] * 10
    return Dataset.from_dict({"text": texts, "label": labels})

# Tokenize function
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=32)

def main():
    print("🚀 Golden Path: Gradience + HuggingFace in one line")
    print("-" * 50)
    
    # Load a tiny model
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create dataset
    dataset = create_tiny_dataset()
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./outputs",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    
    # THE GOLDEN PATH: One line integration
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        callbacks=[GradienceCallback()]  # ← This is all you need!
    )
    
    # Train
    print("\nTraining with Gradience telemetry...")
    trainer.train()
    
    print("\n✅ Training complete!")
    print(f"\n📊 Telemetry written to: {training_args.output_dir}/run.jsonl")
    print("\nNext steps:")
    print("  gradience monitor outputs/run.jsonl")
    print("\nThat's it! You're now using Gradience. 🎉")

if __name__ == "__main__":
    main()