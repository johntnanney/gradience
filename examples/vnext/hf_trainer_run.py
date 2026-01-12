#!/usr/bin/env python3
"""
Minimal HuggingFace + PEFT + Gradience example that runs on CPU.

This demonstrates the HF integration in ~50 lines of copy-pasteable code.
Produces: ./output_dir/run.jsonl with full Gradience telemetry.

Usage:
    python examples/vnext/hf_trainer_run.py

Requirements:
    pip install transformers datasets peft

Output:
    ./hf_example_output/
    ├── run.jsonl              # Gradience telemetry
    ├── adapter_config.json    # PEFT config
    ├── adapter_model.bin      # LoRA weights
    └── trainer_state.json     # HF training state
"""

import os
from pathlib import Path

# Gradience integration
from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig

# Minimal HF + PEFT setup
try:
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer, 
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install transformers datasets peft")
    exit(1)


def create_tiny_dataset(tokenizer, num_samples=50):
    """Create a minimal classification dataset for CPU training."""
    
    # Simple binary classification examples
    texts = [
        "This is a positive example",
        "This is a negative example", 
        "Great work on this project",
        "This could be improved",
        "Excellent results here",
    ] * (num_samples // 5 + 1)
    
    labels = [1, 0, 1, 0, 1] * (num_samples // 5 + 1)
    
    # Take exactly num_samples
    texts = texts[:num_samples]
    labels = labels[:num_samples]
    
    # Tokenize
    encoded = tokenizer(texts, truncation=True, padding=True, max_length=128)
    
    return Dataset.from_dict({
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"], 
        "labels": labels
    })


def main():
    print("🚀 HuggingFace + PEFT + Gradience CPU Example")
    print("=" * 50)
    
    # Use a tiny model for CPU demo
    model_name = "hf-internal-testing/tiny-random-distilbert"
    output_dir = Path("./hf_example_output")
    
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print()
    
    # 1. Load model and tokenizer
    print("📦 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True  # tiny model may have different head
    )
    
    # 2. Add LoRA adapters
    print("🔧 Adding LoRA adapters...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 3. Create tiny dataset
    print("📊 Creating tiny dataset...")
    train_dataset = create_tiny_dataset(tokenizer, num_samples=40)
    eval_dataset = create_tiny_dataset(tokenizer, num_samples=10)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # 4. Training arguments (CPU-optimized, minimal)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8, 
        learning_rate=3e-4,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # CPU-friendly
        fp16=False,  # CPU doesn't support FP16
        report_to=None,  # Disable wandb/tensorboard
        disable_tqdm=False,
    )
    
    # 5. Set up Gradience telemetry callback
    print("📡 Setting up Gradience telemetry...")
    gradience_config = GradienceCallbackConfig(
        output_dir=output_dir,
        filename="run.jsonl",
        dataset_name="tiny_classification",
        task_profile="easy_classification", 
        notes="CPU demo with DistilBERT + LoRA"
    )
    
    # 6. Create trainer with Gradience callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[GradienceCallback(gradience_config)]
    )
    
    # 7. Train!
    print("🏃 Starting training...")
    trainer.train()
    
    # 8. Save LoRA adapters
    print("💾 Saving LoRA adapters...")
    model.save_pretrained(output_dir)
    
    # 9. Show results
    print("✅ Training completed!")
    print(f"📁 Check outputs:")
    print(f"   • {output_dir}/run.jsonl (Gradience telemetry)")
    print(f"   • {output_dir}/adapter_config.json (LoRA config)")
    print(f"   • {output_dir}/adapter_model.bin (LoRA weights)")
    print()
    
    # Show telemetry file if it exists
    telemetry_file = output_dir / "run.jsonl"
    if telemetry_file.exists():
        with open(telemetry_file) as f:
            lines = f.readlines()
        
        print(f"📊 Telemetry summary ({len(lines)} events):")
        for i, line in enumerate(lines[:3]):  # Show first 3 events
            import json
            try:
                event = json.loads(line.strip())
                event_type = event.get("event", "unknown")
                print(f"   {i+1}. {event_type}")
            except:
                print(f"   {i+1}. (parse error)")
        
        if len(lines) > 3:
            print(f"   ... and {len(lines) - 3} more events")
        
        print(f"\n🔍 Inspect with: head {telemetry_file}")
        print(f"🔍 Or use: python -m gradience monitor {telemetry_file}")
    
    print("\n🎯 Next steps:")
    print("   • Run audit: python -m gradience audit --peft-dir hf_example_output")
    print("   • Get suggestions: python -m gradience audit --peft-dir hf_example_output --layers --suggest-per-layer --json")
    print("   • Monitor run: python -m gradience monitor hf_example_output/run.jsonl")


if __name__ == "__main__":
    main()