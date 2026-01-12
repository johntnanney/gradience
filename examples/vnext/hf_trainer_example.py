#!/usr/bin/env python3
"""
Minimal HuggingFace + Gradience example demonstrating "one line callback" integration.

This is the copy-pasteable example referenced in documentation and blog posts.
Trains a tiny model on CPU, generates telemetry, and shows next steps.

Usage:
    python examples/vnext/hf_trainer_example.py

Output:
    ./gradience_example_output/
    ├── run.jsonl              # Gradience telemetry
    ├── adapter_config.json    # PEFT configuration
    └── adapter_model.bin      # LoRA weights

Requirements:
    pip install transformers datasets peft
"""

import os
from pathlib import Path

def main():
    print("🚀 Gradience + HuggingFace: One Line Integration Example")
    print("=" * 60)
    
    # Check dependencies
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
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install transformers datasets peft")
        return 1
    
    # Import Gradience callback
    from gradience.vnext.integrations.hf import GradienceCallback
    
    # Allow override via environment variable (for smoke tests)
    output_dir = Path(os.environ.get("GRADIENCE_OUTPUT_DIR", "./gradience_example_output"))
    
    print(f"📦 Loading tiny model for CPU demonstration...")
    
    # 1. Load a tiny model for fast CPU training
    model_name = "hf-internal-testing/tiny-random-distilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    # 2. Add LoRA adapters  
    print("🔧 Adding LoRA adapters...")
    lora_config = LoraConfig(
        r=4,                    # Small rank for demo
        lora_alpha=8,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)
    
    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(f"LoRA parameters: {trainable_params:,} / {all_params:,} ({trainable_params/all_params*100:.1f}%)")
    
    # 3. Create tiny dataset
    print("📊 Creating tiny dataset...")
    texts = ["This is great!", "This is bad!"] * 10  # 20 samples
    labels = [1, 0] * 10
    
    encoded = tokenizer(texts, truncation=True, padding=True, max_length=64)
    
    # Split into train/eval
    train_dataset = Dataset.from_dict({
        "input_ids": encoded["input_ids"][:15],
        "attention_mask": encoded["attention_mask"][:15],
        "labels": labels[:15]  # 15 train
    })
    
    eval_dataset = Dataset.from_dict({
        "input_ids": encoded["input_ids"][15:],
        "attention_mask": encoded["attention_mask"][15:], 
        "labels": labels[15:]  # 5 eval
    })
    
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    # 4. Training arguments (CPU-optimized, minimal)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,         # Very short for demo
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        learning_rate=5e-4,
        logging_steps=2,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="epoch",
        remove_unused_columns=False,
        dataloader_num_workers=0,   # CPU-friendly
        fp16=False,                 # CPU doesn't support FP16
        report_to=None,             # No wandb/tensorboard
        disable_tqdm=False,
    )
    
    # 5. Create trainer with Gradience callback (THE ONE LINE!)
    print("📡 Adding Gradience telemetry with one line...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[GradienceCallback()]  # ← THE ONE LINE INTEGRATION
    )
    
    print("   ✅ Added: GradienceCallback()")
    print(f"   📁 Telemetry will be written to: {output_dir}/run.jsonl")
    
    # 6. Train!
    print("\n🏃 Training (this will take ~30 seconds)...")
    trainer.train()
    
    # 7. Save adapter
    print("💾 Saving LoRA adapter...")
    model.save_pretrained(output_dir)
    
    # 8. Show outputs
    print("\n✅ Training completed!")
    print(f"📁 Outputs saved to: {output_dir}")
    
    if (output_dir / "run.jsonl").exists():
        with open(output_dir / "run.jsonl") as f:
            events = len(f.readlines())
        print(f"📊 Telemetry: {events} events in run.jsonl")
    
    if (output_dir / "adapter_config.json").exists():
        print("🔧 LoRA adapter: adapter_config.json + weights saved")
    
    # 9. Show next steps
    print("\n🎯 Next Steps (copy-paste these commands):")
    print("")
    print("# Monitor the training run:")
    print(f"python -m gradience monitor {output_dir}/run.jsonl")
    print("")
    print("# Audit the LoRA adapter:")  
    print(f"python -m gradience audit --peft-dir {output_dir}")
    print("")
    print("# Audit + append telemetry:")
    print(f"python -m gradience audit --peft-dir {output_dir} --append {output_dir}/run.jsonl")
    print("")
    print("# Get rank suggestions:")
    print(f"python -m gradience audit --peft-dir {output_dir} --layers --suggest-per-layer --json")
    print("")
    
    print("💡 This example demonstrates:")
    print("   • One line integration: callbacks=[GradienceCallback()]")
    print("   • Automatic telemetry generation (gradience.vnext.telemetry/v1)")
    print("   • LoRA adapter ready for audit and rank suggestions")
    print("   • Complete observability workflow: train → monitor → audit")
    
    return 0

if __name__ == "__main__":
    exit(main())