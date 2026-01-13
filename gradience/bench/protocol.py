"""
Bench protocol (v0.1).

Implements:
1) Train probe adapter (r=16)
2) Audit -> suggestions
3) Retrain: uniform_median, uniform_p90, per_layer
4) Eval all
5) Emit report (JSON + Markdown)

Step 3.1 implementation: Train probe with GradienceCallback
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Check for optional dependencies
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset
    HAS_TRAINING_DEPS = True
except ImportError:
    HAS_TRAINING_DEPS = False

# Gradience imports (always available)
from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_dataset(config: Dict[str, Any], smoke: bool = False):
    """Load and prepare dataset based on config."""
    if not HAS_TRAINING_DEPS:
        raise ImportError("Training dependencies not available (transformers, datasets, peft)")
    
    task_config = config["task"]
    dataset = load_dataset(task_config["dataset"], task_config["subset"])
    
    # Apply smoke test limits if requested
    if smoke:
        runtime = config.get("runtime", {})
        train_samples = runtime.get("smoke_train_samples", 200)
        eval_samples = runtime.get("smoke_eval_samples", 200)
        
        dataset["train"] = dataset["train"].select(range(min(len(dataset["train"]), train_samples)))
        if "validation" in dataset:
            dataset["validation"] = dataset["validation"].select(range(min(len(dataset["validation"]), eval_samples)))
    
    return dataset


def setup_model_and_tokenizer(config: Dict[str, Any], device: str = "cpu"):
    """Setup base model, tokenizer, and LoRA configuration."""
    if not HAS_TRAINING_DEPS:
        raise ImportError("Training dependencies not available (transformers, peft)")
    
    model_name = config["model"]["name"]
    lora_config = config["lora"]
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # SST-2 is binary classification
        torch_dtype=torch.float32 if device == "cpu" else torch.float16
    )
    
    # Setup LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_config["probe_r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"],
    )
    
    model = get_peft_model(model, peft_config)
    
    return tokenizer, model


def preprocess_function(examples, tokenizer):
    """Tokenize the examples and preserve labels."""
    result = tokenizer(examples["sentence"], truncation=True, padding=True, max_length=128)
    # Make sure labels are preserved
    if "label" in examples:
        result["labels"] = examples["label"]
    return result


def run_probe_training(
    config_path: str | Path,
    output_dir: str | Path, 
    smoke: bool = False
) -> Dict[str, Any]:
    """
    Step 3.1: Train probe adapter (r=16).
    
    Returns training results including accuracy and parameter counts.
    """
    if not HAS_TRAINING_DEPS:
        raise ImportError(
            "Training dependencies not available. "
            "Install: pip install transformers>=4.20.0 peft>=0.4.0 datasets torch"
        )
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup output directory for probe
    probe_dir = Path(output_dir) / "probe_r16"
    probe_dir.mkdir(parents=True, exist_ok=True)
    
    # Get device from config
    device = config.get("runtime", {}).get("device", "cpu")
    
    # Setup dataset
    dataset = setup_dataset(config, smoke=smoke)
    
    # Setup model and tokenizer
    tokenizer, model = setup_model_and_tokenizer(config, device=device)
    
    # Preprocess dataset
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=[col for col in dataset["train"].column_names if col not in ["labels", "label"]]
    )
    
    # Setup training arguments
    train_config = config["train"]
    runtime_config = config.get("runtime", {})
    
    max_steps = train_config["max_steps"]
    if smoke:
        max_steps = runtime_config.get("smoke_max_steps", 50)
    
    training_args = TrainingArguments(
        output_dir=str(probe_dir),
        num_train_epochs=1,  # We use max_steps instead
        max_steps=max_steps,
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
        eval_steps=train_config["eval_steps"],
        eval_strategy="steps",  # Updated from evaluation_strategy
        save_steps=train_config["eval_steps"],
        learning_rate=train_config["lr"],
        weight_decay=train_config["weight_decay"],
        logging_dir=str(probe_dir / "logs"),
        logging_steps=10,
        seed=train_config["seed"],
        data_seed=train_config["seed"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        report_to=[],  # Disable wandb/tensorboard
        remove_unused_columns=False,
    )
    
    # Setup data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Setup Gradience callback
    # Optional: pass dataset/task info for richer telemetry if available
    callback_config = GradienceCallbackConfig(
        output_dir=str(probe_dir),
        filename="run.jsonl"
    )
    
    # Add optional dataset/task context for richer telemetry
    task_config = config.get("task", {})
    if task_config.get("dataset") and task_config.get("subset"):
        dataset_name = f"{task_config['dataset']}/{task_config['subset']}"
        # Note: callback doesn't require these fields, but bench can provide them
        # for richer downstream monitor output
        # We'll pass them via environment or config if the callback supports it in future
    
    gradience_callback = GradienceCallback(callback_config)
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        accuracy = float((predictions == labels).mean())
        return {"accuracy": accuracy}
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation", tokenized_dataset["train"]),  # Fallback to train if no validation
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[gradience_callback],
    )
    
    # Train the model
    print(f"Starting probe training (r={config['lora']['probe_r']})...")
    print(f"Output dir: {probe_dir}")
    print(f"Max steps: {max_steps}")
    print(f"Device: {device}")
    
    trainer.train()
    
    # Evaluate final model
    eval_results = trainer.evaluate()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Probe training complete!")
    print(f"Final accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Telemetry written to: {probe_dir / 'run.jsonl'}")
    
    # Return results for the bench report
    return {
        "probe": {
            "rank": config["lora"]["probe_r"],
            "params": trainable_params,
            "total_params": total_params,
            "accuracy": eval_results["eval_accuracy"],
            "eval_loss": eval_results["eval_loss"],
            "output_dir": str(probe_dir)
        }
    }


def run_bench_protocol(
    config_path: str | Path,
    output_dir: str | Path,
    smoke: bool = False,
    ci: bool = False
) -> Dict[str, Any]:
    """
    Run the complete bench protocol.
    
    v0.1: Only implements Step 3.1 (probe training)
    Future: Steps 3.2-3.5 (audit, compress, retrain, report)
    """
    print("Gradience Bench Protocol v0.1")
    print("=" * 40)
    
    # Load configuration for validation
    config = load_config(config_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print(f"Model: {config['model']['name']}")
    print(f"Task: {config['task']['dataset']}/{config['task']['subset']}")
    print(f"Smoke mode: {smoke}")
    print()
    
    # Step 3.1: Train probe
    print("Step 3.1: Training probe adapter...")
    probe_results = run_probe_training(config_path, output_path, smoke=smoke)
    
    # Prepare report structure
    report = {
        "bench_version": config.get("bench_version", "0.1"),
        "model": config["model"]["name"],
        "task": f"{config['task']['dataset']}/{config['task']['subset']}",
        "config_path": str(config_path),
        "output_dir": str(output_path),
        "smoke_mode": smoke,
        **probe_results
    }
    
    print("\nStep 3.1 complete!")
    print("TODO: Steps 3.2-3.5 (audit → compress → retrain → eval → report)")
    
    return report