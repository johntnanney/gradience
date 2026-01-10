"""
Intervention Ablation Study: Proving Diagnosis → Action
"""
import argparse, json, math, os, sys, time, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

@dataclass
class ExperimentConfig:
    model_name: str = "distilbert-base-uncased"
    task: str = "sst2"
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    learning_rate: float = 3e-4
    batch_size: int = 32
    num_epochs: int = 3
    max_train_samples: int = 5000
    max_eval_samples: int = 1000
    num_seeds: int = 5
    seed: int = 42

@dataclass
class ConditionResult:
    condition: str
    seed: int
    total_params: int
    trainable_params: int
    final_accuracy: float
    training_time: float
    lora_config: Dict

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random; random.seed(seed)
    import numpy as np; np.random.seed(seed)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def run_single_condition(config, condition, lora_rank, target_modules, rank_pattern=None, seed=42, device="cuda"):
    set_seed(seed)
    print(f"\n--- {condition} (seed={seed}) ---")
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    lora_config_dict = {"task_type": TaskType.SEQ_CLS, "r": lora_rank, "lora_alpha": config.lora_alpha, "lora_dropout": config.lora_dropout, "target_modules": target_modules}
    if rank_pattern:
        lora_config_dict["rank_pattern"] = rank_pattern
        print(f"  Using rank_pattern: {rank_pattern}")
    try:
        lora_config = LoraConfig(**lora_config_dict)
    except TypeError:
        if rank_pattern: print("  WARNING: rank_pattern not supported, using uniform rank")
        lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=target_modules)
    model = get_peft_model(model, lora_config)
    model.to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"  Trainable params: {trainable_params:,}")
    dataset = load_dataset("glue", config.task)
    def tokenize(examples): return tokenizer(examples["sentence"], truncation=True, padding=False, max_length=128)
    train_dataset = dataset["train"].select(range(min(config.max_train_samples, len(dataset["train"]))))
    eval_dataset = dataset["validation"].select(range(min(config.max_eval_samples, len(dataset["validation"]))))
    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    train_dataset.set_format("torch"); eval_dataset.set_format("torch")
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start_time = time.time()
    for epoch in range(config.num_epochs):
        model.train(); total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch); loss = outputs.loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{config.num_epochs}: loss={total_loss/len(train_loader):.4f}")
    training_time = time.time() - start_time
    model.eval(); correct = 0; total = 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch); preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item(); total += len(batch["labels"])
    accuracy = correct / total
    print(f"  Final accuracy: {accuracy:.1%}")
    return ConditionResult(condition=condition, seed=seed, total_params=total_params, trainable_params=trainable_params, final_accuracy=accuracy, training_time=training_time, lora_config={"rank": lora_rank, "alpha": config.lora_alpha, "target_modules": target_modules, "rank_pattern": rank_pattern})

def run_experiment(output_dir, config, device="cuda"):
    output_path = Path(output_dir); output_path.mkdir(parents=True, exist_ok=True)
    print("="*70 + "\nINTERVENTION ABLATION STUDY\n" + "="*70)
    print(f"\nSeeds: {config.num_seeds}, Epochs: {config.num_epochs}, Train samples: {config.max_train_samples}")
    all_modules = ["q_lin", "k_lin", "v_lin", "lin1", "lin2"]
    conditions = {"baseline": {"rank": 64, "target_modules": all_modules, "rank_pattern": None}, "intervention": {"rank": 64, "target_modules": all_modules, "rank_pattern": {"v_lin": 32, "lin2": 32}}}
    all_results = {}
    for cond_name, cond_config in conditions.items():
        print(f"\n{'='*70}\nCONDITION: {cond_name.upper()}\n{'='*70}")
        cond_results = []
        for seed in range(config.num_seeds):
            result = run_single_condition(config, cond_name, cond_config["rank"], cond_config["target_modules"], cond_config["rank_pattern"], config.seed + seed, device)
            cond_results.append(result)
        all_results[cond_name] = cond_results
    baseline_params = all_results["baseline"][0].trainable_params
    print("\n" + "="*70 + "\nRESULTS SUMMARY\n" + "="*70)
    for cond_name, results in all_results.items():
        accs = [r.final_accuracy for r in results]; params = results[0].trainable_params
        mean_acc = sum(accs)/len(accs); std_acc = (sum((a-mean_acc)**2 for a in accs)/len(accs))**0.5
        savings = (baseline_params - params) / baseline_params if cond_name != "baseline" else 0
        print(f"{cond_name}: {mean_acc:.1%} ± {std_acc:.1%}, params={params:,}, savings={savings:.1%}")
    with open(output_path / "results.json", "w") as f:
        json.dump({"per_seed": {n: [asdict(r) for r in rs] for n, rs in all_results.items()}}, f, indent=2, default=str)
    print(f"\nResults saved to {output_path / 'results.json'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./intervention_results")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig(num_seeds=1 if args.quick else args.seeds, num_epochs=1 if args.quick else args.epochs, max_train_samples=1000 if args.quick else 5000, max_eval_samples=500 if args.quick else 1000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if args.quick: print("QUICK MODE: 1 seed, 1 epoch")
    run_experiment(args.output_dir, config, device)

if __name__ == "__main__": main()
