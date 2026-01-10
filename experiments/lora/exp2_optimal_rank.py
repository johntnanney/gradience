"""
Experiment 2: Optimal LoRA Rank Search

HYPOTHESIS: Lower ranks achieve similar performance to high ranks on simple tasks.

This experiment trains LoRA adapters with different ranks (4, 8, 16, 32, 64)
and compares:
- Final accuracy
- Training cost (time, memory)
- Effective rank utilization

We expect to find that rank-16 or rank-32 matches rank-64 for most tasks,
validating our rank suggestion algorithm.

Usage:
    python experiments/lora/exp2_optimal_rank.py
    
    # Quick mode
    python experiments/lora/exp2_optimal_rank.py --quick
"""

import json
import math
import os
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install transformers peft datasets evaluate")
    exit(1)


@dataclass
class RunResult:
    """Result from a single training run."""
    rank: int
    alpha: int
    final_accuracy: float
    final_loss: float
    avg_effective_rank: float
    rank_utilization: float
    training_time_seconds: float
    trainable_params: int
    peak_memory_mb: float


def compute_effective_rank(S: torch.Tensor) -> float:
    """Compute effective rank from singular values using entropy."""
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    S_norm = S / S.sum()
    entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum().item()
    return math.exp(entropy)


def analyze_lora_effective_rank(model) -> float:
    """Get average effective rank across all LoRA adapters."""
    effective_ranks = []
    
    lora_A_params = {}
    lora_B_params = {}
    
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            base = name.replace('.lora_A.default', '').replace('.lora_A', '')
            lora_A_params[base] = param
        elif 'lora_B' in name:
            base = name.replace('.lora_B.default', '').replace('.lora_B', '')
            lora_B_params[base] = param
    
    for base in lora_A_params:
        if base in lora_B_params:
            A = lora_A_params[base].float()
            B = lora_B_params[base].float()
            
            with torch.no_grad():
                BA = B @ A
                S = torch.linalg.svdvals(BA)
                eff_rank = compute_effective_rank(S)
                effective_ranks.append(eff_rank)
    
    return sum(effective_ranks) / len(effective_ranks) if effective_ranks else 0.0


def train_with_rank(
    rank: int,
    alpha: int,
    model_name: str,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    tokenizer,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> RunResult:
    """Train a model with specific LoRA rank and return results."""
    
    print(f"\n--- Training with rank={rank}, alpha={alpha} ---")
    
    # Load fresh model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    model.to(device)
    
    # Reset peak memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=50,
        num_training_steps=num_training_steps,
    )
    
    # Training
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")
    
    training_time = time.time() - start_time
    
    # Get peak memory
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_memory = 0.0
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += len(batch["labels"])
            total_loss += outputs.loss.item()
    
    accuracy = correct / total
    final_loss = total_loss / len(eval_loader)
    
    # Analyze effective rank
    avg_effective = analyze_lora_effective_rank(model)
    utilization = avg_effective / rank
    
    print(f"  Final: acc={accuracy:.1%}, eff_rank={avg_effective:.1f}/{rank} ({utilization:.0%})")
    
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return RunResult(
        rank=rank,
        alpha=alpha,
        final_accuracy=accuracy,
        final_loss=final_loss,
        avg_effective_rank=avg_effective,
        rank_utilization=utilization,
        training_time_seconds=training_time,
        trainable_params=trainable_params,
        peak_memory_mb=peak_memory,
    )


def run_experiment(output_dir: Path, quick: bool = False) -> Dict:
    """Run the optimal rank search experiment."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Config
    model_name = "distilbert-base-uncased"
    task = "sst2"
    learning_rate = 3e-4
    batch_size = 32
    
    if quick:
        num_epochs = 1
        max_train = 1000
        max_eval = 500
        ranks_to_test = [8, 32, 64]
    else:
        num_epochs = 3
        max_train = 5000
        max_eval = 1000
        ranks_to_test = [4, 8, 16, 32, 64]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load data (once for all runs)
    print(f"\nLoading {task} dataset...")
    dataset = load_dataset("glue", task)
    
    def tokenize(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
    
    train_dataset = dataset["train"].select(range(min(max_train, len(dataset["train"]))))
    eval_dataset = dataset["validation"].select(range(min(max_eval, len(dataset["validation"]))))
    
    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)
    
    train_dataset = train_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.rename_column("label", "labels")
    
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    
    # Run experiments
    results: List[RunResult] = []
    
    for rank in ranks_to_test:
        alpha = rank  # Common default: alpha = rank
        
        result = train_with_rank(
            rank=rank,
            alpha=alpha,
            model_name=model_name,
            train_loader=train_loader,
            eval_loader=eval_loader,
            tokenizer=tokenizer,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
        )
        results.append(result)
    
    # Analysis
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Rank':>6} {'Accuracy':>10} {'Eff.Rank':>10} {'Util':>8} {'Params':>12} {'Time':>8}")
    print("-" * 60)
    
    for r in results:
        print(f"{r.rank:>6} {r.final_accuracy:>10.1%} {r.avg_effective_rank:>10.1f} "
              f"{r.rank_utilization:>8.0%} {r.trainable_params:>12,} {r.training_time_seconds:>7.1f}s")
    
    # Find optimal rank
    best_result = max(results, key=lambda r: r.final_accuracy)
    best_accuracy = best_result.final_accuracy
    
    # Find minimum rank that achieves within 1% of best
    threshold = best_accuracy - 0.01
    efficient_ranks = [r for r in results if r.final_accuracy >= threshold]
    optimal = min(efficient_ranks, key=lambda r: r.rank)
    
    print(f"\n--- ANALYSIS ---")
    print(f"Best accuracy: {best_accuracy:.1%} (rank={best_result.rank})")
    print(f"Optimal rank (within 1% of best): {optimal.rank}")
    print(f"  Achieves: {optimal.final_accuracy:.1%}")
    print(f"  Effective rank: {optimal.avg_effective_rank:.1f}")
    
    if optimal.rank < max(ranks_to_test):
        highest = [r for r in results if r.rank == max(ranks_to_test)][0]
        param_savings = (highest.trainable_params - optimal.trainable_params) / highest.trainable_params
        time_savings = (highest.training_time_seconds - optimal.training_time_seconds) / highest.training_time_seconds
        
        print(f"\n💡 Using rank-{optimal.rank} instead of rank-{highest.rank}:")
        print(f"   Parameter reduction: {param_savings:.0%}")
        print(f"   Time savings: {time_savings:.0%}")
        print(f"   Accuracy difference: {optimal.final_accuracy - highest.final_accuracy:+.1%}")
    
    # Save results
    output = {
        'config': {
            'model_name': model_name,
            'task': task,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'ranks_tested': ranks_to_test,
        },
        'results': [asdict(r) for r in results],
        'analysis': {
            'best_rank': best_result.rank,
            'best_accuracy': best_accuracy,
            'optimal_rank': optimal.rank,
            'optimal_accuracy': optimal.final_accuracy,
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")
    
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='./lora_experiments/optimal_rank')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_experiment(output_dir, quick=args.quick)


if __name__ == "__main__":
    main()
