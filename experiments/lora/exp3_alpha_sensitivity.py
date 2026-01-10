"""
Experiment 3: LoRA Alpha/Rank Sensitivity Study

HYPOTHESIS: The ratio α/r controls effective learning rate and stability.
Suboptimal ratios cause training instability or slow convergence.

This experiment fixes rank and varies alpha to characterize:
- Stability (gradient variance, loss spikes)
- Convergence speed
- Final performance
- Effective rank utilization

Common conventions:
- α = r (most common default)
- α = 2r (more aggressive)
- α = r/2 (more conservative)

Usage:
    python experiments/lora/exp3_alpha_sensitivity.py
"""

import json
import math
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
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
    exit(1)


@dataclass
class AlphaRunResult:
    """Result from a single alpha configuration."""
    rank: int
    alpha: int
    alpha_ratio: float  # alpha / rank
    final_accuracy: float
    avg_effective_rank: float
    rank_utilization: float
    # Stability metrics
    loss_variance: float
    max_loss_spike: float
    gradient_norm_mean: float
    gradient_norm_std: float
    # Convergence
    steps_to_90pct: int  # Steps to reach 90% of final accuracy


def compute_effective_rank(S: torch.Tensor) -> float:
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    S_norm = S / S.sum()
    entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum().item()
    return math.exp(entropy)


def analyze_lora_effective_rank(model) -> float:
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


def train_with_alpha(
    rank: int,
    alpha: int,
    model_name: str,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> AlphaRunResult:
    """Train with specific alpha and track stability metrics."""
    
    ratio = alpha / rank
    print(f"\n--- rank={rank}, alpha={alpha} (ratio={ratio:.2f}) ---")
    
    # Load fresh model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=50, num_training_steps=num_training_steps,
    )
    
    # Tracking
    losses = []
    grad_norms = []
    accuracies = []
    steps_to_90pct = -1
    
    global_step = 0
    eval_interval = max(1, len(train_loader) // 5)  # 5 evals per epoch
    
    for epoch in range(num_epochs):
        model.train()
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(loss.item())
            
            loss.backward()
            
            # Track gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            grad_norms.append(total_norm ** 0.5)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Periodic evaluation
            if global_step % eval_interval == 0:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                        outputs = model(**eval_batch)
                        preds = torch.argmax(outputs.logits, dim=-1)
                        correct += (preds == eval_batch["labels"]).sum().item()
                        total += len(eval_batch["labels"])
                acc = correct / total
                accuracies.append((global_step, acc))
                model.train()
    
    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])
    
    final_accuracy = correct / total
    
    # Calculate steps to 90% of final
    target = final_accuracy * 0.9
    for step, acc in accuracies:
        if acc >= target:
            steps_to_90pct = step
            break
    
    # Stability metrics
    loss_variance = torch.tensor(losses).var().item()
    
    # Find max spike (max increase between consecutive losses)
    loss_diffs = [losses[i+1] - losses[i] for i in range(len(losses)-1)]
    max_spike = max(loss_diffs) if loss_diffs else 0.0
    
    grad_norm_mean = sum(grad_norms) / len(grad_norms)
    grad_norm_std = torch.tensor(grad_norms).std().item()
    
    # Effective rank
    avg_eff_rank = analyze_lora_effective_rank(model)
    utilization = avg_eff_rank / rank
    
    print(f"  Accuracy: {final_accuracy:.1%}")
    print(f"  Loss variance: {loss_variance:.4f}")
    print(f"  Grad norm: {grad_norm_mean:.2f} ± {grad_norm_std:.2f}")
    print(f"  Steps to 90%: {steps_to_90pct}")
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return AlphaRunResult(
        rank=rank,
        alpha=alpha,
        alpha_ratio=ratio,
        final_accuracy=final_accuracy,
        avg_effective_rank=avg_eff_rank,
        rank_utilization=utilization,
        loss_variance=loss_variance,
        max_loss_spike=max_spike,
        gradient_norm_mean=grad_norm_mean,
        gradient_norm_std=grad_norm_std,
        steps_to_90pct=steps_to_90pct,
    )


def run_experiment(output_dir: Path, quick: bool = False) -> Dict:
    """Run the alpha sensitivity experiment."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model_name = "distilbert-base-uncased"
    task = "sst2"
    learning_rate = 3e-4
    batch_size = 32
    
    if quick:
        num_epochs = 1
        max_train = 1000
        max_eval = 500
        rank = 16
        alphas = [8, 16, 32]  # ratio: 0.5, 1.0, 2.0
    else:
        num_epochs = 3
        max_train = 5000
        max_eval = 1000
        rank = 16
        alphas = [4, 8, 16, 32, 64]  # ratios: 0.25, 0.5, 1.0, 2.0, 4.0
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load data
    print(f"\nLoading {task} dataset...")
    dataset = load_dataset("glue", task)
    
    def tokenize(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)
    
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
    results: List[AlphaRunResult] = []
    
    for alpha in alphas:
        result = train_with_alpha(
            rank=rank,
            alpha=alpha,
            model_name=model_name,
            train_loader=train_loader,
            eval_loader=eval_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
        )
        results.append(result)
    
    # Analysis
    print("\n" + "=" * 70)
    print("ALPHA SENSITIVITY RESULTS")
    print("=" * 70)
    
    print(f"\nFixed rank: {rank}")
    print(f"\n{'Alpha':>6} {'Ratio':>6} {'Acc':>8} {'LossVar':>10} {'GradNorm':>12} {'90%Steps':>10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r.alpha:>6} {r.alpha_ratio:>6.2f} {r.final_accuracy:>8.1%} "
              f"{r.loss_variance:>10.4f} {r.gradient_norm_mean:>6.1f}±{r.gradient_norm_std:<5.1f} "
              f"{r.steps_to_90pct:>10}")
    
    # Find sweet spot
    # Best is highest accuracy with reasonable stability
    stable_results = [r for r in results if r.loss_variance < 0.1]  # Threshold
    if stable_results:
        best_stable = max(stable_results, key=lambda r: r.final_accuracy)
        print(f"\n--- ANALYSIS ---")
        print(f"Most stable high-performer: alpha={best_stable.alpha} (ratio={best_stable.alpha_ratio})")
        print(f"  Accuracy: {best_stable.final_accuracy:.1%}")
        print(f"  Loss variance: {best_stable.loss_variance:.4f}")
    
    # Check for instability at high alpha
    high_alpha = [r for r in results if r.alpha_ratio >= 2.0]
    if high_alpha and any(r.loss_variance > 0.05 for r in high_alpha):
        print(f"\n⚠️  High alpha ratios (≥2.0) show increased instability")
    
    # Save results
    output = {
        'config': {
            'model_name': model_name,
            'task': task,
            'rank': rank,
            'alphas_tested': alphas,
            'num_epochs': num_epochs,
        },
        'results': [asdict(r) for r in results],
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")
    
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='./lora_experiments/alpha_sensitivity')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_experiment(output_dir, quick=args.quick)


if __name__ == "__main__":
    main()
