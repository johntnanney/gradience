"""
Experiment 1: LoRA Rank Collapse Demonstration

HYPOTHESIS: Even with nominal rank-64, effective rank often collapses to much lower.

This experiment trains LoRA adapters on a simple classification task and measures
the effective rank of BA throughout training. We expect to see:
- Effective rank << nominal rank
- Collapse happens early in training
- Final effective rank depends on task complexity

This is the killer demo: "You're paying for rank-64, but using rank-5"

Usage:
    python experiments/lora/exp1_rank_collapse.py
    
    # Quick mode
    python experiments/lora/exp1_rank_collapse.py --quick
"""

import json
import math
import re
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Check dependencies
try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset
    import evaluate
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install transformers peft datasets evaluate")
    exit(1)


@dataclass
class LoRASnapshot:
    """Snapshot of LoRA adapter state."""
    step: int
    layer_name: str
    nominal_rank: int
    effective_rank_A: float
    effective_rank_B: float
    effective_rank_BA: float
    rank_utilization: float
    kappa_A: float
    kappa_B: float
    frobenius_BA: float


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    model_name: str = "distilbert-base-uncased"
    task: str = "sst2"
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    learning_rate: float = 3e-4
    batch_size: int = 32
    num_epochs: int = 3
    max_train_samples: int = 5000
    max_eval_samples: int = 1000
    snapshot_interval: int = 50
    seed: int = 42


def compute_effective_rank(S: torch.Tensor) -> float:
    """Compute effective rank from singular values using entropy."""
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    S_norm = S / S.sum()
    entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum().item()
    return math.exp(entropy)


def compute_kappa(S: torch.Tensor) -> float:
    """Compute condition number from singular values."""
    S = S[S > 1e-10]
    if len(S) < 2:
        return 1.0
    return (S[0] / S[-1]).item()


def analyze_lora_layer(name: str, lora_A: torch.Tensor, lora_B: torch.Tensor, step: int) -> LoRASnapshot:
    """Analyze a single LoRA adapter layer."""
    with torch.no_grad():
        A = lora_A.float()
        B = lora_B.float()
        
        # Nominal rank
        nominal_rank = min(A.shape)
        
        # Analyze A
        S_A = torch.linalg.svdvals(A)
        eff_rank_A = compute_effective_rank(S_A)
        kappa_A = compute_kappa(S_A)
        
        # Analyze B
        S_B = torch.linalg.svdvals(B)
        eff_rank_B = compute_effective_rank(S_B)
        kappa_B = compute_kappa(S_B)
        
        # Analyze product BA
        BA = B @ A
        S_BA = torch.linalg.svdvals(BA)
        eff_rank_BA = compute_effective_rank(S_BA)
        frob_BA = torch.norm(BA, 'fro').item()
        
        utilization = eff_rank_BA / nominal_rank
        
        return LoRASnapshot(
            step=step,
            layer_name=name,
            nominal_rank=nominal_rank,
            effective_rank_A=eff_rank_A,
            effective_rank_B=eff_rank_B,
            effective_rank_BA=eff_rank_BA,
            rank_utilization=utilization,
            kappa_A=kappa_A,
            kappa_B=kappa_B,
            frobenius_BA=frob_BA,
        )


def find_lora_layers(model) -> Dict[str, Dict[str, nn.Parameter]]:
    """Find all LoRA adapter pairs in the model."""
    import re
    adapters = {}
    
    lora_a_params = {}
    lora_b_params = {}
    
    for name, param in model.named_parameters():
        if 'lora_A' in name and 'weight' in name:
            base_name = re.sub(r'\.lora_A\..*', '', name)
            lora_a_params[base_name] = param
        elif 'lora_B' in name and 'weight' in name:
            base_name = re.sub(r'\.lora_B\..*', '', name)
            lora_b_params[base_name] = param
    
    for base_name in lora_a_params:
        if base_name in lora_b_params:
            adapters[base_name] = {
                'A': lora_a_params[base_name],
                'B': lora_b_params[base_name],
            }
    
    return adapters


def run_experiment(output_dir: Path, quick: bool = False) -> Dict:
    """Run the rank collapse experiment."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Config
    config = ExperimentConfig()
    if quick:
        config.num_epochs = 1
        config.max_train_samples = 1000
        config.max_eval_samples = 500
        config.snapshot_interval = 25
    
    print(f"Config: {asdict(config)}")
    
    # Set seed
    torch.manual_seed(config.seed)
    
    # Load model
    print(f"\nLoading {config.model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Apply LoRA
    print(f"Applying LoRA (rank={config.lora_rank}, alpha={config.lora_alpha})...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_lin", "v_lin"],  # DistilBERT naming
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    
    # Find LoRA layers
    lora_layers = find_lora_layers(model)
    print(f"Found {len(lora_layers)} LoRA adapter pairs")
    
    # Load data
    print(f"\nLoading {config.task} dataset...")
    dataset = load_dataset("glue", config.task)
    
    def tokenize(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
    
    train_dataset = dataset["train"].select(range(min(config.max_train_samples, len(dataset["train"]))))
    eval_dataset = dataset["validation"].select(range(min(config.max_eval_samples, len(dataset["validation"]))))
    
    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)
    
    train_dataset = train_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.rename_column("label", "labels")
    
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    num_training_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )
    
    # Tracking
    snapshots: List[LoRASnapshot] = []
    training_history = []
    
    # Initial snapshot
    print("\nInitial LoRA state:")
    for layer_name, adapter in lora_layers.items():
        snap = analyze_lora_layer(layer_name, adapter['A'], adapter['B'], step=0)
        snapshots.append(snap)
        print(f"  {layer_name}: eff_rank={snap.effective_rank_BA:.1f} / {snap.nominal_rank} ({snap.rank_utilization:.0%})")
    
    # Training loop
    print(f"\nTraining for {config.num_epochs} epochs...")
    global_step = 0
    
    for epoch in range(config.num_epochs):
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
            global_step += 1
            
            # Take snapshots
            if global_step % config.snapshot_interval == 0:
                for layer_name, adapter in lora_layers.items():
                    snap = analyze_lora_layer(layer_name, adapter['A'], adapter['B'], step=global_step)
                    snapshots.append(snap)
                
                # Aggregate utilization
                current_utils = [s.rank_utilization for s in snapshots if s.step == global_step]
                avg_util = sum(current_utils) / len(current_utils) if current_utils else 0.0
                
                training_history.append({
                    'step': global_step,
                    'loss': loss.item(),
                    'avg_rank_utilization': avg_util,
                })
        
        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch["labels"]).sum().item()
                total += len(batch["labels"])
        
        accuracy = correct / total
        
        # Get current utilization
        current_snaps = [s for s in snapshots if s.step == global_step]
        avg_util = sum(s.rank_utilization for s in current_snaps) / len(current_snaps) if current_snaps else 0.0
        min_util = min((s.rank_utilization for s in current_snaps), default=0.0)
        
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Rank utilization: {avg_util:.0%} avg, {min_util:.0%} min")
    
    # Final analysis
    print("\n" + "=" * 60)
    print("FINAL ANALYSIS")
    print("=" * 60)
    
    # Get last snapshot for each layer
    last_step = max(s.step for s in snapshots) if snapshots else 0
    final_snaps = [s for s in snapshots if s.step == last_step]
    
    print(f"\nNominal rank: {config.lora_rank}")
    print("\nPer-layer effective rank:")
    
    for snap in final_snaps:
        status = "⚠️" if snap.rank_utilization < 0.5 else "✓"
        print(f"  {status} {snap.layer_name}")
        print(f"      Effective rank: {snap.effective_rank_BA:.1f} / {snap.nominal_rank}")
        print(f"      Utilization: {snap.rank_utilization:.0%}")
        print(f"      κ(A)={snap.kappa_A:.0f}, κ(B)={snap.kappa_B:.0f}")
    
    # Summary statistics
    avg_effective = sum(s.effective_rank_BA for s in final_snaps) / len(final_snaps) if final_snaps else 0.0
    avg_util = sum(s.rank_utilization for s in final_snaps) / len(final_snaps) if final_snaps else 0.0
    
    print(f"\n--- SUMMARY ---")
    print(f"Configured rank: {config.lora_rank}")
    print(f"Average effective rank: {avg_effective:.1f}")
    print(f"Average utilization: {avg_util:.0%}")
    print(f"Final accuracy: {accuracy:.1%}")
    
    if avg_util < 0.5:
        suggested_rank = max(4, int(avg_effective * 2))
        # Round to common value
        for r in [4, 8, 16, 32, 64]:
            if r >= suggested_rank:
                suggested_rank = r
                break
        print(f"\n💡 RECOMMENDATION: Rank-{suggested_rank} would likely achieve similar results")
        print(f"   Potential memory savings: ~{(config.lora_rank - suggested_rank) / config.lora_rank:.0%}")
    
    # Save results
    results = {
        'config': asdict(config),
        'final_accuracy': accuracy,
        'final_avg_effective_rank': avg_effective,
        'final_avg_utilization': avg_util,
        'suggested_rank': suggested_rank if avg_util < 0.5 else config.lora_rank,
        'snapshots': [asdict(s) for s in snapshots],
        'training_history': training_history,
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")
    
    # Generate plot data
    plot_data = {
        'steps': [h['step'] for h in training_history],
        'utilization': [h['avg_rank_utilization'] for h in training_history],
        'loss': [h['loss'] for h in training_history],
    }
    
    with open(output_dir / 'plot_data.json', 'w') as f:
        json.dump(plot_data, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='./lora_experiments/rank_collapse')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_experiment(output_dir, quick=args.quick)


if __name__ == "__main__":
    main()
