"""
Experiment 5: Layer-wise LoRA Utilization Patterns

HYPOTHESIS: Different layer types (attention vs FFN, early vs late) have
different rank requirements. A uniform rank across all layers is suboptimal.

This experiment applies LoRA to multiple layer types and measures:
- Per-layer effective rank
- Utilization patterns by layer type
- Utilization patterns by layer depth

Product implications:
- "Attention layers use 80% of rank, FFN only 30%"
- "Consider rank-32 for attention, rank-8 for FFN"
- Layer-specific rank recommendations

Usage:
    python experiments/lora/exp5_layer_utilization.py
"""

import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import argparse
import re

import torch
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
class LayerMetrics:
    """Metrics for a single LoRA layer."""
    name: str
    layer_type: str  # "attention_q", "attention_v", "ffn", etc.
    layer_idx: int   # Layer number (0, 1, 2, ...)
    nominal_rank: int
    effective_rank: float
    utilization: float
    kappa: float


@dataclass 
class LayerUtilizationResult:
    """Results from the layer utilization experiment."""
    task: str
    rank: int
    final_accuracy: float
    layers: List[LayerMetrics]
    # Aggregates
    avg_utilization_by_type: Dict[str, float]
    avg_utilization_by_depth: Dict[str, float]  # "early", "middle", "late"


def compute_effective_rank(S: torch.Tensor) -> float:
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    S_norm = S / S.sum()
    entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum().item()
    return math.exp(entropy)


def compute_kappa(S: torch.Tensor) -> float:
    S = S[S > 1e-10]
    if len(S) < 2:
        return 1.0
    return (S[0] / S[-1]).item()


def parse_layer_info(name: str) -> Tuple[str, int]:
    """Extract layer type and index from parameter name."""
    
    # Determine layer type
    name_lower = name.lower()
    if 'q_lin' in name_lower or 'q_proj' in name_lower or '.q.' in name_lower:
        layer_type = 'attention_q'
    elif 'k_lin' in name_lower or 'k_proj' in name_lower or '.k.' in name_lower:
        layer_type = 'attention_k'
    elif 'v_lin' in name_lower or 'v_proj' in name_lower or '.v.' in name_lower:
        layer_type = 'attention_v'
    elif 'o_lin' in name_lower or 'o_proj' in name_lower or '.o.' in name_lower:
        layer_type = 'attention_o'
    elif 'lin1' in name_lower or 'fc1' in name_lower or 'up_proj' in name_lower:
        layer_type = 'ffn_up'
    elif 'lin2' in name_lower or 'fc2' in name_lower or 'down_proj' in name_lower:
        layer_type = 'ffn_down'
    else:
        layer_type = 'other'
    
    # Extract layer index
    match = re.search(r'layer\.(\d+)', name) or re.search(r'layers\.(\d+)', name)
    layer_idx = int(match.group(1)) if match else -1
    
    return layer_type, layer_idx


def analyze_all_lora_layers(model, rank: int) -> List[LayerMetrics]:
    """Analyze all LoRA adapters and return per-layer metrics."""
    
    # Find all A/B pairs
    lora_A_params = {}
    lora_B_params = {}
    
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            base = name.replace('.lora_A.default', '').replace('.lora_A', '')
            lora_A_params[base] = (name, param)
        elif 'lora_B' in name:
            base = name.replace('.lora_B.default', '').replace('.lora_B', '')
            lora_B_params[base] = (name, param)
    
    layer_metrics = []
    
    for base in lora_A_params:
        if base not in lora_B_params:
            continue
        
        A_name, A = lora_A_params[base]
        B_name, B = lora_B_params[base]
        
        with torch.no_grad():
            A = A.float()
            B = B.float()
            BA = B @ A
            
            S = torch.linalg.svdvals(BA)
            eff_rank = compute_effective_rank(S)
            kappa = compute_kappa(S)
        
        layer_type, layer_idx = parse_layer_info(base)
        
        layer_metrics.append(LayerMetrics(
            name=base,
            layer_type=layer_type,
            layer_idx=layer_idx,
            nominal_rank=rank,
            effective_rank=eff_rank,
            utilization=eff_rank / rank,
            kappa=kappa,
        ))
    
    return layer_metrics


def run_experiment(output_dir: Path, quick: bool = False) -> Dict:
    """Run the layer utilization experiment."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model_name = "distilbert-base-uncased"
    task = "sst2"
    rank = 64
    learning_rate = 3e-4
    batch_size = 32
    
    if quick:
        num_epochs = 1
        max_train = 1000
        max_eval = 500
    else:
        num_epochs = 3
        max_train = 5000
        max_eval = 1000
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load data
    print(f"\nLoading {task} dataset...")
    dataset = load_dataset("glue", task)
    
    def tokenize(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)
    
    train_data = dataset["train"].select(range(min(max_train, len(dataset["train"]))))
    eval_data = dataset["validation"].select(range(min(max_eval, len(dataset["validation"]))))
    
    train_data = train_data.map(tokenize, batched=True)
    eval_data = eval_data.map(tokenize, batched=True)
    train_data = train_data.rename_column("label", "labels")
    eval_data = eval_data.rename_column("label", "labels")
    train_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size)
    
    # Create model with LoRA on multiple layer types
    print(f"\nApplying LoRA (rank={rank}) to attention and FFN layers...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )
    
    # Apply to both attention and FFN
    # DistilBERT: q_lin, k_lin, v_lin (attention), lin1, lin2 (FFN)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.1,
        target_modules=["q_lin", "k_lin", "v_lin", "lin1", "lin2"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    
    # Initial analysis
    print("\nInitial layer analysis:")
    initial_metrics = analyze_all_lora_layers(model, rank)
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 50, num_steps)
    
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")
    
    # Evaluate
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
    
    accuracy = correct / total
    print(f"\nFinal accuracy: {accuracy:.1%}")
    
    # Final layer analysis
    final_metrics = analyze_all_lora_layers(model, rank)
    
    # Aggregate by type
    by_type: Dict[str, List[float]] = {}
    for m in final_metrics:
        if m.layer_type not in by_type:
            by_type[m.layer_type] = []
        by_type[m.layer_type].append(m.utilization)
    
    avg_by_type = {t: sum(v)/len(v) for t, v in by_type.items()}
    
    # Aggregate by depth
    max_idx = max(m.layer_idx for m in final_metrics if m.layer_idx >= 0)
    by_depth: Dict[str, List[float]] = {'early': [], 'middle': [], 'late': []}
    
    for m in final_metrics:
        if m.layer_idx < 0:
            continue
        if m.layer_idx <= max_idx // 3:
            by_depth['early'].append(m.utilization)
        elif m.layer_idx <= 2 * max_idx // 3:
            by_depth['middle'].append(m.utilization)
        else:
            by_depth['late'].append(m.utilization)
    
    avg_by_depth = {d: sum(v)/len(v) if v else 0 for d, v in by_depth.items()}
    
    # Report
    print("\n" + "=" * 70)
    print("LAYER-WISE UTILIZATION ANALYSIS")
    print("=" * 70)
    
    print(f"\n--- BY LAYER TYPE ---")
    print(f"{'Type':<15} {'Avg Util':>10} {'Suggested Rank':>15}")
    print("-" * 40)
    
    for layer_type, util in sorted(avg_by_type.items(), key=lambda x: -x[1]):
        # Suggest rank based on utilization
        eff = util * rank
        suggested = max(4, int(eff * 1.5))
        for r in [4, 8, 16, 32, 64]:
            if r >= suggested:
                suggested = r
                break
        
        print(f"{layer_type:<15} {util:>10.0%} {suggested:>15}")
    
    print(f"\n--- BY LAYER DEPTH ---")
    print(f"{'Depth':<15} {'Avg Util':>10}")
    print("-" * 25)
    
    for depth in ['early', 'middle', 'late']:
        print(f"{depth:<15} {avg_by_depth[depth]:>10.0%}")
    
    print(f"\n--- PER-LAYER DETAILS ---")
    print(f"{'Layer':<40} {'Type':<15} {'Util':>8}")
    print("-" * 65)
    
    for m in sorted(final_metrics, key=lambda x: (x.layer_idx, x.layer_type)):
        print(f"{m.name[:40]:<40} {m.layer_type:<15} {m.utilization:>8.0%}")
    
    # Recommendations
    print(f"\n--- RECOMMENDATIONS ---")
    
    # Find types with low utilization
    low_util_types = [t for t, u in avg_by_type.items() if u < 0.3]
    high_util_types = [t for t, u in avg_by_type.items() if u > 0.7]
    
    if low_util_types:
        print(f"\n⚠️  Low utilization ({', '.join(low_util_types)}):")
        print(f"   Consider using lower rank for these layer types")
    
    if high_util_types:
        print(f"\n✓ High utilization ({', '.join(high_util_types)}):")
        print(f"   These layers are making good use of the configured rank")
    
    # Check depth pattern
    if avg_by_depth['late'] > avg_by_depth['early'] * 1.5:
        print(f"\n💡 Later layers use more rank than early layers")
        print(f"   Consider layer-wise rank: higher for late, lower for early")
    
    # Create result
    result = LayerUtilizationResult(
        task=task,
        rank=rank,
        final_accuracy=accuracy,
        layers=final_metrics,
        avg_utilization_by_type=avg_by_type,
        avg_utilization_by_depth=avg_by_depth,
    )
    
    # Save results
    output = {
        'config': {
            'model_name': model_name,
            'task': task,
            'rank': rank,
            'num_epochs': num_epochs,
        },
        'final_accuracy': accuracy,
        'layers': [asdict(m) for m in final_metrics],
        'avg_utilization_by_type': avg_by_type,
        'avg_utilization_by_depth': avg_by_depth,
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")
    
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='./lora_experiments/layer_utilization')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_experiment(output_dir, quick=args.quick)


if __name__ == "__main__":
    main()
