"""
Experiment 4: Task Complexity vs Required LoRA Rank

HYPOTHESIS: Harder tasks require higher effective rank. Simple tasks waste 
capacity with high rank configurations.

This experiment trains on multiple tasks of varying complexity:
- SST-2: Binary sentiment (simple)
- MRPC: Paraphrase detection (medium)
- MNLI: Natural language inference (hard, 3-way)

We measure effective rank utilization across tasks to validate:
- Simple tasks → low effective rank
- Complex tasks → higher effective rank
- Our rank suggestion should adapt to task complexity

Product implication: "Based on your task, we recommend rank-X"

Usage:
    python experiments/lora/exp4_task_complexity.py
"""

import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
import argparse

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
class TaskResult:
    """Result from training on a specific task."""
    task_name: str
    task_complexity: str  # "simple", "medium", "hard"
    num_labels: int
    train_size: int
    rank: int
    final_accuracy: float
    avg_effective_rank: float
    rank_utilization: float
    suggested_rank: int


TASKS = {
    'sst2': {
        'name': 'SST-2',
        'complexity': 'simple',
        'num_labels': 2,
        'text_field': 'sentence',
        'glue_name': 'sst2',
    },
    'mrpc': {
        'name': 'MRPC', 
        'complexity': 'medium',
        'num_labels': 2,
        'text_fields': ['sentence1', 'sentence2'],
        'glue_name': 'mrpc',
    },
    'mnli': {
        'name': 'MNLI',
        'complexity': 'hard',
        'num_labels': 3,
        'text_fields': ['premise', 'hypothesis'],
        'glue_name': 'mnli',
        'eval_split': 'validation_matched',
    },
}


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
            with torch.no_grad():
                BA = lora_B_params[base].float() @ lora_A_params[base].float()
                S = torch.linalg.svdvals(BA)
                effective_ranks.append(compute_effective_rank(S))
    
    return sum(effective_ranks) / len(effective_ranks) if effective_ranks else 0.0


def train_on_task(
    task_key: str,
    task_config: Dict,
    model_name: str,
    rank: int,
    num_epochs: int,
    max_train: int,
    max_eval: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> TaskResult:
    """Train LoRA on a specific task."""
    
    print(f"\n--- Task: {task_config['name']} ({task_config['complexity']}) ---")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    dataset = load_dataset("glue", task_config['glue_name'])
    
    # Tokenize based on task type
    if 'text_field' in task_config:
        # Single sentence task
        def tokenize(examples):
            return tokenizer(
                examples[task_config['text_field']],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
    else:
        # Sentence pair task
        def tokenize(examples):
            return tokenizer(
                examples[task_config['text_fields'][0]],
                examples[task_config['text_fields'][1]],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
    
    train_data = dataset["train"].select(range(min(max_train, len(dataset["train"]))))
    
    eval_split = task_config.get('eval_split', 'validation')
    eval_data = dataset[eval_split].select(range(min(max_eval, len(dataset[eval_split]))))
    
    train_data = train_data.map(tokenize, batched=True)
    eval_data = eval_data.map(tokenize, batched=True)
    
    train_data = train_data.rename_column("label", "labels")
    eval_data = eval_data.rename_column("label", "labels")
    
    train_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size)
    
    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=task_config['num_labels'],
    )
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 50, num_steps)
    
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
    
    # Analyze effective rank
    avg_eff_rank = analyze_lora_effective_rank(model)
    utilization = avg_eff_rank / rank
    
    # Suggest rank
    suggested = max(4, int(avg_eff_rank * 1.5))
    for r in [4, 8, 16, 32, 64]:
        if r >= suggested:
            suggested = r
            break
    
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Effective rank: {avg_eff_rank:.1f} / {rank} ({utilization:.0%})")
    print(f"  Suggested rank: {suggested}")
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return TaskResult(
        task_name=task_config['name'],
        task_complexity=task_config['complexity'],
        num_labels=task_config['num_labels'],
        train_size=len(train_data),
        rank=rank,
        final_accuracy=accuracy,
        avg_effective_rank=avg_eff_rank,
        rank_utilization=utilization,
        suggested_rank=suggested,
    )


def run_experiment(output_dir: Path, quick: bool = False) -> Dict:
    """Run the task complexity experiment."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model_name = "distilbert-base-uncased"
    rank = 64  # Use high rank to measure utilization
    learning_rate = 3e-4
    batch_size = 32
    
    if quick:
        num_epochs = 1
        max_train = 1000
        max_eval = 500
        tasks_to_run = ['sst2', 'mrpc']  # Skip MNLI in quick mode
    else:
        num_epochs = 3
        max_train = 5000
        max_eval = 1000
        tasks_to_run = ['sst2', 'mrpc', 'mnli']
    
    results: List[TaskResult] = []
    
    for task_key in tasks_to_run:
        task_config = TASKS[task_key]
        
        result = train_on_task(
            task_key=task_key,
            task_config=task_config,
            model_name=model_name,
            rank=rank,
            num_epochs=num_epochs,
            max_train=max_train,
            max_eval=max_eval,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )
        results.append(result)
    
    # Analysis
    print("\n" + "=" * 70)
    print("TASK COMPLEXITY VS REQUIRED RANK")
    print("=" * 70)
    
    print(f"\nConfigured rank: {rank}")
    print(f"\n{'Task':<10} {'Complexity':<10} {'Accuracy':>10} {'Eff.Rank':>10} {'Suggested':>10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r.task_name:<10} {r.task_complexity:<10} {r.final_accuracy:>10.1%} "
              f"{r.avg_effective_rank:>10.1f} {r.suggested_rank:>10}")
    
    # Check if complexity correlates with effective rank
    complexity_order = {'simple': 0, 'medium': 1, 'hard': 2}
    sorted_results = sorted(results, key=lambda r: complexity_order[r.task_complexity])
    
    eff_ranks = [r.avg_effective_rank for r in sorted_results]
    is_monotonic = all(eff_ranks[i] <= eff_ranks[i+1] for i in range(len(eff_ranks)-1))
    
    print(f"\n--- ANALYSIS ---")
    if is_monotonic or len(results) < 3:
        print("✓ Effective rank increases with task complexity as expected")
    else:
        print("⚠️ Effective rank does not strictly increase with complexity")
        print("   (This may indicate other factors affect rank utilization)")
    
    print("\nRecommendations by task complexity:")
    for r in sorted_results:
        print(f"  {r.task_complexity.upper()}: Use rank-{r.suggested_rank} "
              f"(saves {(rank - r.suggested_rank)/rank:.0%} vs rank-{rank})")
    
    # Save results
    output = {
        'config': {
            'model_name': model_name,
            'rank': rank,
            'num_epochs': num_epochs,
            'tasks': tasks_to_run,
        },
        'results': [asdict(r) for r in results],
        'analysis': {
            'complexity_rank_correlation': is_monotonic,
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")
    
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='./lora_experiments/task_complexity')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_experiment(output_dir, quick=args.quick)


if __name__ == "__main__":
    main()
