"""RoBERTa Replication: Do Findings Generalize Beyond DistilBERT?"""
import argparse, json
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

@dataclass
class Config:
    model_name: str = "roberta-base"
    task: str = "sst2"
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    learning_rate: float = 2e-4
    batch_size: int = 32
    num_epochs: int = 3
    max_train_samples: int = 5000
    max_eval_samples: int = 1000
    num_seeds: int = 3
    seed: int = 42

def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    import random; random.seed(seed)
    import numpy as np; np.random.seed(seed)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_condition(config, cond_name, base_rank, rank_pattern, seed, device):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    target_modules = ["query", "key", "value", "intermediate.dense"]
    lora_kwargs = {"task_type": TaskType.SEQ_CLS, "r": base_rank, "lora_alpha": config.lora_alpha, "lora_dropout": config.lora_dropout, "target_modules": target_modules}
    if rank_pattern: lora_kwargs["rank_pattern"] = rank_pattern
    try:
        lora_config = LoraConfig(**lora_kwargs)
    except TypeError:
        lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=base_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=target_modules)
    model = get_peft_model(model, lora_config); model.to(device)
    trainable_params = count_params(model)
    dataset = load_dataset("glue", config.task)
    def tokenize(ex): return tokenizer(ex["sentence"], truncation=True, padding=False, max_length=128)
    train_data = dataset["train"].select(range(min(config.max_train_samples, len(dataset["train"]))))
    eval_data = dataset["validation"].select(range(min(config.max_eval_samples, len(dataset["validation"]))))
    train_data = train_data.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    eval_data = eval_data.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    train_data.set_format("torch"); eval_data.set_format("torch")
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_data, batch_size=config.batch_size, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    for epoch in range(config.num_epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval(); correct = 0; total = 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch).logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])
    return {"condition": cond_name, "seed": seed, "accuracy": correct/total, "trainable_params": trainable_params}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./roberta_replication")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = Config(num_seeds=1 if args.quick else args.seeds, num_epochs=1 if args.quick else args.epochs, max_train_samples=1000 if args.quick else 5000, max_eval_samples=500 if args.quick else 1000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\nModel: {config.model_name}")
    output_path = Path(args.output_dir); output_path.mkdir(parents=True, exist_ok=True)
    conditions = {
        "baseline": {"base_rank": 64, "rank_pattern": None, "desc": "Uniform rank-64"},
        "moderate_r32": {"base_rank": 64, "rank_pattern": {"value": 32}, "desc": "V→32"},
        "aggressive_r16": {"base_rank": 64, "rank_pattern": {"value": 16}, "desc": "V→16"},
        "v16_ffn16": {"base_rank": 64, "rank_pattern": {"value": 16, "intermediate.dense": 16}, "desc": "V/FFN→16"},
    }
    print("="*70 + "\nROBERTA REPLICATION\n" + "="*70)
    print(f"Seeds: {config.num_seeds}, Epochs: {config.num_epochs}\n")
    all_results = {name: [] for name in conditions}
    for cond_name, cond in conditions.items():
        print(f"\n--- {cond_name} ({cond['desc']}) ---")
        for i in range(config.num_seeds):
            seed = config.seed + i
            result = run_condition(config, cond_name, cond["base_rank"], cond["rank_pattern"], seed, device)
            print(f"  Seed {seed}: {result['accuracy']:.1%}, {result['trainable_params']:,} params")
            all_results[cond_name].append(result)
    baseline_params = all_results["baseline"][0]["trainable_params"]
    baseline_acc = sum(r["accuracy"] for r in all_results["baseline"]) / len(all_results["baseline"])
    print("\n" + "="*70 + "\nRESULTS\n" + "="*70)
    print(f"\n{'Condition':<18} {'Accuracy':<16} {'Params':<12} {'Savings':<10} {'Δ Acc'}")
    print("-"*66)
    for cond_name, results in all_results.items():
        accs = [r["accuracy"] for r in results]
        mean_acc = sum(accs)/len(accs)
        std_acc = (sum((a-mean_acc)**2 for a in accs)/len(accs))**0.5
        params = results[0]["trainable_params"]
        savings = (baseline_params - params) / baseline_params
        delta = mean_acc - baseline_acc
        print(f"{cond_name:<18} {mean_acc:.1%} ± {std_acc:.1%}    {params:<12,} {savings:.1%}      {delta:+.1%}" if cond_name != "baseline" else f"{cond_name:<18} {mean_acc:.1%} ± {std_acc:.1%}    {params:<12,} -          -")
    print("\n" + "-"*66)
    print("DistilBERT reference: V/FFN→16 = 88.3% (+1.5%, 29% savings)")
    with open(output_path / "results.json", "w") as f:
        json.dump({"per_seed": all_results}, f, indent=2)
    print(f"\nSaved to {output_path}/results.json")

if __name__ == "__main__": main()
