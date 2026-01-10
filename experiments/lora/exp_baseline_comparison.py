"""Baseline Comparison: Do Spectral Metrics Add Value Over Simple Metrics?"""
import argparse, json, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

@dataclass
class Config:
    model_name: str = "distilbert-base-uncased"
    task: str = "sst2"
    batch_size: int = 32
    num_epochs: int = 3
    max_train_samples: int = 3000
    max_eval_samples: int = 500
    log_interval: int = 25
    seed: int = 42

def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    import random; random.seed(seed)
    import numpy as np; np.random.seed(seed)

def find_lora_layers(model):
    adapters = {}
    lora_a, lora_b = {}, {}
    for name, param in model.named_parameters():
        if 'lora_A' in name and 'weight' in name:
            lora_a[re.sub(r'\.lora_A\..*', '', name)] = param
        elif 'lora_B' in name and 'weight' in name:
            lora_b[re.sub(r'\.lora_B\..*', '', name)] = param
    for base in lora_a:
        if base in lora_b:
            adapters[base] = {'A': lora_a[base], 'B': lora_b[base]}
    return adapters

def compute_simple_metrics(model, loss_val):
    grad_norm = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
    adapters = find_lora_layers(model)
    norms = [((a['B'] @ a['A']).norm().item()) for a in adapters.values()]
    return {"loss": loss_val, "grad_norm": grad_norm, "adapter_norm": sum(norms)/len(norms) if norms else 0}

def compute_spectral_metrics(model):
    adapters = find_lora_layers(model)
    eff_ranks, kappas, utils = [], [], []
    for a in adapters.values():
        with torch.no_grad():
            A, B = a['A'].float(), a['B'].float()
            BA = B @ A
            S = torch.linalg.svdvals(BA); S = S[S > 1e-10]
            if len(S) > 0:
                p = S / S.sum()
                eff_rank = math.exp(-(p * torch.log(p + 1e-10)).sum().item())
                eff_ranks.append(eff_rank)
                utils.append(eff_rank / min(A.shape[0], B.shape[1]))
            S_B = torch.linalg.svdvals(B); S_B = S_B[S_B > 1e-10]
            if len(S_B) >= 2: kappas.append((S_B[0] / S_B[-1]).item())
    return {"effective_rank": sum(eff_ranks)/len(eff_ranks) if eff_ranks else 0,
            "utilization": sum(utils)/len(utils) if utils else 0,
            "kappa_B": sum(kappas)/len(kappas) if kappas else 1,
            "kappa_B_max": max(kappas) if kappas else 1}

def run_training(config, lora_cfg, cond_name, device):
    set_seed(config.seed)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    lr = lora_cfg.pop("lr", 3e-4)
    model = get_peft_model(model, LoraConfig(**lora_cfg)); model.to(device)
    dataset = load_dataset("glue", config.task)
    def tok(ex): return tokenizer(ex["sentence"], truncation=True, padding=False, max_length=128)
    train_data = dataset["train"].select(range(min(config.max_train_samples, len(dataset["train"]))))
    eval_data = dataset["validation"].select(range(min(config.max_eval_samples, len(dataset["validation"]))))
    train_data = train_data.map(tok, batched=True, remove_columns=["sentence", "idx"]); train_data.set_format("torch")
    eval_data = eval_data.map(tok, batched=True, remove_columns=["sentence", "idx"]); eval_data.set_format("torch")
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))
    eval_loader = DataLoader(eval_data, batch_size=config.batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    trajectory, step = [], 0
    for epoch in range(config.num_epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            optimizer.zero_grad(); loss.backward()
            if step % config.log_interval == 0:
                trajectory.append({"step": step, **compute_simple_metrics(model, loss.item()), **compute_spectral_metrics(model)})
            optimizer.step(); step += 1
    model.eval(); correct, total = 0, 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            correct += (model(**batch).logits.argmax(-1) == batch["labels"]).sum().item()
            total += len(batch["labels"])
    return {"condition": cond_name, "final_accuracy": correct/total, "final_spectral": compute_spectral_metrics(model), "trajectory": trajectory}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./baseline_comparison")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = Config(num_epochs=1 if args.quick else 3, max_train_samples=1000 if args.quick else 3000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    output_path = Path(args.output_dir); output_path.mkdir(parents=True, exist_ok=True)
    target_modules = ["q_lin", "k_lin", "v_lin", "lin1", "lin2"]
    conditions = {
        "healthy": {"task_type": TaskType.SEQ_CLS, "r": 16, "lora_alpha": 16, "lora_dropout": 0.1, "target_modules": target_modules, "lr": 3e-4},
        "high_alpha": {"task_type": TaskType.SEQ_CLS, "r": 16, "lora_alpha": 64, "lora_dropout": 0.1, "target_modules": target_modules, "lr": 3e-4},
        "high_lr": {"task_type": TaskType.SEQ_CLS, "r": 16, "lora_alpha": 16, "lora_dropout": 0.1, "target_modules": target_modules, "lr": 1e-3},
        "overparameterized": {"task_type": TaskType.SEQ_CLS, "r": 128, "lora_alpha": 128, "lora_dropout": 0.1, "target_modules": target_modules, "lr": 3e-4},
        "underparameterized": {"task_type": TaskType.SEQ_CLS, "r": 4, "lora_alpha": 4, "lora_dropout": 0.1, "target_modules": target_modules, "lr": 3e-4},
    }
    print("="*70 + "\nBASELINE COMPARISON: Simple vs Spectral Metrics\n" + "="*70)
    all_results = {}
    for cond_name, cond in conditions.items():
        print(f"\n--- {cond_name} ---")
        result = run_training(config, dict(cond), cond_name, device)
        traj = result["trajectory"]
        first, last = traj[0], traj[-1]
        result["analysis"] = {
            "loss_ok": last["loss"] < first["loss"],
            "grad_ok": last["grad_norm"] < 10,
            "rank_collapsed": last["utilization"] < 0.3,
            "kappa_problem": last["kappa_B_max"] > 500,
            "final_util": last["utilization"],
            "final_kappa": last["kappa_B_max"],
        }
        all_results[cond_name] = result
        a = result["analysis"]
        print(f"  Acc: {result['final_accuracy']:.1%}, Util: {a['final_util']:.0%}, κ_max: {a['final_kappa']:.0f}")
        if a["loss_ok"] and a["grad_ok"] and (a["rank_collapsed"] or a["kappa_problem"]):
            print(f"  ⚠️  SPECTRAL CAUGHT ISSUE SIMPLE METRICS MISSED!")
    print("\n" + "="*70 + "\nSUMMARY\n" + "="*70)
    print(f"\n{'Condition':<20} {'Acc':<8} {'Loss↓':<8} {'Grad OK':<8} {'Util':<8} {'κ_max':<10} {'Issue?'}")
    print("-"*70)
    for name, r in all_results.items():
        a = r["analysis"]
        issue = ""
        if a["rank_collapsed"]: issue += "RankCollapse "
        if a["kappa_problem"]: issue += "HighKappa "
        print(f"{name:<20} {r['final_accuracy']:.1%}    {'✓' if a['loss_ok'] else '✗':<8} {'✓' if a['grad_ok'] else '✗':<8} {a['final_util']:.0%}     {a['final_kappa']:<10.0f} {issue or '-'}")
    with open(output_path / "results.json", "w") as f:
        json.dump({n: {"accuracy": r["final_accuracy"], "analysis": r["analysis"]} for n, r in all_results.items()}, f, indent=2)
    print(f"\nSaved to {output_path}/results.json")

if __name__ == "__main__": main()
