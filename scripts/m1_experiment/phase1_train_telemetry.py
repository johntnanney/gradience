#!/usr/bin/env python3
"""
Phase 1 (Telemetry-Enhanced): Train adapters with GradienceCallback + eval.

Identical to phase1_train.py but adds:
  1. GradienceCallback — logs run.jsonl with loss, lr, grad_norm per step
  2. Eval split — holds out 10% of training data for validation
  3. eval_strategy="steps", eval_steps=25 — yields ~48 eval events per 1200-step run
  4. logging_steps=10 — finer-grained grad_norm telemetry between evals
  5. StructuralMetricsCallback — periodic SVD on LoRA A/B pairs to log
     stable_rank, effective_rank, energy_rank_90, adapter_frobenius_norm

The resulting run.jsonl files can be fed directly into
`gradience.analysis.extract_timeseries` for lead-lag analysis.

Usage:
    python scripts/m1_experiment/phase1_train_telemetry.py \
        --config scripts/m1_experiment/m1_config.yaml

    # Smoke test (5 steps, 1 seed):
    python scripts/m1_experiment/phase1_train_telemetry.py \
        --config scripts/m1_experiment/m1_config.yaml --smoke

    # Single task:
    python scripts/m1_experiment/phase1_train_telemetry.py \
        --config scripts/m1_experiment/m1_config.yaml --task math
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _iter_lora_pairs(model) -> List[Tuple[str, Any, Any]]:
    """Yield (layer_name, A_tensor, B_tensor) for each LoRA pair in a PEFT model.

    PEFT stores weights as:
      base_model.model.model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_A.default.weight
      base_model.model.model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_B.default.weight
    """
    pairs = {}
    for name, param in model.named_parameters():
        if "lora_A" in name:
            prefix = name.split(".lora_A")[0]
            pairs.setdefault(prefix, {})["A"] = param
        elif "lora_B" in name:
            prefix = name.split(".lora_B")[0]
            pairs.setdefault(prefix, {})["B"] = param

    result = []
    for prefix, ab in sorted(pairs.items()):
        if "A" in ab and "B" in ab:
            result.append((prefix, ab["A"], ab["B"]))
    return result


def _compute_structural_metrics(model) -> Dict[str, float]:
    """Compute aggregated structural metrics across all LoRA pairs.

    Returns dict with keys like stable_rank_mean, effective_rank_mean, etc.
    Uses the same SVD math as gradience.vnext.audit.lora_audit but
    inlined here to avoid import-chain issues on RunPod.

    Performance: The expensive matmuls (A @ A.T, B.T @ B) are done on-device
    in float32 since the operands are (r, d_in) and (d_out, r) with d_in/d_out
    up to 4096. Only the small (r, r) results are moved to CPU for the
    eigendecomposition in float64. This brings 128-pair Mistral-7B from ~180s
    down to ~1-2s.
    """
    import torch

    eps = 1e-12
    pairs = _iter_lora_pairs(model)
    if not pairs:
        return {}

    stable_ranks = []
    effective_ranks = []
    r90s = []
    frob_norms = []
    sigma_maxes = []

    for prefix, A, B in pairs:
        # A: (r, d_in), B: (d_out, r)
        # Do the big matmuls on-device in float32 (bf16 loses too much precision
        # for eigendecomposition, but float32 is fine for the r×r intermediates)
        A_ = A.detach().float()       # (r, d_in) on GPU, float32
        B_ = B.detach().float()       # (d_out, r) on GPU, float32

        AAT = A_ @ A_.T              # (r, r) on GPU — fast
        BTB = B_.T @ B_              # (r, r) on GPU — fast

        # Move only the small (r, r) matrices to CPU for float64 eigen
        AAT_cpu = AAT.to(dtype=torch.float64, device="cpu")
        BTB_cpu = BTB.to(dtype=torch.float64, device="cpu")

        # Frobenius norm squared of BA = tr(AAT @ BTB)
        frob_sq = torch.sum(AAT_cpu * BTB_cpu).item()
        frob_norms.append(frob_sq ** 0.5)

        # Singular values via eigendecomposition (avoids forming full BA)
        eigA, UA = torch.linalg.eigh(AAT_cpu)
        eigA = torch.clamp(eigA, min=0.0)
        sqrt_eigA = torch.sqrt(eigA)
        C = UA.T @ BTB_cpu @ UA
        M = (sqrt_eigA.unsqueeze(1) * C) * sqrt_eigA.unsqueeze(0)
        M = 0.5 * (M + M.T)  # symmetrize
        eigM = torch.linalg.eigvalsh(M)
        eigM = torch.clamp(eigM, min=0.0)
        s = torch.sqrt(eigM)
        s = torch.sort(s, descending=True).values

        sigma_max = s[0].item() if s.numel() else 0.0
        sigma_maxes.append(sigma_max)

        # Stable rank = ||BA||_F^2 / sigma_max^2
        sigma_max_sq = max(sigma_max * sigma_max, eps)
        stable_ranks.append(frob_sq / sigma_max_sq)

        # Effective rank (Shannon entropy of normalized singular values)
        s_pos = s[s > eps]
        if s_pos.numel() > 0:
            p = s_pos / (s_pos.sum() + eps)
            entropy = -(p * torch.log(p + eps)).sum()
            effective_ranks.append(torch.exp(entropy).item())
        else:
            effective_ranks.append(0.0)

        # Energy rank 90: min k s.t. sum_{i<=k} s_i^2 / total >= 0.90
        if s_pos.numel() > 0:
            e = s_pos.pow(2)
            total = e.sum()
            if total > eps:
                c = torch.cumsum(e, dim=0) / total
                idx = torch.searchsorted(
                    c, torch.tensor([0.90], device=c.device, dtype=c.dtype), right=False
                )
                r90s.append(min(int(idx.item()) + 1, int(s_pos.numel())))
            else:
                r90s.append(0)
        else:
            r90s.append(0)

    return {
        "stable_rank_mean": statistics.mean(stable_ranks),
        "stable_rank_median": statistics.median(stable_ranks),
        "effective_rank_mean": statistics.mean(effective_ranks),
        "effective_rank_median": statistics.median(effective_ranks),
        "energy_rank_90_median": statistics.median(r90s),
        "energy_rank_90_mean": statistics.mean(r90s),
        "adapter_frob_norm_mean": statistics.mean(frob_norms),
        "adapter_frob_norm_total": sum(frob_norms),
        "sigma_max_mean": statistics.mean(sigma_maxes),
        "sigma_max_max": max(sigma_maxes),
        "n_layers": len(pairs),
    }


def _make_structural_callback(gradience_callback, structural_every_n: int = 50):
    """Factory that creates a StructuralMetricsCallback inheriting from TrainerCallback.

    We defer the import so the module can be loaded without transformers installed.
    """
    from transformers import TrainerCallback

    class StructuralMetricsCallback(TrainerCallback):
        """HF TrainerCallback that periodically computes LoRA structural metrics.

        Writes a 'metrics' event (kind='structural') to the GradienceCallback's
        TelemetryWriter every `structural_every_n` training steps.
        """

        def __init__(self):
            super().__init__()
            self.gradience_callback = gradience_callback
            self.structural_every_n = structural_every_n

        def on_log(self, args, state, control, logs=None, **kwargs):
            """Piggyback on HF's on_log hook (fires every logging_steps)."""
            step = int(getattr(state, "global_step", 0))
            if step == 0 or step % self.structural_every_n != 0:
                return

            model = kwargs.get("model", None)
            if model is None:
                return

            writer = getattr(self.gradience_callback, "writer", None)
            if writer is None:
                return

            t0 = time.monotonic()
            metrics = _compute_structural_metrics(model)
            elapsed_ms = (time.monotonic() - t0) * 1000
            metrics["compute_time_ms"] = round(elapsed_ms, 1)

            writer.metrics(step, kind="structural", metrics=metrics)
            print(f"    [structural@{step}] stable_rank={metrics['stable_rank_mean']:.2f} "
                  f"eff_rank={metrics['effective_rank_mean']:.2f} "
                  f"r90={metrics['energy_rank_90_median']} "
                  f"({elapsed_ms:.0f}ms)")

    return StructuralMetricsCallback()


def load_config(config_path: str, smoke: bool = False) -> dict:
    """Load and optionally apply smoke test overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if smoke:
        smoke_cfg = config.get("smoke", {})
        config["training"]["max_steps"] = smoke_cfg.get("max_steps", 5)
        for task in config["adapters"].values():
            task["max_train_samples"] = smoke_cfg.get("max_train_samples", 50)
        config["experiment"]["seeds"] = smoke_cfg.get("seeds", [42])

    return config


def train_single_adapter(
    base_model: str,
    task_name: str,
    task_config: dict,
    training_config: dict,
    seed: int,
    output_dir: Path,
    device: str = "cuda",
    eval_steps: int = 25,
    logging_steps: int = 10,
    eval_fraction: float = 0.10,
    structural_every_n: int = 50,
) -> Path:
    """Train one LoRA adapter with GradienceCallback + eval + structural metrics."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    # Gradience telemetry callback
    from gradience.vnext.integrations.hf import (
        GradienceCallback,
        GradienceCallbackConfig,
    )

    # Import task formatter (sibling module)
    sys.path.insert(0, str(Path(__file__).parent))
    from task_configs import get_formatter

    adapter_dir = output_dir / task_name / f"seed_{seed}"

    # Skip if already trained
    if (adapter_dir / "adapter_config.json").exists():
        print(f"  [SKIP] {task_name}/seed_{seed} -- already exists")
        return adapter_dir

    print(f"\n  Training {task_name}/seed_{seed}...")
    start = time.monotonic()

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(training_config.get("torch_dtype", "bfloat16"), torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=training_config["rank"],
        lora_alpha=training_config["alpha"],
        lora_dropout=0.0,
        target_modules=training_config["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Load dataset
    ds_name = task_config["dataset"]
    ds_subset = task_config.get("subset", None)
    if ds_subset:
        ds = load_dataset(ds_name, ds_subset, split="train")
    else:
        ds = load_dataset(ds_name, split="train")

    # Subsample
    max_samples = task_config.get("max_train_samples", 10000)
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    print(f"    Dataset: {ds_name}, {len(ds)} examples (before split)")

    # Format + tokenize
    formatter = get_formatter(task_name)

    def tokenize_fn(example):
        text = formatter(example)
        # Don't set labels here — DataCollatorForLanguageModeling(mlm=False)
        # copies input_ids to labels and pads with -100 automatically.
        enc = tokenizer(text, truncation=True, max_length=512, padding=False)
        return enc

    tokenized = ds.map(tokenize_fn, remove_columns=ds.column_names)

    # ---- NEW: Train/eval split ----
    n_eval = max(1, int(len(tokenized) * eval_fraction))
    n_train = len(tokenized) - n_eval
    split = tokenized.train_test_split(
        test_size=n_eval,
        seed=seed,
        shuffle=True,
    )
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"    Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ---- Gradience telemetry setup ----
    telemetry_dir = adapter_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    gradience_config = GradienceCallbackConfig(
        output_dir=str(telemetry_dir),
        filename="run.jsonl",
        dataset_name=ds_name,
        notes=f"m1_lead_lag | task={task_name} seed={seed}",
    )
    gradience_callback = GradienceCallback(config=gradience_config)

    # Structural metrics callback (SVD on LoRA pairs every N steps)
    structural_callback = _make_structural_callback(
        gradience_callback=gradience_callback,
        structural_every_n=structural_every_n,
    )

    # Training arguments (enhanced for lead-lag)
    train_dir = adapter_dir / "training_logs"
    training_args = TrainingArguments(
        output_dir=str(train_dir),
        per_device_train_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation"],
        learning_rate=training_config["learning_rate"],
        max_steps=training_config["max_steps"],
        # ---- Telemetry-specific settings ----
        logging_steps=logging_steps,       # grad_norm every 10 steps (was 50)
        eval_strategy="steps",             # periodic eval
        eval_steps=eval_steps,             # eval every 25 steps (~48 evals per 1200 steps)
        # ---- Unchanged ----
        save_strategy="no",
        bf16=(training_config.get("torch_dtype") == "bfloat16"),
        fp16=(training_config.get("torch_dtype") == "float16"),
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
        # Disable find_unused_parameters warning with gradient checkpointing
        ddp_find_unused_parameters=False,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[gradience_callback, structural_callback],
    )
    trainer.train()

    # Save adapter
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_dir))

    elapsed = time.monotonic() - start
    print(f"    Saved to: {adapter_dir} ({elapsed / 60:.1f} min)")
    print(f"    Telemetry: {telemetry_dir / 'run.jsonl'}")

    # Cleanup GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return adapter_dir


def main():
    parser = argparse.ArgumentParser(description="M1 Phase 1 (Telemetry-Enhanced): Train adapters")
    parser.add_argument("--config", required=True, help="Path to m1_config.yaml")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (5 steps, 1 seed)")
    parser.add_argument("--task", type=str, default=None,
                        help="Train a single task (e.g. 'math'). Default: all tasks")
    parser.add_argument("--eval-steps", type=int, default=25,
                        help="Eval every N steps (default: 25)")
    parser.add_argument("--logging-steps", type=int, default=10,
                        help="Log grad_norm every N steps (default: 10)")
    parser.add_argument("--eval-fraction", type=float, default=0.10,
                        help="Fraction of data to hold out for eval (default: 0.10)")
    parser.add_argument("--structural-every-n", type=int, default=50,
                        help="Compute structural SVD metrics every N steps (default: 50)")
    args = parser.parse_args()

    config = load_config(args.config, smoke=args.smoke)
    workspace = Path(config["runtime"]["workspace"])
    adapters_dir = workspace / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    base_model = config["experiment"]["base_model"]
    seeds = config["experiment"]["seeds"]
    training_config = config["training"]

    # Filter tasks if --task specified
    adapter_configs = config["adapters"]
    if args.task:
        if args.task not in adapter_configs:
            print(f"Error: unknown task '{args.task}'. Available: {sorted(adapter_configs.keys())}")
            sys.exit(1)
        adapter_configs = {args.task: adapter_configs[args.task]}

    total_start = time.monotonic()
    n_total = len(adapter_configs) * len(seeds)
    n_done = 0

    max_steps = training_config["max_steps"]
    expected_evals = max_steps // args.eval_steps

    print(f"Phase 1 (Telemetry-Enhanced): Training {n_total} adapters")
    print(f"  Base model: {base_model}")
    print(f"  Seeds: {seeds}")
    print(f"  Tasks: {list(adapter_configs.keys())}")
    print(f"  Max steps: {max_steps}")
    print(f"  Eval every: {args.eval_steps} steps (~{expected_evals} eval events)")
    print(f"  Log every: {args.logging_steps} steps")
    print(f"  Eval fraction: {args.eval_fraction:.0%}")
    print(f"  Structural SVD every: {args.structural_every_n} steps")

    for task_name, task_config in adapter_configs.items():
        for seed in seeds:
            n_done += 1
            print(f"\n[{n_done}/{n_total}] {task_name}/seed_{seed}")
            train_single_adapter(
                base_model=base_model,
                task_name=task_name,
                task_config=task_config,
                training_config=training_config,
                seed=seed,
                output_dir=adapters_dir,
                device=config["runtime"]["device"],
                eval_steps=args.eval_steps,
                logging_steps=args.logging_steps,
                eval_fraction=args.eval_fraction,
                structural_every_n=args.structural_every_n,
            )

    elapsed = time.monotonic() - total_start
    print(f"\nPhase 1 complete: {n_total} adapters in {elapsed / 3600:.1f} hours")
    print(f"\nTelemetry files:")
    for task_name in adapter_configs:
        for seed in seeds:
            tpath = adapters_dir / task_name / f"seed_{seed}" / "telemetry" / "run.jsonl"
            print(f"  {tpath}")


if __name__ == "__main__":
    main()
