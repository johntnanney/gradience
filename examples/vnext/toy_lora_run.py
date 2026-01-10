#!/usr/bin/env python3
"""
Toy LoRA run (vNext telemetry producer)

Purpose
-------
This script validates the canonical Gradience vNext loop on minimal hardware:

  - small model + small dataset slice
  - short training budget (steps)
  - writes vNext telemetry JSONL (run_start/train_step/eval/metrics/run_end)
  - saves a PEFT adapter directory (adapter_config + adapter_model) for `gradience audit`

It is NOT intended to produce meaningful benchmark results.

Typical usage
-------------
python examples/vnext/toy_lora_run.py --out runs/toy_run

Then:
  gradience check --task sst2 --peft-dir runs/toy_run/peft --training-dir runs/toy_run/training
  gradience monitor runs/toy_run/run.jsonl
  gradience audit --peft-dir runs/toy_run/peft --top-wasteful 10

Defaults are CPU-friendly and trivial on any GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _infer_lora_targets(model: Any) -> List[str]:
    """Best-effort target_modules inference across common architectures."""
    # PEFT matches by *module name suffix*, so we look at leaf module names.
    leaf_names = set()
    try:
        for name, _mod in model.named_modules():
            if not name:
                continue
            leaf_names.add(name.split(".")[-1])
    except Exception:
        return []

    candidate_sets = [
        # LLaMA/Mistral-style
        ["q_proj", "k_proj", "v_proj", "o_proj"],
        # DistilBERT-style
        ["q_lin", "k_lin", "v_lin", "out_lin"],
        # BERT-style
        ["query", "key", "value", "dense"],
        # GPT-2-style
        ["c_attn", "c_proj"],
    ]

    for cand in candidate_sets:
        present = [c for c in cand if c in leaf_names]
        # prefer full matches, else accept partial but non-trivial
        if len(present) == len(cand):
            return cand
        if len(present) >= 2:
            return present

    return []


def _select_small_split(ds: Any, n: int, seed: int) -> Any:
    if n is None or n <= 0:
        return ds
    try:
        ds = ds.shuffle(seed=seed)
    except Exception:
        pass
    try:
        return ds.select(range(min(n, len(ds))))
    except Exception:
        return ds


def _device_from_arg(device_str: str) -> Tuple[str, bool]:
    """Return (device, use_cuda) where device is a torch device string."""
    device_str = (device_str or "auto").lower().strip()
    try:
        import torch
    except Exception:
        return "cpu", False

    if device_str == "cpu":
        return "cpu", False
    if device_str in ("cuda", "gpu"):
        return "cuda", True
    if device_str == "auto":
        return ("cuda", True) if torch.cuda.is_available() else ("cpu", False)
    if device_str.startswith("cuda"):
        return device_str, True
    return "cpu", False


def _dtype_from_arg(dtype_str: str, *, device: str) -> str:
    dtype_str = (dtype_str or "fp32").lower().strip()
    if device.startswith("cpu"):
        return "fp32"
    if dtype_str in ("bf16", "bfloat16"):
        return "bf16"
    if dtype_str in ("fp16", "float16"):
        return "fp16"
    return "fp32"


def _torch_dtype(dtype_key: str) -> Any:
    import torch
    if dtype_key == "bf16":
        return torch.bfloat16
    if dtype_key == "fp16":
        return torch.float16
    return torch.float32


def _evaluate(model: Any, dataloader: Any, device: str) -> Dict[str, Any]:
    import torch

    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            logits = out.logits
            labels = batch.get("labels")
            bs = int(labels.shape[0]) if labels is not None else int(logits.shape[0])

            total_loss += float(loss.item()) * bs
            total += bs

            if labels is not None:
                preds = torch.argmax(logits, dim=-1)
                correct += int((preds == labels).sum().item())

    if total == 0:
        return {"loss": None, "ppl": None, "accuracy": None, "n": 0}

    avg_loss = total_loss / total
    ppl = float(math.exp(avg_loss)) if avg_loss < 50 else float("inf")
    acc = float(correct) / float(total) if total > 0 else None

    return {"loss": avg_loss, "ppl": ppl, "accuracy": acc, "n": total}


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy LoRA run that emits Gradience vNext telemetry.")
    parser.add_argument("--out", type=str, default="runs/toy_lora", help="Output directory (telemetry + peft + training args).")

    parser.add_argument("--model", type=str, default="hf-internal-testing/tiny-random-distilbert",
                        help="HF model name (tiny by default for low compute).")
    parser.add_argument("--dataset", type=str, default="glue/sst2",
                        help="Dataset identifier. Default: glue/sst2.")
    parser.add_argument("--max-length", type=int, default=64, help="Max sequence length (lower = faster).")

    parser.add_argument("--train-samples", type=int, default=256, help="Train subset size (lower = faster).")
    parser.add_argument("--eval-samples", type=int, default=256, help="Eval subset size (lower = faster).")

    parser.add_argument("--max-steps", type=int, default=100, help="Training steps.")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--log-every", type=int, default=10, help="Telemetry train_step logging frequency.")

    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (restraint-first default).")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")

    parser.add_argument("--r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--alpha", type=float, default=8.0, help="LoRA alpha.")
    parser.add_argument("--targets", type=str, default="",
                        help="Comma-separated LoRA target module suffixes. If empty, inferred from model.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")

    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:0 ...")
    parser.add_argument("--dtype", type=str, default="fp32", help="fp32|bf16|fp16 (GPU only).")

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--no-audit", action="store_true", help="Skip LoRA audit metrics emission (faster).")

    parser.add_argument("--telemetry-allow-text", action="store_true", help="Allow logging long strings (e.g., prompts/examples) into telemetry JSONL. Default redacts >256 chars.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    telemetry_path = out_dir / "run.jsonl"
    peft_dir = out_dir / "peft"
    training_dir = out_dir / "training"
    peft_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)

    device, _use_cuda = _device_from_arg(args.device)
    dtype_key = _dtype_from_arg(args.dtype, device=device)

    # --- imports (delayed so --help works even if deps missing) ---
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as e:
        raise SystemExit(f"Missing dependency: torch. Install with: pip install torch\nError: {e}")

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing dependency: datasets. Install with: pip install datasets\nError: {e}")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing dependency: transformers. Install with: pip install transformers\nError: {e}")

    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing dependency: peft. Install with: pip install peft\nError: {e}")

    # Gradience vNext
    try:
        from gradience.vnext import (
            ConfigSnapshot,
            LoRAConfigSnapshot,
            OptimizerConfigSnapshot,
            TrainingConfigSnapshot,
            TaskProfile,
            Severity,
        )
        from gradience.vnext.telemetry import TelemetryWriter
    except Exception as e:
        raise SystemExit(
            "Could not import gradience.vnext. Make sure you're running from the repo root "
            "or installed Gradience (pip install -e .).\n"
            f"Error: {e}"
        )

    # Dataset loader
    ds_name = args.dataset
    if ds_name.lower() in ("glue/sst2", "sst2", "glue:sst2"):
        ds = load_dataset("glue", "sst2")
        dataset_name_for_config = "glue/sst2"
        train_split = ds["train"]
        test_split = ds["validation"]  # treat validation as test for toy run
        text_key = "sentence"
        label_key = "label"
    else:
        parts = ds_name.split("/", 1)
        if len(parts) == 2:
            ds = load_dataset(parts[0], parts[1])
            dataset_name_for_config = ds_name
        else:
            ds = load_dataset(ds_name)
            dataset_name_for_config = ds_name

        train_split = ds["train"] if "train" in ds else list(ds.values())[0]
        test_split = ds["validation"] if "validation" in ds else (ds["test"] if "test" in ds else train_split)

        cols = set(train_split.column_names)
        text_key = "sentence" if "sentence" in cols else ("text" if "text" in cols else train_split.column_names[0])
        label_key = "label" if "label" in cols else ("labels" if "labels" in cols else None)

    train_split = _select_small_split(train_split, args.train_samples, args.seed)
    test_split = _select_small_split(test_split, args.eval_samples, args.seed + 1)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def _tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        texts = batch[text_key]
        return tokenizer(texts, truncation=True, max_length=int(args.max_length))

    train_tok = train_split.map(_tokenize, batched=True)
    test_tok = test_split.map(_tokenize, batched=True)

    if label_key is not None and label_key in train_tok.column_names and label_key != "labels":
        train_tok = train_tok.rename_column(label_key, "labels")
    if label_key is not None and label_key in test_tok.column_names and label_key != "labels":
        test_tok = test_tok.rename_column(label_key, "labels")

    keep_cols = ["input_ids", "attention_mask"]
    if "labels" in train_tok.column_names:
        keep_cols.append("labels")

    train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in keep_cols])
    test_tok = test_tok.remove_columns([c for c in test_tok.column_names if c not in keep_cols])

    train_tok.set_format(type="torch")
    test_tok.set_format(type="torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(train_tok, batch_size=int(args.batch_size), shuffle=True, collate_fn=collator)
    test_loader = DataLoader(test_tok, batch_size=int(args.batch_size), shuffle=False, collate_fn=collator)

    # Model
    load_kwargs: Dict[str, Any] = {}
    if device.startswith("cuda") and dtype_key in ("bf16", "fp16"):
        load_kwargs["torch_dtype"] = _torch_dtype(dtype_key)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, **load_kwargs)
    except Exception as e:
        # fallback for the default tiny model
        if args.model == "hf-internal-testing/tiny-random-distilbert":
            fallback = "distilbert-base-uncased"
            print(f"[toy_lora_run] Warning: failed to load {args.model!r}. Falling back to {fallback!r}. Error: {e}")
            args.model = fallback
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, **load_kwargs)
        else:
            raise

    model.to(device)

    # Target modules
    if args.targets.strip():
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    else:
        targets = _infer_lora_targets(model)

    if not targets:
        raise SystemExit(
            "Could not infer LoRA target_modules for this model. "
            "Pass --targets 'q_proj,k_proj,v_proj,o_proj' (or similar) explicitly."
        )

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=int(args.r),
        lora_alpha=float(args.alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=targets,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Optimizer (train only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    # vNext config snapshot
    cfg = ConfigSnapshot(
        model_name=args.model,
        dataset_name=dataset_name_for_config,
        task_profile=TaskProfile.EASY_CLASSIFICATION,
        optimizer=OptimizerConfigSnapshot(name="adamw", lr=float(args.lr), weight_decay=float(args.weight_decay)),
        lora=LoRAConfigSnapshot(
            r=int(args.r),
            alpha=float(args.alpha),
            target_modules=list(targets),
            dropout=float(args.lora_dropout),
        ),
        training=TrainingConfigSnapshot(
            seed=int(args.seed),
            batch_size=int(args.batch_size),
            gradient_accumulation=int(args.grad_accum),
            max_steps=int(args.max_steps),
            epochs=None,
            dtype=dtype_key,
            extras={"max_length": int(args.max_length)},
        ),
        notes="toy_lora_run (vNext telemetry producer)",
        extras={"toy": True},
    )

    meta = {
        "script": "examples/vnext/toy_lora_run.py",
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        "pid": os.getpid(),
        "device": device,
        "dtype": dtype_key,
        "args": vars(args),
    }

    # Save "training_args.json" (for gradience check wrapper)
    training_args_path = training_dir / "training_args.json"
    training_args_payload = {
        "base_model_name_or_path": args.model,
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "per_device_train_batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.grad_accum),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "torch_dtype": dtype_key,
        "extras": {
            "dataset": dataset_name_for_config,
            "max_length": int(args.max_length),
            "toy": True,
        },
    }
    training_args_path.write_text(json.dumps(training_args_payload, indent=2), encoding="utf-8")

    # Train loop (short)
    global_step = 0
    model.train()

    train_iter = iter(train_loader)

    with TelemetryWriter(telemetry_path, allow_text=args.telemetry_allow_text) as tw:
        tw.run_start(cfg, meta=meta)

        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()

        while global_step < int(args.max_steps):
            global_step += 1
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss

            loss_scaled = loss / float(args.grad_accum)
            loss_scaled.backward()

            if global_step % int(args.grad_accum) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % int(args.log_every) == 0 or global_step == 1:
                lr_val = optimizer.param_groups[0]["lr"] if optimizer.param_groups else None
                tw.train_step(global_step, loss=float(loss.item()), lr=lr_val, wall_time_s=time.time() - t0)

        # Eval on train/test
        train_metrics = _evaluate(model, train_loader, device)
        test_metrics = _evaluate(model, test_loader, device)

        tw.eval(global_step, split="train", metrics=train_metrics)
        tw.eval(global_step, split="test", metrics=test_metrics)

        # Save adapter weights (PEFT dir)
        # Prefer .bin for portability (no safetensors dependency required).
        model.save_pretrained(peft_dir, safe_serialization=True)
        # Log what adapter file was saved (so logs tell the truth without ls)
        saved_sft = Path(peft_dir) / 'adapter_model.safetensors'
        saved_bin = Path(peft_dir) / 'adapter_model.bin'
        if saved_sft.exists():
            print(f"[toy_lora_run] Saved {saved_sft}")
        elif saved_bin.exists():
            print(f"[toy_lora_run] Saved {saved_bin}")
        else:
            print(f"[toy_lora_run] Saved adapter to {peft_dir}")

        # Emit LoRA audit metrics into telemetry (optional)
        if not args.no_audit:
            try:
                from gradience.vnext.audit import audit_lora_peft_dir
                audit = audit_lora_peft_dir(peft_dir)
                tw.metrics(global_step, kind="lora_audit", metrics=audit.to_summary_dict(include_layers=False))
            except Exception as e:
                tw.alert(
                    severity=Severity.WARNING,
                    code="lora_audit_failed",
                    message=f"LoRA audit failed: {e}",
                    step=global_step,
                    context={"peft_dir": str(peft_dir)},
                )

        tw.run_end(status="ok")

    # Print next steps
    print("\n" + "=" * 72)
    print("TOY RUN COMPLETE")
    print("=" * 72)
    print(f"Telemetry: {telemetry_path}")
    print(f"PEFT dir:  {peft_dir}")
    print(f"Training:  {training_args_path}")
    print("\nNext commands:")
    print(f"  gradience check --task sst2 --peft-dir {peft_dir} --training-dir {training_dir}")
    print(f"  gradience monitor {telemetry_path}")
    print(f"  gradience audit --peft-dir {peft_dir} --top-wasteful 10")
    print("=" * 72)


if __name__ == "__main__":
    main()
