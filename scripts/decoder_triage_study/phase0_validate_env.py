#!/usr/bin/env python3
"""Phase 0 smoke validation for controlled decoder merge triage (RunPod GPU).

Implements the six-gate check sequence from the study spec:
1) GPU/runtime readiness
2) Environment/dependency readiness
3) Base model download + 10-example inference
4) Throwaway LoRA training on SST-2 (rank-8, 50-step smoke by default)
5) Spectral audit (`gradience audit`) on throwaway adapter
6) Merge pipeline check (`gradience merge-audit`) on two throwaway adapters
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GateResult:
    gate: int
    name: str
    passed: bool
    detail: str
    artifact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "artifact": self.artifact,
        }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=capture,
    )


def _gate_gpu_runtime() -> GateResult:
    try:
        import torch

        if not torch.cuda.is_available():
            return GateResult(
                gate=1,
                name="gpu_runtime_readiness",
                passed=False,
                detail="CUDA is not available.",
            )
        dev = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return GateResult(
            gate=1,
            name="gpu_runtime_readiness",
            passed=True,
            detail=f"CUDA available ({dev}, {mem_gb:.1f} GB).",
        )
    except Exception as e:
        return GateResult(gate=1, name="gpu_runtime_readiness", passed=False, detail=str(e))


def _gate_env_deps() -> GateResult:
    missing: list[str] = []
    for mod in ("torch", "transformers", "datasets", "peft"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)

    if missing:
        return GateResult(
            gate=2,
            name="environment_dependency_readiness",
            passed=False,
            detail=f"Missing import(s): {', '.join(missing)}",
        )

    cli_check = _run(["python3", "-m", "gradience", "--help"], capture=True)
    if cli_check.returncode != 0:
        return GateResult(
            gate=2,
            name="environment_dependency_readiness",
            passed=False,
            detail=f"Gradience CLI check failed: {cli_check.stderr.strip()[:300]}",
        )

    return GateResult(
        gate=2,
        name="environment_dependency_readiness",
        passed=True,
        detail="Imports and Gradience CLI are available.",
    )


def _gate_base_model_inference(base_model: str, n_examples: int) -> GateResult:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        prompts = [
            "Classify sentiment: I loved this movie.",
            "Classify sentiment: This was the worst product I bought.",
            "Question entailment: Is the sentence supported by the premise?",
            "Write one sentence about spectral analysis.",
            "Summarize: Adapter compatibility requires both structure and behavior.",
            "Instruction: give two bullet points about LoRA.",
            "Explain why merge audits matter.",
            "Describe one risk of cross-task merging.",
            "Give a short definition of near-miss candidate.",
            "State one practical GPU experiment guardrail.",
        ][:n_examples]

        t0 = time.monotonic()
        tok = AutoTokenizer.from_pretrained(base_model)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        model.eval()

        batch_size = 2
        seen = 0
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
                enc = {k: v.to(model.device) for k, v in enc.items()}
                out = model(**enc)
                _ = out.logits[:, -1, :].argmax(dim=-1)
                seen += len(batch)

        elapsed = time.monotonic() - t0
        # release
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return GateResult(
            gate=3,
            name="base_model_download_and_inference",
            passed=True,
            detail=f"Loaded {base_model}; forward-pass inference succeeded on {seen} examples in {elapsed:.1f}s.",
        )
    except Exception as e:
        return GateResult(gate=3, name="base_model_download_and_inference", passed=False, detail=str(e))


def _train_throwaway_adapter(
    *,
    base_model: str,
    adapter_dir: Path,
    seed: int,
    max_steps: int,
    train_samples: int,
    eval_samples: int,
    rank: int,
) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("glue", "sst2")
    train_ds = ds["train"].shuffle(seed=seed).select(range(min(train_samples, len(ds["train"]))))
    val_ds = ds["validation"].shuffle(seed=seed).select(range(min(eval_samples, len(ds["validation"]))))

    def preprocess(examples: dict[str, Any]) -> dict[str, Any]:
        out = tok(examples["sentence"], truncation=True, padding=True, max_length=128)
        out["labels"] = examples["label"]
        return out

    train_tok = train_ds.map(preprocess, batched=True)
    val_tok = val_ds.map(preprocess, batched=True)
    collator = DataCollatorWithPadding(tok)

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        base = base.to("cuda")
    base_trainer = Trainer(
        model=base,
        args=TrainingArguments(
            output_dir=str(adapter_dir / "_base_eval"),
            per_device_eval_batch_size=8,
            report_to=[],
        ),
        eval_dataset=val_tok,
        data_collator=collator,
    )
    base_preds = base_trainer.predict(val_tok)
    base_acc = float((np.argmax(base_preds.predictions, axis=1) == base_preds.label_ids).mean())
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        modules_to_save=["score", "classifier", "pre_classifier"],
    )
    model = get_peft_model(model, lora)

    train_args = TrainingArguments(
        output_dir=str(adapter_dir / "checkpoints"),
        max_steps=max_steps,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="steps",
        eval_steps=max(10, max_steps // 2),
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        logging_steps=max(5, max_steps // 10),
        seed=seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
    )
    trainer.train()
    preds = trainer.predict(val_tok)
    adapter_acc = float((np.argmax(preds.predictions, axis=1) == preds.label_ids).mean())

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "adapter_dir": str(adapter_dir),
        "base_accuracy": round(base_acc, 6),
        "adapter_accuracy": round(adapter_acc, 6),
        "margin": round(adapter_acc - base_acc, 6),
        "max_steps": max_steps,
        "seed": seed,
        "rank": rank,
    }


def _gate_throwaway_training(
    *,
    base_model: str,
    out_dir: Path,
    steps: int,
    steps_b: int,
    train_samples: int,
    eval_samples: int,
    rank: int,
    seed_a: int,
    seed_b: int,
) -> GateResult:
    try:
        a_dir = out_dir / "throwaway_a"
        b_dir = out_dir / "throwaway_b"
        a_manifest = a_dir / "training_manifest.json"
        b_manifest = b_dir / "training_manifest.json"

        if a_manifest.exists():
            a = json.loads(a_manifest.read_text())
        else:
            a = _train_throwaway_adapter(
                base_model=base_model,
                adapter_dir=a_dir,
                seed=seed_a,
                max_steps=steps,
                train_samples=train_samples,
                eval_samples=eval_samples,
                rank=rank,
            )
            _write_json(a_manifest, a)

        if b_manifest.exists():
            b = json.loads(b_manifest.read_text())
        else:
            b = _train_throwaway_adapter(
                base_model=base_model,
                adapter_dir=b_dir,
                seed=seed_b,
                max_steps=steps_b,
                train_samples=train_samples,
                eval_samples=eval_samples,
                rank=rank,
            )
            _write_json(b_manifest, b)

        detail = (
            f"Trained throwaway adapters: A(margin={a['margin']:+.4f}, steps={a['max_steps']}) "
            f"and B(margin={b['margin']:+.4f}, steps={b['max_steps']})."
        )
        return GateResult(
            gate=4,
            name="throwaway_lora_training",
            passed=True,
            detail=detail,
            artifact=str(out_dir),
        )
    except Exception as e:
        return GateResult(gate=4, name="throwaway_lora_training", passed=False, detail=str(e), artifact=str(out_dir))


def _gate_spectral_audit(throwaway_dir: Path, out_dir: Path) -> GateResult:
    try:
        audit_json = out_dir / "throwaway_a_audit.json"
        cmd = ["python3", "-m", "gradience", "audit", "--peft-dir", str(throwaway_dir), "--json"]
        proc = _run(cmd, capture=True)
        if proc.returncode != 0:
            return GateResult(
                gate=5,
                name="spectral_audit",
                passed=False,
                detail=f"gradience audit failed: {proc.stderr.strip()[:400]}",
            )
        # stdout is JSON
        payload = json.loads(proc.stdout)
        _write_json(audit_json, payload)
        n_layers = payload.get("structural", {}).get("n_layers", payload.get("n_layers", None))
        return GateResult(
            gate=5,
            name="spectral_audit",
            passed=True,
            detail=f"gradience audit succeeded (n_layers={n_layers}).",
            artifact=str(audit_json),
        )
    except Exception as e:
        return GateResult(gate=5, name="spectral_audit", passed=False, detail=str(e))


def _gate_merge_pipeline(adapter_a: Path, adapter_b: Path, out_dir: Path) -> GateResult:
    try:
        report_json = out_dir / "throwaway_merge_report.json"
        cmd = [
            "python3",
            "-m",
            "gradience",
            "merge-audit",
            "--adapter-a",
            str(adapter_a),
            "--adapter-b",
            str(adapter_b),
            "--emit-report",
            str(report_json),
        ]
        proc = _run(cmd, capture=True)
        if proc.returncode != 0:
            return GateResult(
                gate=6,
                name="merge_pipeline",
                passed=False,
                detail=f"gradience merge-audit failed: {proc.stderr.strip()[:400]}",
            )
        if not report_json.exists():
            return GateResult(
                gate=6,
                name="merge_pipeline",
                passed=False,
                detail="merge report was not produced.",
            )
        payload = json.loads(report_json.read_text())
        risk = payload.get("pair_risk", "unknown")
        strategy = payload.get("recommended_strategy", "unknown")
        return GateResult(
            gate=6,
            name="merge_pipeline",
            passed=True,
            detail=f"merge-audit succeeded (pair_risk={risk}, strategy={strategy}).",
            artifact=str(report_json),
        )
    except Exception as e:
        return GateResult(gate=6, name="merge_pipeline", passed=False, detail=str(e))


def _write_markdown_report(path: Path, results: list[GateResult], base_model: str, output_dir: Path) -> None:
    all_pass = all(r.passed for r in results)
    lines: list[str] = []
    lines.append("# Phase 0 Validation Report")
    lines.append("")
    lines.append(f"- Base model: `{base_model}`")
    lines.append(f"- Output dir: `{output_dir}`")
    lines.append(f"- Gate status: **{'PASS' if all_pass else 'FAIL'}**")
    lines.append("")
    lines.append("| Gate | Name | Pass | Detail |")
    lines.append("|---|---|---|---|")
    for r in results:
        lines.append(f"| {r.gate} | {r.name} | {'YES' if r.passed else 'NO'} | {r.detail} |")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for r in results:
        if r.artifact:
            lines.append(f"- Gate {r.gate}: `{r.artifact}`")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 smoke validation for decoder triage RunPod setup.")
    parser.add_argument(
        "--base-model",
        default="mistralai/Mistral-7B-v0.1",
        help="Base model id/path",
    )
    parser.add_argument(
        "--output-dir",
        default="field_trials/decoder_merge_triage/phase0_validation",
        help="Output directory for phase0 artifacts",
    )
    parser.add_argument(
        "--inference-examples",
        type=int,
        default=10,
        help="Number of examples for base-model inference smoke",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=50,
        help="Training steps for throwaway adapter A",
    )
    parser.add_argument(
        "--train-steps-b",
        type=int,
        default=50,
        help="Training steps for throwaway adapter B",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=256,
        help="SST-2 train samples for throwaway training",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=128,
        help="SST-2 eval samples for throwaway training",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA rank for throwaway adapters",
    )
    parser.add_argument("--seed-a", type=int, default=42)
    parser.add_argument("--seed-b", type=int, default=123)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    throwaway_root = out_dir / "throwaway_adapters"
    throwaway_root.mkdir(parents=True, exist_ok=True)

    results: list[GateResult] = []

    print("[gate 1] GPU/runtime readiness")
    g1 = _gate_gpu_runtime()
    results.append(g1)
    print(f"  -> {'PASS' if g1.passed else 'FAIL'}: {g1.detail}")

    print("[gate 2] Environment/dependency readiness")
    g2 = _gate_env_deps()
    results.append(g2)
    print(f"  -> {'PASS' if g2.passed else 'FAIL'}: {g2.detail}")

    print("[gate 3] Base model download + inference")
    g3 = _gate_base_model_inference(args.base_model, n_examples=args.inference_examples)
    results.append(g3)
    print(f"  -> {'PASS' if g3.passed else 'FAIL'}: {g3.detail}")

    print("[gate 4] Throwaway LoRA training")
    g4 = _gate_throwaway_training(
        base_model=args.base_model,
        out_dir=throwaway_root,
        steps=args.train_steps,
        steps_b=args.train_steps_b,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        rank=args.rank,
        seed_a=args.seed_a,
        seed_b=args.seed_b,
    )
    results.append(g4)
    print(f"  -> {'PASS' if g4.passed else 'FAIL'}: {g4.detail}")

    print("[gate 5] Spectral audit")
    g5 = _gate_spectral_audit(throwaway_root / "throwaway_a", out_dir)
    results.append(g5)
    print(f"  -> {'PASS' if g5.passed else 'FAIL'}: {g5.detail}")

    print("[gate 6] Merge pipeline")
    g6 = _gate_merge_pipeline(
        throwaway_root / "throwaway_a",
        throwaway_root / "throwaway_b",
        out_dir,
    )
    results.append(g6)
    print(f"  -> {'PASS' if g6.passed else 'FAIL'}: {g6.detail}")

    payload = {
        "schema": "gradience.decoder_triage.phase0_validation/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": args.base_model,
        "output_dir": str(out_dir),
        "all_passed": all(r.passed for r in results),
        "gates": [r.to_dict() for r in results],
        "config": {
            "inference_examples": args.inference_examples,
            "train_steps": args.train_steps,
            "train_steps_b": args.train_steps_b,
            "train_samples": args.train_samples,
            "eval_samples": args.eval_samples,
            "rank": args.rank,
            "seed_a": args.seed_a,
            "seed_b": args.seed_b,
        },
    }
    json_path = out_dir / "phase0_report.json"
    md_path = out_dir / "phase0_report.md"
    _write_json(json_path, payload)
    _write_markdown_report(md_path, results, base_model=args.base_model, output_dir=out_dir)

    print(f"\nReport JSON: {json_path}")
    print(f"Report MD:   {md_path}")
    print(f"Phase 0 gate: {'PASS' if payload['all_passed'] else 'FAIL'}")
    if not payload["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
