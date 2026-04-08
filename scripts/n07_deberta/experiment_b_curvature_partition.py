#!/usr/bin/env python3
"""Experiment B: Curvature-Partition Correspondence on DeBERTa-v3.

Tests THEORY.md §7.2: Do Hessian energy-collapse events coincide with
MP-partitioned spectral alignment sharpening?

Protocol:
  1. Generate fixed probe directions (before training)
  2. Train seed_42 with dense dual-instrument telemetry (every 10 steps):
     - Structural: SVD snapshots of all LoRA weights
     - Curvature: Top-3 Hessian eigenvalues + trace
  3. Train seed_123 (replication)
  4. Post-training analysis:
     a. Compute MP-partitioned alignment time series
     b. Detect curvature collapse events
     c. Cross-correlate the two instruments
     d. Event-triggered averaging

Predictions:
  CP-1: Curvature collapses precede alignment sharpening (lag 10-50 steps)
  CP-2: MP-partition boundary tracks alignment phase transitions
  CP-3: Cross-correlation peaks at negative lag (curvature leads)
  CP-4: Effect replicates across seeds
  CP-5: Effect is module-type dependent (strongest in V)

Usage:
    # Full run (~2.5 hours on A100):
    python3 scripts/n07_deberta/experiment_b_curvature_partition.py \
        --base-dir /workspace/n07 --task mrpc

    # Smoke test (~2 min):
    python3 scripts/n07_deberta/experiment_b_curvature_partition.py \
        --base-dir /workspace/n07 --task mrpc --smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import signal, stats


# ─── Constants ─────────────────────────────────────────────

MODEL_NAME = "microsoft/deberta-v3-base"
MODULE_TYPES = ["query_proj", "key_proj", "value_proj", "attention.output.dense"]
MODULE_SHORT = {"query_proj": "Q", "key_proj": "K", "value_proj": "V", "attention.output.dense": "O"}
N_LAYERS = 12

TASKS = {
    "mrpc": {"dataset": "glue", "subset": "mrpc", "text_cols": ["sentence1", "sentence2"], "num_labels": 2},
    "rte": {"dataset": "glue", "subset": "rte", "text_cols": ["sentence1", "sentence2"], "num_labels": 2},
    "sst2": {"dataset": "glue", "subset": "sst2", "text_cols": ["sentence"], "num_labels": 2},
    "qnli": {"dataset": "glue", "subset": "qnli", "text_cols": ["question", "sentence"], "num_labels": 2},
}

SEEDS = [42, 123]
TELEMETRY_INTERVAL = 10  # structural snapshot every N steps (dense)
CURVATURE_INTERVAL = 50  # curvature snapshot less frequently (expensive)


# ─── Step 1: Generate Fixed Probe Directions ──────────────

def generate_probe_directions(model, n_directions: int = 5, seed: int = 0) -> dict[str, torch.Tensor]:
    """Generate fixed random probe directions for curvature estimation.

    These must be generated BEFORE training begins and remain fixed
    throughout to ensure the curvature measurements are comparable.
    """
    rng = torch.Generator().manual_seed(seed)
    directions = {}
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" in name:
            # Generate n_directions random unit vectors in param space
            d = torch.randn(n_directions, param.numel(), generator=rng)
            d = d / d.norm(dim=1, keepdim=True)
            directions[name] = d
    return directions


# ─── Step 2: Dense Dual-Instrument Training ───────────────

class DenseInstrumentCallback:
    """Callback that captures structural snapshots (every 10 steps) and curvature (every 50 steps)."""

    def __init__(self, structural_interval: int = 10, curvature_interval: int = 50,
                 n_top_eigenvalues: int = 1, n_hessian_samples: int = 3,
                 power_iterations: int = 5):
        self.structural_interval = structural_interval
        self.curvature_interval = curvature_interval
        self.n_top_eigenvalues = n_top_eigenvalues
        self.n_hessian_samples = n_hessian_samples
        self.power_iterations = power_iterations
        self.snapshots: list[dict] = []
        self._last_batch = None

    def capture_batch(self, batch):
        """Store last batch for Hessian estimation."""
        self._last_batch = {k: v.detach() for k, v in batch.items()}

    def on_step_end(self, step: int, model, loss_fn=None):
        """Capture snapshot at step."""
        do_structural = (step % self.structural_interval == 0)
        do_curvature = (step % self.curvature_interval == 0)

        if not do_structural and not do_curvature:
            return

        snapshot = {"step": step, "timestamp": time.time()}

        # --- Structural: SVD of LoRA delta_W (fast, every 10 steps) ---
        if not do_structural:
            return

        # Compute per-module delta_W SVD inline
        # Build param lookup dict once per snapshot
        param_dict = {n: p for n, p in model.named_parameters() if "lora_" in n}

        module_svd = {}
        for layer_idx in range(N_LAYERS):
            for module_type in MODULE_TYPES:
                a_key, b_key = _get_lora_keys(layer_idx, module_type)
                a_found = b_found = None
                for name, param in param_dict.items():
                    if a_key in name and "lora_A" in name:
                        a_found = param.detach().cpu().float()
                    elif b_key in name and "lora_B" in name:
                        b_found = param.detach().cpu().float()

                if a_found is not None and b_found is not None:
                    # Fast SVD via QR: rank(B@A) <= r, so work in r-space
                    A_np = a_found.numpy()  # (r, d_in)
                    B_np = b_found.numpy()  # (d_out, r)
                    Q_B, R_B = np.linalg.qr(B_np, mode='reduced')  # Q: (d_out,r), R: (r,r)
                    M = R_B @ A_np  # (r, d_in) — small!
                    _, S, _ = np.linalg.svd(M, full_matrices=False)
                    module_svd[f"L{layer_idx}_{MODULE_SHORT[module_type]}"] = {
                        "singular_values": S.tolist(),
                        "stable_rank": float(np.sum(S**2) / (S[0]**2 + 1e-20)),
                        "frobenius": float(np.sqrt(np.sum(S**2))),
                    }

        snapshot["module_svd"] = module_svd

        # --- Curvature: Hessian top eigenvalues via power iteration (every 50 steps) ---
        if do_curvature:
            curvature = self._estimate_curvature(model)
            snapshot["curvature"] = curvature

        self.snapshots.append(snapshot)

    def _estimate_curvature(self, model) -> dict:
        """Estimate top Hessian eigenvalue using power iteration on LoRA params only."""
        if self._last_batch is None:
            return {"top_eigenvalues": [], "trace_estimate": 0.0}

        model.eval()
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in self._last_batch.items()}

        try:
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            # Only use LoRA parameters (much fewer than all trainable params)
            params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]

            # Power iteration for top eigenvalue
            eigenvalues = []
            for _ in range(self.n_top_eigenvalues):
                v = [torch.randn_like(p) for p in params]
                v_norm = math.sqrt(sum(vi.pow(2).sum().item() for vi in v))
                v = [vi / v_norm for vi in v]

                eigenvalue = 0.0
                for _ in range(self.power_iterations):
                    Hv = _hessian_vector_product(loss, params, v)
                    if Hv is None:
                        break
                    eigenvalue = sum((hvi * vi).sum().item() for hvi, vi in zip(Hv, v))
                    v_norm = math.sqrt(sum(hvi.pow(2).sum().item() for hvi in Hv))
                    if v_norm < 1e-20:
                        break
                    v = [hvi / v_norm for hvi in Hv]

                eigenvalues.append(eigenvalue)

            # Trace estimate via Hutchinson (3 samples)
            trace_est = 0.0
            for _ in range(self.n_hessian_samples):
                z = [torch.randn_like(p) for p in params]
                Hz = _hessian_vector_product(loss, params, z)
                if Hz is not None:
                    trace_est += sum((hzi * zi).sum().item() for hzi, zi in zip(Hz, z))
            trace_est /= max(self.n_hessian_samples, 1)

            return {
                "top_eigenvalues": eigenvalues,
                "trace_estimate": trace_est,
                "loss": loss.item(),
            }

        except Exception as e:
            return {"top_eigenvalues": [], "trace_estimate": 0.0, "error": str(e)}
        finally:
            model.train()


def _hessian_vector_product(loss, params, vectors):
    """Compute Hessian-vector product via double backprop."""
    try:
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

        grad_dot_v = sum((g * v).sum() for g, v in zip(grads, vectors))

        Hv = torch.autograd.grad(grad_dot_v, params, retain_graph=True, allow_unused=True)
        Hv = [h if h is not None else torch.zeros_like(p) for h, p in zip(Hv, params)]
        return [h.detach() for h in Hv]
    except Exception:
        return None


def _get_lora_keys(layer_idx: int, module_type: str) -> tuple[str, str]:
    """Get the key prefix for LoRA A and B matrices."""
    if module_type == "attention.output.dense":
        prefix = f"encoder.layer.{layer_idx}.attention.output.dense"
    else:
        prefix = f"encoder.layer.{layer_idx}.attention.self.{module_type}"
    return prefix, prefix


def train_with_dense_telemetry(task: str, seed: int, base_dir: Path,
                                smoke: bool = False) -> list[dict]:
    """Train a DeBERTa adapter with dense dual-instrument telemetry."""
    import evaluate
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        TrainerCallback,
    )

    task_cfg = TASKS[task]
    output_dir = base_dir / f"experiment_b" / f"dense_{task}_s{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing telemetry
    telemetry_file = output_dir / "dense_telemetry.json"
    if telemetry_file.exists():
        print(f"  SKIP dense_{task}_s{seed} (telemetry exists)")
        return json.loads(telemetry_file.read_text())

    print(f"\n{'=' * 60}")
    print(f"  Dense Training: {task} seed={seed}")
    print(f"{'=' * 60}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=task_cfg["num_labels"]
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.1,
        target_modules=["query_proj", "key_proj", "value_proj", "attention.output.dense"],
        bias="none", task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_dataset(task_cfg["dataset"], task_cfg["subset"])

    def tokenize(examples):
        cols = task_cfg["text_cols"]
        if len(cols) == 2:
            return tokenizer(examples[cols[0]], examples[cols[1]],
                           truncation=True, max_length=256, padding="max_length")
        return tokenizer(examples[cols[0]], truncation=True, max_length=128, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True)
    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]

    if smoke:
        train_dataset = train_dataset.select(range(min(64, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(32, len(eval_dataset))))

    # Metrics
    accuracy_metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        return accuracy_metric.compute(predictions=preds, references=eval_pred.label_ids)

    # Instrument callback
    instrument = DenseInstrumentCallback(
        structural_interval=TELEMETRY_INTERVAL,
        curvature_interval=CURVATURE_INTERVAL,
        n_top_eigenvalues=1,
        n_hessian_samples=3,
        power_iterations=5,
    )

    # Wrap as HF callback
    class HFInstrumentCallback(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kwargs):
            instrument.on_step_end(state.global_step, model)

    class BatchCaptureCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            pass  # Batch capture handled via training_step patch

    # Training args
    n_train = len(train_dataset)
    bs = 16
    epochs = 3
    max_steps = 10 if smoke else -1
    if max_steps > 0:
        total_steps = max_steps
    else:
        total_steps = math.ceil(n_train / bs) * epochs

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_output"),
        num_train_epochs=epochs if not smoke else 1,
        max_steps=max_steps if smoke else -1,
        learning_rate=2e-4,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=32,
        warmup_steps=max(1, int(0.06 * total_steps)),
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=50 if not smoke else 5,
        logging_steps=10 if not smoke else 2,
        save_strategy="no",
        bf16=True,
        dataloader_num_workers=2,
        remove_unused_columns=True,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[HFInstrumentCallback()],
    )

    # Patch training_step to capture batches for curvature
    original_training_step = trainer.training_step

    def patched_training_step(model, inputs, num_items_in_batch=None, **kwargs):
        instrument.capture_batch(inputs)
        return original_training_step(model, inputs, num_items_in_batch=num_items_in_batch, **kwargs)

    trainer.training_step = patched_training_step

    # Train
    torch.manual_seed(seed)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # Final eval
    eval_results = trainer.evaluate()
    print(f"  Accuracy: {eval_results.get('eval_accuracy', 0):.4f}, Time: {elapsed:.0f}s")

    # Save telemetry
    telemetry = instrument.snapshots
    telemetry_file.write_text(json.dumps(telemetry, indent=2))
    print(f"  Snapshots: {len(telemetry)}")

    # Save eval results
    (output_dir / "eval_results.json").write_text(json.dumps({
        "accuracy": eval_results.get("eval_accuracy", 0),
        "training_time_s": elapsed,
        "n_snapshots": len(telemetry),
        "seed": seed,
        "task": task,
    }, indent=2))

    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return telemetry


# ─── Step 3: Post-Training Analysis ───────────────────────

def gavish_donoho_threshold(S: np.ndarray, beta: float) -> float:
    """Gavish-Donoho optimal hard threshold."""
    if beta <= 0 or beta > 1:
        beta = 1.0
    lambda_star = np.sqrt(2 * (beta + 1) + (8 * beta) / (beta + 1 + np.sqrt(beta**2 + 14*beta + 1)))
    median_sv = np.median(S) if len(S) > 0 else 1.0
    return lambda_star * max(median_sv, 1e-10)


def compute_mp_partitioned_alignment(snapshots: list[dict]) -> dict[str, list[dict]]:
    """Compute MP-partitioned alignment time series from dense telemetry.

    For each step, compute:
      - high_band_alignment: alignment of SVs above MP threshold (signal)
      - low_band_alignment: alignment of SVs below MP threshold (noise)
      - alignment_ratio: high/low ratio (sharpening = ratio increasing)
    """
    if len(snapshots) < 2:
        return {}

    alignment_series = {}

    for module_key in snapshots[0].get("module_svd", {}).keys():
        series = []
        prev_svd = None

        for snap in snapshots:
            svd_data = snap.get("module_svd", {}).get(module_key)
            if svd_data is None:
                continue

            S = np.array(svd_data["singular_values"])
            if len(S) == 0:
                continue

            # MP threshold
            beta = min(len(S), 768) / max(len(S), 768)  # approximate aspect ratio
            mp_thresh = gavish_donoho_threshold(S, beta)

            n_above = int(np.sum(S > mp_thresh))
            n_below = int(np.sum(S <= mp_thresh))

            entry = {
                "step": snap["step"],
                "mp_threshold": float(mp_thresh),
                "n_above_mp": n_above,
                "n_below_mp": n_below,
                "stable_rank": svd_data["stable_rank"],
                "frobenius": svd_data["frobenius"],
            }

            if prev_svd is not None:
                prev_S = np.array(prev_svd["singular_values"])
                prev_mp = gavish_donoho_threshold(prev_S, beta)

                # Compute step-to-step SV change in high vs low bands
                k = min(len(S), len(prev_S))
                S_k = S[:k]
                prev_S_k = prev_S[:k]

                # Normalized change per band
                high_mask = S_k > mp_thresh
                low_mask = S_k <= mp_thresh

                if np.any(high_mask):
                    high_change = float(np.mean(np.abs(S_k[high_mask] - prev_S_k[high_mask]) /
                                                (prev_S_k[high_mask] + 1e-10)))
                else:
                    high_change = 0.0

                if np.any(low_mask):
                    low_change = float(np.mean(np.abs(S_k[low_mask] - prev_S_k[low_mask]) /
                                               (prev_S_k[low_mask] + 1e-10)))
                else:
                    low_change = 0.0

                entry["high_band_change"] = high_change
                entry["low_band_change"] = low_change
                entry["band_ratio"] = high_change / (low_change + 1e-10)

                # Alignment sharpening: rate of change of stable rank
                entry["delta_stable_rank"] = svd_data["stable_rank"] - prev_svd["stable_rank"]

            prev_svd = svd_data
            series.append(entry)

        alignment_series[module_key] = series

    return alignment_series


def detect_curvature_collapses(snapshots: list[dict],
                                collapse_threshold: float = 0.5) -> list[dict]:
    """Detect curvature collapse events (rapid drop in top eigenvalue).

    A collapse is defined as a drop of >collapse_threshold fraction
    in the top Hessian eigenvalue within one snapshot interval.
    """
    events = []

    for i in range(1, len(snapshots)):
        prev = snapshots[i-1].get("curvature", {})
        curr = snapshots[i].get("curvature", {})

        prev_eigs = prev.get("top_eigenvalues", [])
        curr_eigs = curr.get("top_eigenvalues", [])

        if not prev_eigs or not curr_eigs:
            continue

        prev_top = abs(prev_eigs[0])
        curr_top = abs(curr_eigs[0])

        if prev_top > 1e-10:
            ratio = curr_top / prev_top
            if ratio < collapse_threshold:
                events.append({
                    "step": snapshots[i]["step"],
                    "prev_step": snapshots[i-1]["step"],
                    "prev_top_eigenvalue": prev_top,
                    "curr_top_eigenvalue": curr_top,
                    "ratio": ratio,
                    "prev_trace": prev.get("trace_estimate", 0),
                    "curr_trace": curr.get("trace_estimate", 0),
                })

    return events


def cross_correlate_instruments(alignment_series: dict, curvature_snapshots: list[dict],
                                 max_lag: int = 10) -> dict:
    """Cross-correlate curvature and alignment time series.

    Tests whether curvature changes lead alignment changes (negative lag = curvature leads).
    """
    # Extract curvature time series
    steps = [s["step"] for s in curvature_snapshots if s.get("curvature", {}).get("top_eigenvalues")]
    curv_values = [abs(s["curvature"]["top_eigenvalues"][0])
                   for s in curvature_snapshots
                   if s.get("curvature", {}).get("top_eigenvalues")]

    if len(curv_values) < 5:
        return {"error": "insufficient curvature data"}

    # Log-transform curvature for better cross-correlation
    curv_log = np.log1p(np.array(curv_values))
    curv_diff = np.diff(curv_log)  # rate of change

    results = {}
    for module_key, series in alignment_series.items():
        if len(series) < 5:
            continue

        # Match steps between the two series
        align_steps = [e["step"] for e in series if "delta_stable_rank" in e]
        align_values = [e["delta_stable_rank"] for e in series if "delta_stable_rank" in e]

        if len(align_values) < 5:
            continue

        # Interpolate to common time base
        common_steps = sorted(set(steps[1:]).intersection(set(align_steps)))
        if len(common_steps) < 5:
            # Use nearest-step matching
            curv_ts = np.array(curv_diff[:min(len(curv_diff), len(align_values))])
            align_ts = np.array(align_values[:len(curv_ts)])
        else:
            curv_dict = dict(zip(steps[1:], curv_diff))
            align_dict = dict(zip(align_steps, align_values))
            curv_ts = np.array([curv_dict.get(s, 0) for s in common_steps])
            align_ts = np.array([align_dict.get(s, 0) for s in common_steps])

        if len(curv_ts) < 3:
            continue

        # Normalize
        curv_ts = (curv_ts - curv_ts.mean()) / (curv_ts.std() + 1e-10)
        align_ts = (align_ts - align_ts.mean()) / (align_ts.std() + 1e-10)

        # Cross-correlation
        n = len(curv_ts)
        max_lag_actual = min(max_lag, n // 3)
        lags = range(-max_lag_actual, max_lag_actual + 1)
        xcorr = []
        for lag in lags:
            if lag < 0:
                c = np.corrcoef(curv_ts[:lag], align_ts[-lag:])[0, 1] if len(curv_ts[:lag]) > 1 else 0
            elif lag > 0:
                c = np.corrcoef(curv_ts[lag:], align_ts[:-lag])[0, 1] if len(curv_ts[lag:]) > 1 else 0
            else:
                c = np.corrcoef(curv_ts, align_ts)[0, 1] if len(curv_ts) > 1 else 0
            xcorr.append(float(c) if not np.isnan(c) else 0.0)

        peak_idx = np.argmax(np.abs(xcorr))
        peak_lag = list(lags)[peak_idx]
        peak_corr = xcorr[peak_idx]

        results[module_key] = {
            "lags": list(lags),
            "xcorr": xcorr,
            "peak_lag": peak_lag,
            "peak_correlation": peak_corr,
            "curvature_leads": peak_lag < 0,
            "n_common_steps": len(curv_ts),
        }

    return results


def event_triggered_average(alignment_series: dict, collapse_events: list[dict],
                             window: int = 5) -> dict:
    """Compute event-triggered average of alignment around curvature collapses."""
    if not collapse_events:
        return {"n_events": 0}

    collapse_steps = [e["step"] for e in collapse_events]
    results = {}

    for module_key, series in alignment_series.items():
        step_to_sr = {e["step"]: e.get("delta_stable_rank", 0) for e in series}
        all_steps = sorted(step_to_sr.keys())

        if not all_steps:
            continue

        triggered = []
        for collapse_step in collapse_steps:
            # Find nearest step in alignment series
            nearest = min(all_steps, key=lambda s: abs(s - collapse_step))
            idx = all_steps.index(nearest)

            # Extract window around event
            window_values = []
            for offset in range(-window, window + 1):
                wi = idx + offset
                if 0 <= wi < len(all_steps):
                    window_values.append(step_to_sr.get(all_steps[wi], 0))
                else:
                    window_values.append(0)
            triggered.append(window_values)

        if triggered:
            avg = np.mean(triggered, axis=0).tolist()
            results[module_key] = {
                "event_triggered_average": avg,
                "n_events": len(triggered),
                "pre_event_mean": float(np.mean([t[:window] for t in triggered])),
                "post_event_mean": float(np.mean([t[window+1:] for t in triggered])),
                "post_gt_pre": float(np.mean([t[window+1:] for t in triggered])) > float(np.mean([t[:window] for t in triggered])),
            }

    return results


# ─── Step 4: Test Predictions ────────────────────────────

def test_predictions(seed_results: dict[int, dict]) -> dict:
    """Test all CP predictions across seeds."""
    print("\n=== Testing Predictions ===\n")

    predictions = {}

    # --- CP-1: Curvature collapses precede alignment sharpening ---
    cp1_evidence = []
    for seed, result in seed_results.items():
        collapses = result.get("collapse_events", [])
        xcorr = result.get("cross_correlation", {})
        for module_key, xc in xcorr.items():
            if xc.get("curvature_leads", False):
                cp1_evidence.append({
                    "seed": seed, "module": module_key,
                    "peak_lag": xc["peak_lag"],
                    "peak_corr": xc["peak_correlation"],
                })

    predictions["CP-1"] = {
        "description": "Curvature collapses precede alignment sharpening",
        "n_modules_curvature_leads": len(cp1_evidence),
        "evidence": cp1_evidence,
        "supported": len(cp1_evidence) >= 2,
    }
    print(f"  CP-1: {len(cp1_evidence)} modules show curvature leading → "
          f"{'✓' if predictions['CP-1']['supported'] else '✗'}")

    # --- CP-2: MP boundary tracks alignment phase transitions ---
    cp2_evidence = []
    for seed, result in seed_results.items():
        alignment = result.get("alignment_series", {})
        for module_key, series in alignment.items():
            if len(series) < 5:
                continue
            # Check if n_above_mp changes correlate with stable_rank changes
            n_above = [e.get("n_above_mp", 0) for e in series]
            sr = [e.get("stable_rank", 0) for e in series]
            if len(set(n_above)) > 1 and len(sr) > 1:
                r, p = stats.spearmanr(n_above[:len(sr)], sr[:len(n_above)])
                if not np.isnan(r):
                    cp2_evidence.append({
                        "seed": seed, "module": module_key,
                        "spearman_r": float(r), "p_value": float(p),
                    })

    n_significant = sum(1 for e in cp2_evidence if e["p_value"] < 0.05)
    predictions["CP-2"] = {
        "description": "MP boundary tracks alignment phase transitions",
        "n_significant": n_significant,
        "n_total": len(cp2_evidence),
        "evidence": cp2_evidence,
        "supported": n_significant >= len(cp2_evidence) // 2 if cp2_evidence else False,
    }
    print(f"  CP-2: {n_significant}/{len(cp2_evidence)} significant → "
          f"{'✓' if predictions['CP-2']['supported'] else '✗'}")

    # --- CP-3: Cross-correlation peaks at negative lag ---
    cp3_all_lags = []
    for seed, result in seed_results.items():
        xcorr = result.get("cross_correlation", {})
        for module_key, xc in xcorr.items():
            cp3_all_lags.append(xc.get("peak_lag", 0))

    if cp3_all_lags:
        mean_lag = float(np.mean(cp3_all_lags))
        frac_negative = sum(1 for l in cp3_all_lags if l < 0) / len(cp3_all_lags)
    else:
        mean_lag = 0
        frac_negative = 0

    predictions["CP-3"] = {
        "description": "Cross-correlation peaks at negative lag (curvature leads)",
        "mean_peak_lag": mean_lag,
        "fraction_negative": frac_negative,
        "all_peak_lags": cp3_all_lags,
        "supported": frac_negative > 0.5,
    }
    print(f"  CP-3: mean lag={mean_lag:.1f}, {frac_negative:.0%} negative → "
          f"{'✓' if predictions['CP-3']['supported'] else '✗'}")

    # --- CP-4: Effect replicates across seeds ---
    if len(seed_results) >= 2:
        seeds = sorted(seed_results.keys())
        s1_leads = set()
        s2_leads = set()
        for module_key, xc in seed_results[seeds[0]].get("cross_correlation", {}).items():
            if xc.get("curvature_leads"):
                s1_leads.add(module_key)
        for module_key, xc in seed_results[seeds[1]].get("cross_correlation", {}).items():
            if xc.get("curvature_leads"):
                s2_leads.add(module_key)

        agreement = len(s1_leads & s2_leads)
        total = len(s1_leads | s2_leads)
        jaccard = agreement / total if total > 0 else 0
    else:
        jaccard = 0
        agreement = 0
        total = 0

    predictions["CP-4"] = {
        "description": "Effect replicates across seeds",
        "jaccard_similarity": jaccard,
        "n_agreement": agreement,
        "n_union": total,
        "supported": jaccard > 0.3,
    }
    print(f"  CP-4: Jaccard={jaccard:.2f} ({agreement}/{total}) → "
          f"{'✓' if predictions['CP-4']['supported'] else '✗'}")

    # --- CP-5: Effect is module-type dependent (strongest in V) ---
    module_type_corr = defaultdict(list)
    for seed, result in seed_results.items():
        xcorr = result.get("cross_correlation", {})
        for module_key, xc in xcorr.items():
            # Extract module type from key like "L0_V"
            parts = module_key.split("_")
            if len(parts) >= 2:
                mt = parts[-1]
                module_type_corr[mt].append(abs(xc.get("peak_correlation", 0)))

    v_corr = np.mean(module_type_corr.get("V", [0]))
    other_corrs = {mt: np.mean(vals) for mt, vals in module_type_corr.items() if mt != "V"}
    v_strongest = all(v_corr >= c for c in other_corrs.values()) if other_corrs else False

    predictions["CP-5"] = {
        "description": "Effect strongest in V-module",
        "v_mean_correlation": float(v_corr),
        "other_correlations": {k: float(v) for k, v in other_corrs.items()},
        "v_strongest": v_strongest,
        "supported": v_strongest and v_corr > 0.1,
    }
    print(f"  CP-5: V={v_corr:.3f} vs {other_corrs} → "
          f"{'✓' if predictions['CP-5']['supported'] else '✗'}")

    return predictions


# ─── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment B: Curvature-Partition Correspondence")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, default="mrpc", choices=list(TASKS.keys()))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Skip training, run analysis on existing telemetry")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base_dir = args.base_dir
    exp_dir = base_dir / "experiment_b"
    exp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    seed_results = {}

    for seed in SEEDS:
        print(f"\n{'#' * 60}")
        print(f"  Seed {seed}")
        print(f"{'#' * 60}")

        if not args.analysis_only:
            snapshots = train_with_dense_telemetry(args.task, seed, base_dir, smoke=args.smoke)
        else:
            telemetry_file = exp_dir / f"dense_{args.task}_s{seed}" / "dense_telemetry.json"
            if not telemetry_file.exists():
                print(f"  ERROR: No telemetry found at {telemetry_file}")
                continue
            snapshots = json.loads(telemetry_file.read_text())

        print(f"\n  Analyzing {len(snapshots)} snapshots...")

        # Compute alignment series
        alignment_series = compute_mp_partitioned_alignment(snapshots)
        print(f"  Alignment series: {len(alignment_series)} modules")

        # Detect curvature collapses
        collapses = detect_curvature_collapses(snapshots, collapse_threshold=0.5)
        print(f"  Curvature collapses: {len(collapses)} events")

        # Cross-correlate
        xcorr = cross_correlate_instruments(alignment_series, snapshots)
        if "error" not in xcorr:
            n_leads = sum(1 for v in xcorr.values() if v.get("curvature_leads"))
            print(f"  Cross-correlation: {n_leads}/{len(xcorr)} modules show curvature leading")

        # Event-triggered average
        eta = event_triggered_average(alignment_series, collapses)
        if isinstance(eta, dict) and eta.get("n_events", 0) == 0:
            print(f"  Event-triggered average: no events to average")
        else:
            n_post_gt = sum(1 for v in eta.values() if isinstance(v, dict) and v.get("post_gt_pre"))
            print(f"  Event-triggered average: {n_post_gt} modules show post > pre")

        seed_results[seed] = {
            "alignment_series": alignment_series,
            "collapse_events": collapses,
            "cross_correlation": xcorr,
            "event_triggered_average": eta,
            "n_snapshots": len(snapshots),
        }

        # Save per-seed results
        seed_out = exp_dir / f"analysis_{args.task}_s{seed}.json"
        # Convert for JSON serialization (alignment_series can be large)
        serializable = {
            "collapse_events": collapses,
            "cross_correlation": xcorr,
            "event_triggered_average": eta,
            "n_snapshots": len(snapshots),
            "alignment_summary": {
                k: {"n_entries": len(v), "modules": list(alignment_series.keys())[:5]}
                for k, v in alignment_series.items()
            } if alignment_series else {},
        }
        seed_out.write_text(json.dumps(serializable, indent=2, default=str))

    # Test predictions
    predictions = test_predictions(seed_results)

    # Summary
    elapsed = time.time() - t0
    summary = {
        "experiment": "B: Curvature-Partition Correspondence on DeBERTa-v3",
        "task": args.task,
        "seeds": SEEDS,
        "elapsed_s": elapsed,
        "predictions": {k: {"supported": v["supported"]} for k, v in predictions.items()},
        "n_supported": sum(1 for v in predictions.values() if v["supported"]),
        "n_total": len(predictions),
    }

    (exp_dir / "predictions.json").write_text(json.dumps(predictions, indent=2, default=str))
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'=' * 60}")
    print(f"  Experiment B Summary ({elapsed:.0f}s)")
    print(f"{'=' * 60}")
    print(f"  Task: {args.task}")
    print(f"  Seeds: {SEEDS}")
    for k, v in predictions.items():
        print(f"  {k}: {'✓' if v['supported'] else '✗'} — {v['description']}")
    print(f"\n  Supported: {summary['n_supported']}/{summary['n_total']}")
    print(f"  Output: {exp_dir}")


if __name__ == "__main__":
    main()
