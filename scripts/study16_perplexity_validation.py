#!/usr/bin/env python3
"""
Study 16 Perplexity Validation.

Determines whether spectral merge quality metrics (Q_min, D, cosine retention)
predict downstream task performance.  Evaluates source adapters, merged adapters
(naive, norm-equalized, and recommended), and the bare base model on held-out
data, measuring token-weighted perplexity throughout.

Because Study 16's spectral pipeline deletes merged .safetensors after
evaluation, this script re-executes the merges to produce loadable PEFT
adapters before running GPU-side perplexity evaluation.

Pipeline:
    Phase 1 (CPU)  — Download source adapters, re-run merge audit,
                      re-execute merges (3 conditions), save PEFT adapter
                      dirs with weights.
    Phase 2 (GPU)  — Load base model once, load eval datasets by task,
                      evaluate base → source → merged adapters, swapping
                      PEFT weights between evaluations.

Merge conditions:
    - naive:           uniform_linear (0.5/0.5 coefficients)
    - norm_equalized:  uniform_linear with global Frobenius-proportional
                       reweighting (isolates magnitude correction from
                       the full layerwise engine)
    - recommended:     audit_aware (full 6-branch layerwise engine)

Methodological notes (per external review):
    - Token-weighted NLL, not example-averaged loss
    - Completion-only scoring where prompt/completion structure exists
    - Explicit per-dataset formatters (no fallback to str(example))
    - Deterministic example selection via fixed seed
    - Per-example loss logging for tail inspection
    - Clean adapter reinstantiation (no PEFT unload/reuse)
    - Separate eval sets for same-domain pairs with different provenance

Usage:
    # Full run (merge + eval):
    python scripts/study16_perplexity_validation.py \\
        --base-model meta-llama/Llama-2-7b-hf \\
        --output-dir results/study16_perplexity \\
        --cache-dir ~/.cache/gradience/adapters \\
        --max-examples 500 \\
        --verbose

    # Skip merge phase if adapter dirs already have weights:
    python scripts/study16_perplexity_validation.py \\
        --base-model meta-llama/Llama-2-7b-hf \\
        --output-dir results/study16_perplexity \\
        --skip-merge \\
        --max-examples 500

    # CPU dry-run (test imports / paths, no GPU):
    python scripts/study16_perplexity_validation.py \\
        --base-model meta-llama/Llama-2-7b-hf \\
        --output-dir results/study16_perplexity \\
        --max-examples 2 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("study16_ppl")

SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# Pair definitions  (mirrors study16_merge_ablation.py)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AdapterDef:
    """A single adapter with its HF repo, label, and task domain."""

    repo: str
    label: str
    task: str  # math | code | chat


@dataclass
class EvalSetDef:
    """An evaluation set tied to a specific adapter or domain."""

    eval_id: str  # unique key, e.g. "metamath_math"
    task: str  # math | code | chat
    dataset: str  # HF dataset name
    subset: str | None  # HF dataset subset
    split: str  # HF split
    description: str
    adapter_label: str | None  # if adapter-specific, else None (generic)


@dataclass
class PairDef:
    """An adapter pair to merge and evaluate."""

    pair_id: str
    adapter_a: AdapterDef
    adapter_b: AdapterDef
    expected_verdict: str
    eval_sets: list[str] = field(default_factory=list)  # eval_id references


# ─── Unique source adapters ──────────────────────────────────────────────

SOURCE_ADAPTERS: list[AdapterDef] = [
    AdapterDef("LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32", "metamath-r16", "math"),
    AdapterDef("LoRA-TMLR-2024/openwebmath-lora-rank-16-20B-tokens", "openwebmath-r16", "math"),
    AdapterDef("LoRA-TMLR-2024/openwebmath-lora-rank-64-20B-tokens", "openwebmath-r64", "math"),
    AdapterDef("LoRA-TMLR-2024/magicoder-lora-rank-16-alpha-32", "magicoder-r16", "code"),
    AdapterDef("AIRLab-POLIMI/llama-2-7b-chat-hf-btgenbot-adapter", "btgenbot-r8", "chat"),
    AdapterDef("shivanikerai/Llama-2-7b-chat-hf-adapter-cat-subcat-mapping-v2.0", "catsubcat-r16", "chat"),
]

# ─── Eval set definitions ────────────────────────────────────────────────
# Adapter-specific where possible; generic domain benchmarks as secondary.
#
# metamath → trained on MetaMathQA; eval on GSM8K (standard math reasoning)
# openwebmath → trained on OpenWebMath; eval on GSM8K as well BUT these are
#   different enough training distributions that same-domain pairs still need
#   careful handling.  We use GSM8K for both since it's a neutral math bench.
#   However, Pair 01 gets TWO eval rows: one per source adapter, both on GSM8K,
#   to keep the analysis structure parallel with cross-domain pairs.
# magicoder → trained on OSS-Instruct code data; eval on MBPP
# btgenbot → chat/instruction adapter; eval on oasst2
# catsubcat → category-mapping chat adapter; eval on oasst2
#   Pair 06 similarly gets two eval rows for structural parallelism.

EVAL_SETS: dict[str, EvalSetDef] = {
    "gsm8k_math": EvalSetDef(
        eval_id="gsm8k_math",
        task="math",
        dataset="gsm8k",
        subset="main",
        split="test",
        description="GSM8K test (math reasoning)",
        adapter_label=None,  # generic math benchmark
    ),
    "mbpp_code": EvalSetDef(
        eval_id="mbpp_code",
        task="code",
        dataset="mbpp",
        subset="full",
        split="test",
        description="MBPP test (code generation)",
        adapter_label=None,
    ),
    "oasst2_chat": EvalSetDef(
        eval_id="oasst2_chat",
        task="chat",
        dataset="OpenAssistant/oasst2",
        subset=None,
        split="validation",
        description="OpenAssistant oasst2 validation (chat/instruction)",
        adapter_label=None,
    ),
}

# ─── Pair definitions ────────────────────────────────────────────────────
# Same-domain pairs get separate eval entries for EACH source adapter.
# This prevents collapsing into a single eval that blurs provenance diffs.

PAIRS: list[PairDef] = [
    PairDef(
        pair_id="pair_01",
        adapter_a=SOURCE_ADAPTERS[0],  # metamath-r16
        adapter_b=SOURCE_ADAPTERS[1],  # openwebmath-r16
        expected_verdict="redundant",
        # Same-domain: both math.  Eval each source on GSM8K separately.
        # "task_a_eval" and "task_b_eval" both point to gsm8k_math,
        # but we still run two rows in the analysis for structural parallelism.
        eval_sets=["gsm8k_math", "gsm8k_math"],
    ),
    PairDef(
        pair_id="pair_02",
        adapter_a=SOURCE_ADAPTERS[0],  # metamath-r16
        adapter_b=SOURCE_ADAPTERS[3],  # magicoder-r16
        expected_verdict="moderate",
        eval_sets=["gsm8k_math", "mbpp_code"],
    ),
    PairDef(
        pair_id="pair_03",
        adapter_a=SOURCE_ADAPTERS[3],  # magicoder-r16
        adapter_b=SOURCE_ADAPTERS[4],  # btgenbot-r8
        expected_verdict="conflicting",
        eval_sets=["mbpp_code", "oasst2_chat"],
    ),
    PairDef(
        pair_id="pair_04",
        adapter_a=SOURCE_ADAPTERS[2],  # openwebmath-r64
        adapter_b=SOURCE_ADAPTERS[4],  # btgenbot-r8
        expected_verdict="imbalanced",
        eval_sets=["gsm8k_math", "oasst2_chat"],
    ),
    PairDef(
        pair_id="pair_06",
        adapter_a=SOURCE_ADAPTERS[5],  # catsubcat-r16
        adapter_b=SOURCE_ADAPTERS[4],  # btgenbot-r8
        expected_verdict="mixed",
        # Same-domain: both chat.  Both eval on oasst2 but as two rows.
        eval_sets=["oasst2_chat", "oasst2_chat"],
    ),
]

# Which pairs are magnitude-imbalanced (get the norm_equalized condition)
IMBALANCED_PAIRS = {"pair_01", "pair_03", "pair_04", "pair_06"}


# ═══════════════════════════════════════════════════════════════════════════
# Dataset formatting
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FormattedExample:
    """A tokenization-ready example with prompt/completion split."""

    full_text: str  # entire text for tokenization
    prompt_text: str  # prompt portion (masked from loss)
    completion_text: str  # completion portion (scored)
    source_id: str | None  # dataset row id for traceability


def format_gsm8k(example: dict) -> FormattedExample | None:
    """GSM8K: score the answer conditioned on the question."""
    q = example.get("question", "")
    a = example.get("answer", "")
    if not q or not a:
        return None
    prompt = f"Question: {q}\nAnswer:"
    completion = f" {a}"
    return FormattedExample(
        full_text=prompt + completion,
        prompt_text=prompt,
        completion_text=completion,
        source_id=None,
    )


def format_mbpp(example: dict) -> FormattedExample | None:
    """MBPP: score the code conditioned on the description."""
    desc = example.get("text", example.get("prompt", ""))
    code = example.get("code", example.get("canonical_solution", ""))
    if not desc or not code:
        return None
    prompt = f"{desc}\n\n"
    completion = code
    return FormattedExample(
        full_text=prompt + completion,
        prompt_text=prompt,
        completion_text=completion,
        source_id=str(example.get("task_id")),
    )


def format_oasst2(example: dict) -> FormattedExample | None:
    """OpenAssistant oasst2: score full text (no clear prompt/completion split
    in the flat validation set, so we score all tokens)."""
    text = example.get("text", "")
    if not text or len(text.strip()) < 30:
        return None
    return FormattedExample(
        full_text=text,
        prompt_text="",  # no masking — score all tokens
        completion_text=text,
        source_id=example.get("message_id"),
    )


FORMATTERS = {
    "gsm8k_math": format_gsm8k,
    "mbpp_code": format_mbpp,
    "oasst2_chat": format_oasst2,
}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Download + Merge  (CPU)
# ═══════════════════════════════════════════════════════════════════════════


def download_adapter(repo_id: str, cache_dir: Path) -> Path:
    """Download a PEFT adapter from HuggingFace Hub.  Returns local path."""
    from huggingface_hub import snapshot_download

    local_dir = cache_dir / repo_id.replace("/", "--")

    config_candidates = list(local_dir.glob("adapter_config*"))
    weight_candidates = list(local_dir.glob("adapter_model*"))
    if config_candidates and weight_candidates:
        logger.info("  Already cached: %s", local_dir)
        return local_dir

    logger.info("  Downloading %s ...", repo_id)
    snapshot_download(
        repo_id,
        local_dir=str(local_dir),
        allow_patterns=[
            "adapter_config*",
            "adapter_model*",
            "pytorch_model.bin",
        ],
        ignore_patterns=[
            "*.md",
            "*.txt",
            ".gitattributes",
            "tokenizer*",
            "special_tokens*",
            "training_args*",
        ],
    )
    return local_dir


def merged_adapter_has_weights(adapter_dir: Path) -> bool:
    """Check whether a merged adapter directory contains actual weight files."""
    if not adapter_dir.exists():
        return False
    safetensors = list(adapter_dir.glob("*.safetensors"))
    bins = list(adapter_dir.glob("*.bin"))
    return bool(safetensors or bins)


def compute_global_frob_coefficients(
    dir_a: Path,
    dir_b: Path,
) -> tuple[float, float]:
    """Compute global Frobenius-proportional merge coefficients.

    Returns (coeff_a, coeff_b) such that the weaker adapter gets
    proportionally more weight, counteracting the magnitude gap.
    This is the simplest possible scale correction — no per-layer
    analysis, just global rebalancing.
    """
    import torch

    from gradience.vnext.merge.io import extract_factors, load_adapter

    info_a = load_adapter(dir_a)
    info_b = load_adapter(dir_b)

    frob_sq_a = 0.0
    frob_sq_b = 0.0

    for prefix in info_a.module_prefixes:
        try:
            A, B, r = extract_factors(info_a, prefix)
            dW = (info_a.alpha / r) * B @ A
            frob_sq_a += torch.norm(dW, p="fro").item() ** 2
        except (KeyError, RuntimeError):
            continue

    for prefix in info_b.module_prefixes:
        try:
            A, B, r = extract_factors(info_b, prefix)
            dW = (info_b.alpha / r) * B @ A
            frob_sq_b += torch.norm(dW, p="fro").item() ** 2
        except (KeyError, RuntimeError):
            continue

    frob_a = frob_sq_a**0.5
    frob_b = frob_sq_b**0.5
    total = frob_a + frob_b

    if total < 1e-12:
        return (0.5, 0.5)

    # Inverse-proportional: weaker adapter gets more weight
    coeff_a = frob_b / total
    coeff_b = frob_a / total

    return (coeff_a, coeff_b)


def run_merges(
    pairs: list[PairDef],
    cache_dir: Path,
    output_dir: Path,
    verbose: bool = False,
) -> dict[str, dict[str, Path]]:
    """Execute naive, norm_equalized, and recommended merges for each pair.

    Returns mapping: pair_id → {"naive": Path, "norm_equalized": Path,
                                 "recommended": Path}
    """
    from gradience.vnext.merge import (
        execute_merge,
        merge_audit,
        plan_from_audit,
        recommend_merge,
    )

    merge_dir = output_dir / "merged_adapters"
    merge_dir.mkdir(parents=True, exist_ok=True)

    merged_paths: dict[str, dict[str, Path]] = {}

    for pair in pairs:
        pair_label = f"{pair.adapter_a.label} × {pair.adapter_b.label}"
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Merging: {pair_label}")
            print(f"{'=' * 60}")

        # Download
        dir_a = download_adapter(pair.adapter_a.repo, cache_dir)
        dir_b = download_adapter(pair.adapter_b.repo, cache_dir)

        # Audit
        audit_dir = merge_dir / pair.pair_id / "audit"
        try:
            report = merge_audit(
                str(dir_a),
                str(dir_b),
                output_dir=str(audit_dir),
                verbose=verbose,
            )
        except Exception as e:
            logger.error("Audit failed for %s: %s", pair.pair_id, e)
            continue

        rank_a = report.adapter_a.get("rank", 16)
        rank_b = report.adapter_b.get("rank", 16)
        output_rank = max(rank_a, rank_b)
        output_alpha = report.adapter_a.get("alpha", 32.0)

        # Determine which conditions to run
        conditions = [
            ("naive", "uniform_linear", {}),
            ("recommended", "audit_aware", {}),
        ]

        # Add norm-equalized condition for imbalanced pairs
        if pair.pair_id in IMBALANCED_PAIRS:
            try:
                coeff_a, coeff_b = compute_global_frob_coefficients(dir_a, dir_b)
                conditions.insert(
                    1,
                    (
                        "norm_equalized",
                        "uniform_linear",
                        {"coeff_a": coeff_a, "coeff_b": coeff_b},
                    ),
                )
                if verbose:
                    print(f"  Norm-equalized coefficients: A={coeff_a:.4f}, B={coeff_b:.4f}")
            except Exception as e:
                logger.warning("Could not compute norm-eq coefficients for %s: %s", pair.pair_id, e)

        pair_paths: dict[str, Path] = {}

        # Pre-load adapters once for all merge conditions in this pair
        # (avoids 2-3 redundant load_adapter calls per condition)
        from gradience.vnext.merge.io import load_adapter as _load_adapter

        needs_merge = any(not merged_adapter_has_weights(merge_dir / pair.pair_id / c[0]) for c in conditions)
        if needs_merge:
            if verbose:
                print("  Pre-loading adapters for merge...")
            _info_a = _load_adapter(dir_a)
            _info_b = _load_adapter(dir_b)
        else:
            _info_a = _info_b = None

        for cond_name, strategy_name, extra_kwargs in conditions:
            cond_dir = merge_dir / pair.pair_id / cond_name
            cond_dir.mkdir(parents=True, exist_ok=True)

            if merged_adapter_has_weights(cond_dir):
                if verbose:
                    print(f"  {cond_name}: weights exist, skipping merge")
                pair_paths[cond_name] = cond_dir
                continue

            if verbose:
                print(f"  {cond_name} ({strategy_name}) ...")

            try:
                t0 = time.time()

                plan_kwargs = dict(
                    output_rank=output_rank,
                    output_alpha=output_alpha,
                )
                # For norm_equalized, pass custom coefficients to uniform_linear
                # plan_uniform_linear expects a single 'coefficients' tuple
                if extra_kwargs.get("coeff_a") is not None:
                    plan_kwargs["coefficients"] = (
                        extra_kwargs["coeff_a"],
                        extra_kwargs["coeff_b"],
                    )

                plan = plan_from_audit(
                    strategy_name,
                    report,
                    str(dir_a),
                    str(dir_b),
                    **plan_kwargs,
                )

                merge_result = execute_merge(
                    plan,
                    str(cond_dir),
                    verbose=verbose,
                    preloaded_a=_info_a,
                    preloaded_b=_info_b,
                )
                dt = time.time() - t0

                if verbose:
                    print(f"    Done in {dt:.1f}s  (recon error: {merge_result.mean_reconstruction_error:.4f})")

                merge_result.to_json(cond_dir / "merge_result.json")
                pair_paths[cond_name] = cond_dir

            except Exception as e:
                logger.error("Merge %s/%s failed: %s", pair.pair_id, cond_name, e)

        # Free pre-loaded adapters
        del _info_a, _info_b

        # Require at least naive + recommended
        if "naive" in pair_paths and "recommended" in pair_paths:
            merged_paths[pair.pair_id] = pair_paths
        else:
            logger.warning("Incomplete merges for %s — got %s", pair.pair_id, list(pair_paths.keys()))

    return merged_paths


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Perplexity evaluation  (GPU)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EvalResult:
    """Perplexity evaluation result for a single adapter × eval_set."""

    adapter_label: str
    adapter_type: str  # "base" | "source" | "naive" | "norm_equalized" | "recommended"
    pair_id: str | None  # None for base/source
    eval_set_id: str  # which eval set was used
    eval_task: str  # domain: math | code | chat
    # Primary metrics (token-weighted)
    perplexity: float  # exp(total_nll / total_scored_tokens)
    token_mean_loss: float  # total_nll / total_scored_tokens
    # Secondary metrics
    example_mean_loss: float  # mean of per-example losses (for comparison)
    example_std_loss: float  # std of per-example losses
    # Counts
    n_examples: int
    total_tokens: int  # total tokens seen (prompt + completion)
    scored_tokens: int  # tokens included in loss (completion only)
    truncated_examples: int  # how many examples hit max_length
    # Per-example detail (saved in JSON, not printed)
    per_example_losses: list[float] = field(default_factory=list)
    per_example_tokens: list[int] = field(default_factory=list)
    per_example_scored: list[int] = field(default_factory=list)
    eval_time_s: float = 0.0


def load_base_model(model_name: str, dtype=None):
    """Load base model and tokenizer.  Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype is None:
        dtype = torch.float16

    print(f"Loading base model: {model_name} ({dtype}) ...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def load_eval_dataset(
    eval_set: EvalSetDef,
    max_examples: int,
) -> list[FormattedExample]:
    """Load, format, and deterministically sample an eval dataset."""
    from datasets import load_dataset

    formatter = FORMATTERS.get(eval_set.eval_id)
    if formatter is None:
        raise ValueError(f"No formatter registered for eval set '{eval_set.eval_id}'")

    print(f"  Loading eval data: {eval_set.description} ...")

    kwargs = {"split": eval_set.split}
    if eval_set.subset:
        ds = load_dataset(eval_set.dataset, eval_set.subset, **kwargs)
    else:
        ds = load_dataset(eval_set.dataset, **kwargs)

    # Format all valid examples first, then sample deterministically
    all_formatted = []
    for example in ds:
        fmt = formatter(example)
        if fmt is not None and len(fmt.full_text.strip()) > 30:
            all_formatted.append(fmt)

    # Deterministic shuffle and slice
    rng = random.Random(SEED)
    rng.shuffle(all_formatted)
    selected = all_formatted[:max_examples]

    print(f"    {len(selected)} examples selected (from {len(all_formatted)} valid, seed={SEED})")

    # Preview first 2 examples for sanity
    for i, ex in enumerate(selected[:2]):
        prompt_preview = ex.prompt_text[:80].replace("\n", "\\n")
        comp_preview = ex.completion_text[:80].replace("\n", "\\n")
        print(f"    Example {i}: prompt='{prompt_preview}...' completion='{comp_preview}...'")

    return selected


def evaluate_perplexity(
    model,
    tokenizer,
    examples: list[FormattedExample],
    max_length: int = 512,
) -> dict[str, Any]:
    """Compute token-weighted perplexity with completion-only scoring.

    For each example:
    1. Tokenize full_text (prompt + completion)
    2. Determine how many tokens belong to prompt vs completion
    3. Create labels that mask prompt tokens (set to -100)
    4. Compute cross-entropy only on completion tokens
    5. Accumulate total NLL and total scored tokens

    Perplexity = exp(total_NLL / total_scored_tokens)
    """
    import torch

    per_example_losses = []
    per_example_tokens = []
    per_example_scored = []

    total_nll = 0.0
    total_scored = 0
    total_tokens = 0
    n_truncated = 0

    for ex in examples:
        # Tokenize full text
        full_enc = tokenizer(
            ex.full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        input_ids = full_enc["input_ids"]
        seq_len = input_ids.shape[1]

        if seq_len >= max_length:
            n_truncated += 1

        total_tokens += seq_len

        # Determine prompt length in tokens for masking
        if ex.prompt_text:
            prompt_enc = tokenizer(
                ex.prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            prompt_len = prompt_enc["input_ids"].shape[1]
        else:
            prompt_len = 0

        # Build labels: -100 for prompt tokens, actual ids for completion
        labels = input_ids.clone()
        if prompt_len > 0:
            labels[0, :prompt_len] = -100

        # For causal LM, the model internally shifts labels left by 1,
        # so we also need to mask the first completion token if it's
        # the one predicted from the last prompt token.  HuggingFace
        # handles this shift internally.

        n_scored = max(seq_len - prompt_len, 0)
        if n_scored == 0:
            # Nothing to score — skip
            continue

        with torch.no_grad():
            outputs = model(**full_enc, labels=labels)

        # outputs.loss is already mean over non-(-100) tokens
        example_loss = outputs.loss.item()
        example_nll = example_loss * n_scored  # recover total NLL

        per_example_losses.append(example_loss)
        per_example_tokens.append(seq_len)
        per_example_scored.append(n_scored)

        total_nll += example_nll
        total_scored += n_scored

    # Token-weighted perplexity
    if total_scored > 0:
        token_mean_loss = total_nll / total_scored
        perplexity = float(np.exp(token_mean_loss))
    else:
        token_mean_loss = float("nan")
        perplexity = float("nan")

    # Example-averaged loss (secondary, for comparison)
    losses_arr = np.array(per_example_losses) if per_example_losses else np.array([])
    example_mean_loss = float(losses_arr.mean()) if len(losses_arr) > 0 else float("nan")
    example_std_loss = float(losses_arr.std()) if len(losses_arr) > 0 else float("nan")

    return {
        "perplexity": perplexity,
        "token_mean_loss": token_mean_loss,
        "example_mean_loss": example_mean_loss,
        "example_std_loss": example_std_loss,
        "n_examples": len(per_example_losses),
        "total_tokens": int(total_tokens),
        "scored_tokens": int(total_scored),
        "truncated_examples": n_truncated,
        "per_example_losses": per_example_losses,
        "per_example_tokens": per_example_tokens,
        "per_example_scored": per_example_scored,
    }


def evaluate_with_adapter(
    base_model_name: str,
    tokenizer,
    adapter_path: Path,
    examples: list[FormattedExample],
    max_length: int = 512,
    dtype=None,
) -> dict[str, Any]:
    """Load a fresh base model + PEFT adapter, evaluate, then discard.

    Uses clean reinstantiation rather than PEFT unload to avoid
    state contamination between adapter evaluations.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if dtype is None:
        dtype = torch.float16

    # Fresh base model load
    fresh_base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    fresh_base.eval()

    # Attach adapter
    peft_model = PeftModel.from_pretrained(fresh_base, str(adapter_path))
    peft_model.eval()

    result = evaluate_perplexity(peft_model, tokenizer, examples, max_length)

    # Clean teardown
    del peft_model
    del fresh_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation scheduler
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EvalJob:
    """A single evaluation to run."""

    adapter_label: str
    adapter_type: str  # "base" | "source" | "naive" | "norm_equalized" | "recommended"
    pair_id: str | None
    adapter_path: Path | None
    eval_set_id: str
    side: str | None = None  # "a" or "b" — which source adapter this eval relates to


def build_eval_schedule(
    pairs: list[PairDef],
    source_adapters: list[AdapterDef],
    merged_paths: dict[str, dict[str, Path]],
    source_adapter_dirs: dict[str, Path],
) -> dict[str, list[EvalJob]]:
    """Build evaluation schedule grouped by eval_set_id.

    Grouping by eval_set_id means the dataset loads once per unique eval set.
    """
    schedule: dict[str, list[EvalJob]] = {}

    def add_job(eval_set_id: str, job: EvalJob):
        if eval_set_id not in schedule:
            schedule[eval_set_id] = []
        schedule[eval_set_id].append(job)

    # 1. Base model — one eval per unique eval set
    for es_id in EVAL_SETS:
        add_job(
            es_id,
            EvalJob(
                adapter_label="base_model",
                adapter_type="base",
                pair_id=None,
                adapter_path=None,
                eval_set_id=es_id,
            ),
        )

    # 2. Source adapters — each on its primary eval set
    seen_source = set()
    for adapter in source_adapters:
        if adapter.label in seen_source:
            continue
        seen_source.add(adapter.label)

        if adapter.label not in source_adapter_dirs:
            continue

        # Find which eval sets this adapter maps to
        for pair in pairs:
            if pair.adapter_a.label == adapter.label:
                es_id = pair.eval_sets[0]  # adapter_a's eval set
            elif pair.adapter_b.label == adapter.label:
                es_id = pair.eval_sets[1]  # adapter_b's eval set
            else:
                continue

            # Deduplicate
            key = (adapter.label, es_id)
            if key in seen_source:
                continue
            seen_source.add(key)

            add_job(
                es_id,
                EvalJob(
                    adapter_label=adapter.label,
                    adapter_type="source",
                    pair_id=None,
                    adapter_path=source_adapter_dirs[adapter.label],
                    eval_set_id=es_id,
                ),
            )

    # 3. Merged adapters — each on both source adapters' eval sets
    for pair in pairs:
        if pair.pair_id not in merged_paths:
            continue
        for cond in ["naive", "norm_equalized", "recommended"]:
            if cond not in merged_paths[pair.pair_id]:
                continue
            adapter_path = merged_paths[pair.pair_id][cond]
            label = f"{pair.pair_id}_{cond}"

            # Eval on adapter_a's eval set
            add_job(
                pair.eval_sets[0],
                EvalJob(
                    adapter_label=label,
                    adapter_type=cond,
                    pair_id=pair.pair_id,
                    adapter_path=adapter_path,
                    eval_set_id=pair.eval_sets[0],
                    side="a",
                ),
            )
            # Eval on adapter_b's eval set (may be same for same-domain)
            add_job(
                pair.eval_sets[1],
                EvalJob(
                    adapter_label=label,
                    adapter_type=cond,
                    pair_id=pair.pair_id,
                    adapter_path=adapter_path,
                    eval_set_id=pair.eval_sets[1],
                    side="b",
                ),
            )

    return schedule


def run_evaluation(
    base_model,
    base_model_name: str,
    tokenizer,
    schedule: dict[str, list[EvalJob]],
    max_examples: int = 500,
    max_length: int = 512,
    verbose: bool = False,
    dtype=None,
) -> list[EvalResult]:
    """Execute all evaluations, grouped by eval set to minimise data reloading.

    Uses clean model reinstantiation for each adapter to prevent PEFT
    state contamination.
    """
    all_results: list[EvalResult] = []

    for es_id, jobs in schedule.items():
        if not jobs:
            continue

        eval_set = EVAL_SETS[es_id]
        unique_adapters = len(set(j.adapter_label for j in jobs))

        print(f"\n{'─' * 60}")
        print(f"  Eval set: {es_id}  ({eval_set.description})")
        print(f"  {len(jobs)} evaluations, {unique_adapters} unique adapters")
        print(f"{'─' * 60}")

        # Load eval data once per eval set
        examples = load_eval_dataset(eval_set, max_examples)

        # Deduplicate: if same adapter appears twice for same-domain pair,
        # compute once and duplicate the result
        seen_results: dict[str, dict[str, Any]] = {}

        for i, job in enumerate(jobs):
            label = job.adapter_label
            atype = job.adapter_type

            if verbose:
                print(f"\n  [{i + 1}/{len(jobs)}] {label} ({atype}) on {es_id}")

            # Check if we already computed this exact adapter × eval_set
            cache_key = f"{label}_{es_id}"
            if cache_key in seen_results:
                if verbose:
                    print("    (reusing cached result)")
                metrics = seen_results[cache_key]
            else:
                t0 = time.time()
                try:
                    if atype == "base":
                        metrics = evaluate_perplexity(
                            base_model,
                            tokenizer,
                            examples,
                            max_length,
                        )
                    else:
                        # Clean reinstantiation for each adapter
                        metrics = evaluate_with_adapter(
                            base_model_name,
                            tokenizer,
                            job.adapter_path,
                            examples,
                            max_length,
                            dtype=dtype,
                        )
                    metrics["eval_time_s"] = round(time.time() - t0, 2)
                    seen_results[cache_key] = metrics

                except Exception as e:
                    logger.error("Eval failed for %s on %s: %s", label, es_id, e, exc_info=True)
                    metrics = {
                        "perplexity": float("nan"),
                        "token_mean_loss": float("nan"),
                        "example_mean_loss": float("nan"),
                        "example_std_loss": float("nan"),
                        "n_examples": 0,
                        "total_tokens": 0,
                        "scored_tokens": 0,
                        "truncated_examples": 0,
                        "per_example_losses": [],
                        "per_example_tokens": [],
                        "per_example_scored": [],
                        "eval_time_s": 0.0,
                    }

            if verbose and not np.isnan(metrics["perplexity"]):
                print(
                    f"    PPL: {metrics['perplexity']:.2f}  "
                    f"(token_loss: {metrics['token_mean_loss']:.4f}, "
                    f"scored: {metrics['scored_tokens']} tokens, "
                    f"truncated: {metrics['truncated_examples']})"
                )

            result = EvalResult(
                adapter_label=label,
                adapter_type=atype,
                pair_id=job.pair_id,
                eval_set_id=es_id,
                eval_task=eval_set.task,
                perplexity=metrics["perplexity"],
                token_mean_loss=metrics["token_mean_loss"],
                example_mean_loss=metrics["example_mean_loss"],
                example_std_loss=metrics["example_std_loss"],
                n_examples=metrics["n_examples"],
                total_tokens=metrics["total_tokens"],
                scored_tokens=metrics["scored_tokens"],
                truncated_examples=metrics["truncated_examples"],
                per_example_losses=metrics["per_example_losses"],
                per_example_tokens=metrics["per_example_tokens"],
                per_example_scored=metrics["per_example_scored"],
                eval_time_s=metrics.get("eval_time_s", 0.0),
            )
            all_results.append(result)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════


def save_results(
    results: list[EvalResult],
    output_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    """Save evaluation results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "perplexity_results.json"

    # For JSON: include per-example detail
    payload = {
        "metadata": metadata,
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return output_path


def print_summary(results: list[EvalResult]) -> None:
    """Print a human-readable summary table."""
    print(f"\n{'=' * 100}")
    print("  PERPLEXITY RESULTS (token-weighted)")
    print(f"{'=' * 100}")
    print(
        f"  {'Adapter':<35s}  {'Type':<14s}  {'EvalSet':<12s}  "
        f"{'PPL':>8s}  {'TokLoss':>8s}  {'Scored':>7s}  {'Trunc':>5s}"
    )
    print(f"  {'─' * 96}")

    for r in sorted(results, key=lambda x: (x.eval_set_id, x.adapter_type, x.adapter_label)):
        ppl_str = f"{r.perplexity:.2f}" if not np.isnan(r.perplexity) else "FAILED"
        loss_str = f"{r.token_mean_loss:.4f}" if not np.isnan(r.token_mean_loss) else "—"
        print(
            f"  {r.adapter_label:<35s}  {r.adapter_type:<14s}  {r.eval_set_id:<12s}  "
            f"{ppl_str:>8s}  {loss_str:>8s}  {r.scored_tokens:>7d}  "
            f"{r.truncated_examples:>5d}"
        )

    print(f"{'=' * 100}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Study 16 Perplexity Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Base model name or path (default: meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/study16_perplexity"),
        help="Directory for all output files",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "gradience" / "adapters",
        help="Cache directory for downloaded adapters",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=500,
        help="Max examples per eval set (default: 500)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max token length per example (default: 512)",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        nargs="*",
        default=None,
        help="Specific pair IDs to run (default: all)",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merge phase (assumes merged adapters already have weights)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU dry-run: test imports and paths, skip GPU evaluation",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Filter pairs
    pairs_to_run = PAIRS
    if args.pairs:
        pairs_to_run = [p for p in PAIRS if p.pair_id in args.pairs]
        if not pairs_to_run:
            print(f"No matching pairs. Available: {[p.pair_id for p in PAIRS]}")
            sys.exit(1)

    # Determine which source adapters are needed
    needed_sources = set()
    for pair in pairs_to_run:
        needed_sources.add(pair.adapter_a.label)
        needed_sources.add(pair.adapter_b.label)
    source_adapters_to_eval = [a for a in SOURCE_ADAPTERS if a.label in needed_sources]

    print("Study 16 — Perplexity Validation")
    print(f"  Base model:     {args.base_model}")
    print(f"  Pairs:          {len(pairs_to_run)}")
    print(f"  Source adapters: {len(source_adapters_to_eval)}")
    print(f"  Max examples:   {args.max_examples}")
    print(f"  Output:         {args.output_dir}")
    print(f"  Seed:           {SEED}")
    print()

    t_start = time.time()

    # ── Phase 1a: Download source adapters ─────────────────────────────
    print("Phase 1a: Downloading source adapters ...")
    source_adapter_dirs: dict[str, Path] = {}
    for adapter in source_adapters_to_eval:
        try:
            local_dir = download_adapter(adapter.repo, args.cache_dir)
            source_adapter_dirs[adapter.label] = local_dir
        except Exception as e:
            logger.error("Failed to download %s: %s", adapter.label, e)

    print(f"  {len(source_adapter_dirs)} source adapters ready\n")

    # ── Phase 1b: Merge (or skip) ─────────────────────────────────────
    if args.skip_merge:
        print("Phase 1b: Skipping merge (--skip-merge)")
        merge_base = args.output_dir / "merged_adapters"
        merged_paths: dict[str, dict[str, Path]] = {}
        for pair in pairs_to_run:
            pair_dir = merge_base / pair.pair_id
            pair_merged: dict[str, Path] = {}
            for cond in ["naive", "norm_equalized", "recommended"]:
                cond_dir = pair_dir / cond
                if merged_adapter_has_weights(cond_dir):
                    pair_merged[cond] = cond_dir
            if "naive" in pair_merged and "recommended" in pair_merged:
                merged_paths[pair.pair_id] = pair_merged
            else:
                logger.warning("Missing weights for %s — will skip", pair.pair_id)
    else:
        print("Phase 1b: Executing merges ...")
        merged_paths = run_merges(
            pairs_to_run,
            args.cache_dir,
            args.output_dir,
            verbose=args.verbose,
        )

    print(f"\n  {len(merged_paths)} pairs merged successfully\n")

    # ── Dry-run exit ───────────────────────────────────────────────────
    if args.dry_run:
        print("Dry-run complete. Merge phase done. Skipping GPU evaluation.")
        schedule = build_eval_schedule(
            pairs_to_run,
            source_adapters_to_eval,
            merged_paths,
            source_adapter_dirs,
        )
        total_evals = sum(len(jobs) for jobs in schedule.values())
        print(f"  Would run {total_evals} evaluations:")
        for es_id, jobs in schedule.items():
            print(f"    {es_id}: {len(jobs)} evals")
            for job in jobs:
                print(f"      - {job.adapter_label} ({job.adapter_type})")
        sys.exit(0)

    # ── Phase 2: GPU evaluation ────────────────────────────────────────
    print("Phase 2: GPU-side perplexity evaluation")
    print()

    import torch

    dtype = torch.float16

    base_model, tokenizer = load_base_model(args.base_model, dtype=dtype)

    schedule = build_eval_schedule(
        pairs_to_run,
        source_adapters_to_eval,
        merged_paths,
        source_adapter_dirs,
    )
    total_evals = sum(len(jobs) for jobs in schedule.values())
    unique_evals = len(set(f"{j.adapter_label}_{es}" for es, jobs in schedule.items() for j in jobs))
    print(f"\n  Total eval jobs: {total_evals}  (unique adapter×set: {unique_evals})")

    results = run_evaluation(
        base_model,
        args.base_model,
        tokenizer,
        schedule,
        max_examples=args.max_examples,
        max_length=args.max_length,
        verbose=args.verbose,
        dtype=dtype,
    )

    # ── Output ─────────────────────────────────────────────────────────
    total_time = time.time() - t_start

    metadata = {
        "study_id": "study16_perplexity_validation",
        "date": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "max_examples": args.max_examples,
        "max_length": args.max_length,
        "seed": SEED,
        "n_pairs": len(pairs_to_run),
        "n_evaluations": len(results),
        "total_time_s": round(total_time, 2),
        "merge_conditions": ["naive", "norm_equalized", "recommended"],
        "scoring": "completion-only token-weighted NLL",
    }

    output_path = save_results(results, args.output_dir, metadata)
    print_summary(results)
    print(f"Results saved to {output_path}")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")


if __name__ == "__main__":
    main()
