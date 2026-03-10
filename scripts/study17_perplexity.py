#!/usr/bin/env python3
"""
Study 17: Compress-Then-Merge — Phase 2 (GPU perplexity evaluation).

Re-executes merges for all 5 conditions (since Phase 1 deletes .safetensors
to save disk) and evaluates each merged adapter plus source adapters and
base model on held-out data.

Merge conditions (re-merged here):
    A. Full-rank naive          — 0.5/0.5 uniform_linear
    B. Full-rank norm-equalized — global Frobenius-aware reweighting
    C. Full-rank recommended    — audit_aware
    D. Compress + norm-equalized
    E. Compress + recommended

Evaluation scoring matches Study 16:
    - Token-weighted NLL (completion-only where applicable)
    - Prompt masking
    - Fixed eval subsets (deterministic seed)
    - Per-example loss logging for tail inspection

Usage:
    # Full run (re-merge + GPU eval):
    python scripts/study17_perplexity.py \\
        --base-model meta-llama/Llama-2-7b-hf \\
        --output-dir results/study17 \\
        --cache-dir ~/.cache/gradience/adapters \\
        --max-examples 500 \\
        --verbose

    # Skip merge phase if adapter dirs already have weights:
    python scripts/study17_perplexity.py \\
        --base-model meta-llama/Llama-2-7b-hf \\
        --output-dir results/study17 \\
        --skip-merge \\
        --max-examples 500

    # CPU dry-run:
    python scripts/study17_perplexity.py \\
        --base-model meta-llama/Llama-2-7b-hf \\
        --output-dir results/study17 \\
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
from typing import Any

import numpy as np

logger = logging.getLogger("study17_ppl")

SEED = 42
ENERGY_THRESHOLD = 0.95


# ═══════════════════════════════════════════════════════════════════════════
# Adapter and pair definitions (mirrors study17_compress_merge.py)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AdapterDef:
    """A single adapter with its HF repo, label, and task domain."""

    repo: str
    label: str
    task: str  # math | code | chat


@dataclass
class EvalSetDef:
    """An evaluation set tied to a task domain."""

    eval_id: str
    task: str
    dataset: str
    subset: str | None
    split: str
    description: str


@dataclass
class PairDef:
    """An adapter pair to merge and evaluate."""

    pair_id: str
    adapter_a: AdapterDef
    adapter_b: AdapterDef
    expected_verdict: str
    is_boundary: bool = False
    eval_sets: list[str] = field(default_factory=list)


# ─── Source adapters ────────────────────────────────────────────────────

SOURCE_ADAPTERS: list[AdapterDef] = [
    AdapterDef("LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32", "metamath-r16", "math"),
    AdapterDef("LoRA-TMLR-2024/openwebmath-lora-rank-16-20B-tokens", "openwebmath-r16", "math"),
    AdapterDef("LoRA-TMLR-2024/openwebmath-lora-rank-64-20B-tokens", "openwebmath-r64", "math"),
    AdapterDef("LoRA-TMLR-2024/magicoder-lora-rank-16-alpha-32", "magicoder-r16", "code"),
    AdapterDef("AIRLab-POLIMI/llama-2-7b-chat-hf-btgenbot-adapter", "btgenbot-r8", "chat"),
    AdapterDef("shivanikerai/Llama-2-7b-chat-hf-adapter-cat-subcat-mapping-v2.0", "catsubcat-r16", "chat"),
]

# ─── Eval sets ──────────────────────────────────────────────────────────

EVAL_SETS: dict[str, EvalSetDef] = {
    "gsm8k_math": EvalSetDef(
        eval_id="gsm8k_math",
        task="math",
        dataset="gsm8k",
        subset="main",
        split="test",
        description="GSM8K test (math reasoning)",
    ),
    "mbpp_code": EvalSetDef(
        eval_id="mbpp_code",
        task="code",
        dataset="mbpp",
        subset="full",
        split="test",
        description="MBPP test (code generation)",
    ),
    "oasst2_chat": EvalSetDef(
        eval_id="oasst2_chat",
        task="chat",
        dataset="OpenAssistant/oasst2",
        subset=None,
        split="validation",
        description="OpenAssistant oasst2 validation (chat/instruction)",
    ),
}

# ─── Pairs ──────────────────────────────────────────────────────────────

PAIRS: list[PairDef] = [
    PairDef(
        pair_id="pair_01",
        adapter_a=SOURCE_ADAPTERS[0],  # metamath-r16
        adapter_b=SOURCE_ADAPTERS[1],  # openwebmath-r16
        expected_verdict="redundant",
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
        is_boundary=True,
        eval_sets=["oasst2_chat", "oasst2_chat"],
    ),
]

# Condition names that map to strategy + source type
CONDITION_DEFS = [
    ("A_fullrank_naive", "uniform_linear", False, {}),
    ("B_fullrank_normeq", "uniform_linear", False, {"use_normeq": True}),
    ("C_fullrank_recommended", "audit_aware", False, {}),
    ("D_compress_normeq", "uniform_linear", True, {"use_normeq": True}),
    ("E_compress_recommended", "audit_aware", True, {}),
]


# ═══════════════════════════════════════════════════════════════════════════
# Dataset formatting (identical to study16)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FormattedExample:
    """A tokenization-ready example with prompt/completion split."""

    full_text: str
    prompt_text: str
    completion_text: str
    source_id: str | None


def format_gsm8k(example: dict) -> FormattedExample | None:
    q = example.get("question", "")
    a = example.get("answer", "")
    if not q or not a:
        return None
    prompt = f"Question: {q}\nAnswer:"
    completion = f" {a}"
    return FormattedExample(full_text=prompt + completion, prompt_text=prompt, completion_text=completion, source_id=None)


def format_mbpp(example: dict) -> FormattedExample | None:
    desc = example.get("text", example.get("prompt", ""))
    code = example.get("code", example.get("canonical_solution", ""))
    if not desc or not code:
        return None
    prompt = f"{desc}\n\n"
    return FormattedExample(full_text=prompt + code, prompt_text=prompt, completion_text=code, source_id=str(example.get("task_id")))


def format_oasst2(example: dict) -> FormattedExample | None:
    text = example.get("text", "")
    if not text or len(text.strip()) < 30:
        return None
    return FormattedExample(full_text=text, prompt_text="", completion_text=text, source_id=example.get("message_id"))


FORMATTERS = {
    "gsm8k_math": format_gsm8k,
    "mbpp_code": format_mbpp,
    "oasst2_chat": format_oasst2,
}


# ═══════════════════════════════════════════════════════════════════════════
# Download + merge helpers
# ═══════════════════════════════════════════════════════════════════════════


def download_adapter(repo_id: str, cache_dir: Path) -> Path:
    """Download a PEFT adapter from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    local_dir = cache_dir / repo_id.replace("/", "--")

    if list(local_dir.glob("adapter_config*")) and list(local_dir.glob("adapter_model*")):
        logger.info("  Already cached: %s", local_dir)
        return local_dir

    logger.info("  Downloading %s ...", repo_id)
    snapshot_download(
        repo_id,
        local_dir=str(local_dir),
        allow_patterns=["adapter_config*", "adapter_model*", "pytorch_model.bin"],
        ignore_patterns=["*.md", "*.txt", ".gitattributes", "tokenizer*", "special_tokens*", "training_args*"],
    )
    return local_dir


def merged_adapter_has_weights(adapter_dir: Path) -> bool:
    """Check whether a merged adapter directory contains weight files."""
    if not adapter_dir.exists():
        return False
    return bool(list(adapter_dir.glob("*.safetensors")) or list(adapter_dir.glob("*.bin")))


def compute_global_frob_coefficients(dir_a: Path, dir_b: Path) -> tuple[float, float]:
    """Compute global Frobenius-proportional merge coefficients."""
    import torch

    from gradience.vnext.merge.io import extract_factors, load_adapter

    info_a = load_adapter(dir_a)
    info_b = load_adapter(dir_b)

    frob_sq_a = frob_sq_b = 0.0

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

    return (frob_b / total, frob_a / total)


def compute_effective_rank_at_threshold(peft_dir: Path, threshold: float = 0.95) -> int:
    """Compute the smallest global rank such that every layer retains >=threshold energy.

    Returns max(k_i) across all LoRA layers.
    """
    import torch

    from gradience.vnext.audit.lora_audit import find_peft_files, load_adapter_state_dict
    from gradience.vnext.svd_truncate import _parse_and_pair_lora_matrices

    config_path, weights_path, issues = find_peft_files(peft_dir)
    if config_path is None or weights_path is None:
        raise FileNotFoundError(f"Could not find PEFT files in {peft_dir}. Issues: {issues}")

    state_dict = load_adapter_state_dict(weights_path)
    pairs = _parse_and_pair_lora_matrices(state_dict)

    max_effective = 1
    for _base_key, pair_data in pairs.items():
        _, lora_A = pair_data["A"]
        _, lora_B = pair_data["B"]

        A_f = lora_A.cpu().to(torch.float32)
        B_f = lora_B.cpu().to(torch.float32)

        Qb, Rb = torch.linalg.qr(B_f)
        Qa, Ra = torch.linalg.qr(A_f.T)
        M = Rb @ Ra.T
        S = torch.linalg.svdvals(M)

        total_energy = (S**2).sum().item()
        if total_energy < 1e-12:
            continue

        cumulative = torch.cumsum(S**2, dim=0) / total_energy
        above = (cumulative >= threshold).nonzero(as_tuple=True)[0]
        k = int(above[0].item()) + 1 if len(above) > 0 else int(S.shape[0])
        max_effective = max(max_effective, k)

    return max_effective


def compress_adapter_for_eval(source_dir: Path, out_dir: Path, threshold: float = ENERGY_THRESHOLD) -> Path:
    """Compress an adapter using energy-threshold SVD truncation.

    Returns path to the compressed adapter directory.
    """
    from gradience.vnext.audit.lora_audit import find_peft_files, load_peft_adapter_config
    from gradience.vnext.svd_truncate import svd_truncate_peft_dir

    config_path, _, _ = find_peft_files(source_dir)
    config = load_peft_adapter_config(config_path)
    nominal_rank = config.r

    target_rank = compute_effective_rank_at_threshold(source_dir, threshold)

    if target_rank >= nominal_rank:
        import shutil

        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(source_dir, out_dir)
        return out_dir

    svd_truncate_peft_dir(source_dir, out_dir, target_rank)
    return out_dir


def run_merges(
    pairs: list[PairDef],
    cache_dir: Path,
    output_dir: Path,
    verbose: bool = False,
) -> dict[str, dict[str, Path]]:
    """Execute all 5 merge conditions for each pair.

    Returns mapping: pair_id -> {condition_name: merged_adapter_path}
    """
    from gradience.vnext.merge import execute_merge, merge_audit, plan_from_audit

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

        # Audit original
        audit_dir = merge_dir / pair.pair_id / "audit"
        try:
            report = merge_audit(str(dir_a), str(dir_b), output_dir=str(audit_dir), verbose=verbose)
        except Exception as e:
            logger.error("Audit failed for %s: %s", pair.pair_id, e)
            continue

        rank_a = report.adapter_a.get("rank", 16)
        rank_b = report.adapter_b.get("rank", 16)
        output_rank = max(rank_a, rank_b)
        output_alpha = report.adapter_a.get("alpha", 32.0)

        # Compress adapters
        comp_dir = merge_dir / pair.pair_id / "compressed"
        comp_dir.mkdir(parents=True, exist_ok=True)

        try:
            compressed_a = compress_adapter_for_eval(dir_a, comp_dir / "a")
            compressed_b = compress_adapter_for_eval(dir_b, comp_dir / "b")
        except Exception as e:
            logger.error("Compression failed for %s: %s", pair.pair_id, e)
            continue

        # Audit compressed
        audit_comp_dir = merge_dir / pair.pair_id / "audit_compressed"
        try:
            compressed_report = merge_audit(str(compressed_a), str(compressed_b), output_dir=str(audit_comp_dir), verbose=verbose)
        except Exception as e:
            logger.warning("Compressed audit failed for %s: %s", pair.pair_id, e)
            compressed_report = None

        # Compute norm-equalized coefficients
        try:
            coeff_a_orig, coeff_b_orig = compute_global_frob_coefficients(dir_a, dir_b)
        except Exception:
            coeff_a_orig, coeff_b_orig = 0.5, 0.5

        try:
            coeff_a_comp, coeff_b_comp = compute_global_frob_coefficients(compressed_a, compressed_b)
        except Exception:
            coeff_a_comp, coeff_b_comp = 0.5, 0.5

        # Determine compressed output rank
        from gradience.vnext.audit.lora_audit import find_peft_files, load_peft_adapter_config

        comp_rank_a = load_peft_adapter_config(find_peft_files(compressed_a)[0]).r
        comp_rank_b = load_peft_adapter_config(find_peft_files(compressed_b)[0]).r
        comp_output_rank = max(comp_rank_a, comp_rank_b)

        pair_paths: dict[str, Path] = {}

        for cond_name, strategy, is_compressed, extra in CONDITION_DEFS:
            cond_dir = merge_dir / pair.pair_id / cond_name

            if merged_adapter_has_weights(cond_dir):
                if verbose:
                    print(f"  {cond_name}: weights exist, skipping merge")
                pair_paths[cond_name] = cond_dir
                continue

            cond_dir.mkdir(parents=True, exist_ok=True)

            if verbose:
                print(f"  {cond_name} ({strategy}) ...")

            try:
                t0 = time.time()

                src_a = compressed_a if is_compressed else dir_a
                src_b = compressed_b if is_compressed else dir_b
                audit_rep = (compressed_report or report) if is_compressed else report
                o_rank = comp_output_rank if is_compressed else output_rank

                plan_kwargs: dict[str, Any] = {"output_rank": o_rank, "output_alpha": output_alpha}

                if extra.get("use_normeq"):
                    ca, cb = (coeff_a_comp, coeff_b_comp) if is_compressed else (coeff_a_orig, coeff_b_orig)
                    plan_kwargs["coefficients"] = (ca, cb)

                plan = plan_from_audit(strategy, audit_rep, str(src_a), str(src_b), **plan_kwargs)
                merge_result = execute_merge(plan, str(cond_dir), verbose=verbose)

                dt = time.time() - t0
                if verbose:
                    print(f"    Done in {dt:.1f}s  (recon: {merge_result.mean_reconstruction_error:.4f})")

                merge_result.to_json(cond_dir / "merge_result.json")
                pair_paths[cond_name] = cond_dir

            except Exception as e:
                logger.error("Merge %s/%s failed: %s", pair.pair_id, cond_name, e)

        # Require at least naive + recommended
        if "A_fullrank_naive" in pair_paths and "C_fullrank_recommended" in pair_paths:
            merged_paths[pair.pair_id] = pair_paths
        else:
            logger.warning("Incomplete merges for %s — got %s", pair.pair_id, list(pair_paths.keys()))

    return merged_paths


# ═══════════════════════════════════════════════════════════════════════════
# Perplexity evaluation (GPU)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EvalResult:
    """Perplexity evaluation result for a single adapter × eval_set."""

    adapter_label: str
    adapter_type: str  # "base" | "source" | condition name
    pair_id: str | None
    eval_set_id: str
    eval_task: str
    side: str | None  # "a" or "b"
    perplexity: float
    token_mean_loss: float
    example_mean_loss: float
    example_std_loss: float
    n_examples: int
    total_tokens: int
    scored_tokens: int
    truncated_examples: int
    per_example_losses: list[float] = field(default_factory=list)
    per_example_tokens: list[int] = field(default_factory=list)
    per_example_scored: list[int] = field(default_factory=list)
    eval_time_s: float = 0.0


def load_base_model(model_name: str, dtype=None):
    """Load base model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype is None:
        dtype = torch.float16

    print(f"Loading base model: {model_name} ({dtype}) ...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    model.eval()

    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def load_eval_dataset(eval_set: EvalSetDef, max_examples: int) -> list[FormattedExample]:
    """Load, format, and deterministically sample an eval dataset."""
    from datasets import load_dataset

    formatter = FORMATTERS.get(eval_set.eval_id)
    if formatter is None:
        raise ValueError(f"No formatter for eval set '{eval_set.eval_id}'")

    print(f"  Loading eval data: {eval_set.description} ...")

    kwargs: dict[str, Any] = {"split": eval_set.split}
    if eval_set.subset:
        ds = load_dataset(eval_set.dataset, eval_set.subset, **kwargs)
    else:
        ds = load_dataset(eval_set.dataset, **kwargs)

    all_formatted = []
    for example in ds:
        fmt = formatter(example)
        if fmt is not None and len(fmt.full_text.strip()) > 30:
            all_formatted.append(fmt)

    rng = random.Random(SEED)
    rng.shuffle(all_formatted)
    selected = all_formatted[:max_examples]

    print(f"    {len(selected)} examples selected (from {len(all_formatted)} valid, seed={SEED})")
    return selected


def evaluate_perplexity(model, tokenizer, examples: list[FormattedExample], max_length: int = 512) -> dict[str, Any]:
    """Compute token-weighted perplexity with completion-only scoring."""
    import torch

    per_example_losses = []
    per_example_tokens = []
    per_example_scored = []

    total_nll = 0.0
    total_scored = 0
    total_tokens = 0
    n_truncated = 0

    for ex in examples:
        full_enc = tokenizer(ex.full_text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        input_ids = full_enc["input_ids"]
        seq_len = input_ids.shape[1]

        if seq_len >= max_length:
            n_truncated += 1
        total_tokens += seq_len

        if ex.prompt_text:
            prompt_enc = tokenizer(ex.prompt_text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False)
            prompt_len = prompt_enc["input_ids"].shape[1]
        else:
            prompt_len = 0

        labels = input_ids.clone()
        if prompt_len > 0:
            labels[0, :prompt_len] = -100

        n_scored = max(seq_len - prompt_len, 0)
        if n_scored == 0:
            continue

        with torch.no_grad():
            outputs = model(**full_enc, labels=labels)

        example_loss = outputs.loss.item()
        example_nll = example_loss * n_scored

        per_example_losses.append(example_loss)
        per_example_tokens.append(seq_len)
        per_example_scored.append(n_scored)

        total_nll += example_nll
        total_scored += n_scored

    if total_scored > 0:
        token_mean_loss = total_nll / total_scored
        perplexity = float(np.exp(token_mean_loss))
    else:
        token_mean_loss = float("nan")
        perplexity = float("nan")

    losses_arr = np.array(per_example_losses) if per_example_losses else np.array([])

    return {
        "perplexity": perplexity,
        "token_mean_loss": token_mean_loss,
        "example_mean_loss": float(losses_arr.mean()) if len(losses_arr) > 0 else float("nan"),
        "example_std_loss": float(losses_arr.std()) if len(losses_arr) > 0 else float("nan"),
        "n_examples": len(per_example_losses),
        "total_tokens": int(total_tokens),
        "scored_tokens": int(total_scored),
        "truncated_examples": n_truncated,
        "per_example_losses": per_example_losses,
        "per_example_tokens": per_example_tokens,
        "per_example_scored": per_example_scored,
    }


def evaluate_with_adapter(
    base_model_name: str, tokenizer, adapter_path: Path, examples: list[FormattedExample],
    max_length: int = 512, dtype=None,
) -> dict[str, Any]:
    """Load fresh base + PEFT adapter, evaluate, discard."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if dtype is None:
        dtype = torch.float16

    fresh_base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype, device_map="auto")
    fresh_base.eval()

    peft_model = PeftModel.from_pretrained(fresh_base, str(adapter_path))
    peft_model.eval()

    result = evaluate_perplexity(peft_model, tokenizer, examples, max_length)

    del peft_model, fresh_base
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
    adapter_type: str
    pair_id: str | None
    adapter_path: Path | None
    eval_set_id: str
    side: str | None = None


def build_eval_schedule(
    pairs: list[PairDef],
    merged_paths: dict[str, dict[str, Path]],
    source_adapter_dirs: dict[str, Path],
) -> dict[str, list[EvalJob]]:
    """Build evaluation schedule grouped by eval_set_id."""
    schedule: dict[str, list[EvalJob]] = {}

    def add_job(eval_set_id: str, job: EvalJob):
        if eval_set_id not in schedule:
            schedule[eval_set_id] = []
        schedule[eval_set_id].append(job)

    # 1. Base model — one eval per unique eval set
    for es_id in EVAL_SETS:
        add_job(es_id, EvalJob(adapter_label="base_model", adapter_type="base", pair_id=None, adapter_path=None, eval_set_id=es_id))

    # 2. Source adapters
    seen_source: set[tuple[str, str]] = set()
    for pair in pairs:
        for adapter, es_idx in [(pair.adapter_a, 0), (pair.adapter_b, 1)]:
            es_id = pair.eval_sets[es_idx]
            key = (adapter.label, es_id)
            if key in seen_source or adapter.label not in source_adapter_dirs:
                continue
            seen_source.add(key)
            add_job(es_id, EvalJob(
                adapter_label=adapter.label, adapter_type="source", pair_id=None,
                adapter_path=source_adapter_dirs[adapter.label], eval_set_id=es_id,
            ))

    # 3. Merged adapters — each condition on both eval sets
    for pair in pairs:
        if pair.pair_id not in merged_paths:
            continue
        for cond_name in ["A_fullrank_naive", "B_fullrank_normeq", "C_fullrank_recommended", "D_compress_normeq", "E_compress_recommended"]:
            if cond_name not in merged_paths[pair.pair_id]:
                continue
            adapter_path = merged_paths[pair.pair_id][cond_name]
            label = f"{pair.pair_id}_{cond_name}"

            # Eval on side a's eval set
            add_job(pair.eval_sets[0], EvalJob(
                adapter_label=label, adapter_type=cond_name, pair_id=pair.pair_id,
                adapter_path=adapter_path, eval_set_id=pair.eval_sets[0], side="a",
            ))
            # Eval on side b's eval set
            add_job(pair.eval_sets[1], EvalJob(
                adapter_label=label, adapter_type=cond_name, pair_id=pair.pair_id,
                adapter_path=adapter_path, eval_set_id=pair.eval_sets[1], side="b",
            ))

    return schedule


def run_evaluation(
    base_model, base_model_name: str, tokenizer,
    schedule: dict[str, list[EvalJob]],
    max_examples: int = 500, max_length: int = 512,
    verbose: bool = False, dtype=None,
) -> list[EvalResult]:
    """Execute all evaluations, grouped by eval set."""
    all_results: list[EvalResult] = []

    for es_id, jobs in schedule.items():
        if not jobs:
            continue

        eval_set = EVAL_SETS[es_id]
        print(f"\n{'─' * 60}")
        print(f"  Eval set: {es_id}  ({eval_set.description})")
        print(f"  {len(jobs)} evaluations")
        print(f"{'─' * 60}")

        examples = load_eval_dataset(eval_set, max_examples)

        seen_results: dict[str, dict[str, Any]] = {}

        for i, job in enumerate(jobs):
            label = job.adapter_label
            atype = job.adapter_type

            if verbose:
                print(f"\n  [{i + 1}/{len(jobs)}] {label} ({atype}) on {es_id}")

            cache_key = f"{label}_{es_id}"
            if cache_key in seen_results:
                if verbose:
                    print("    (reusing cached result)")
                metrics = seen_results[cache_key]
            else:
                t0 = time.time()
                try:
                    if atype == "base":
                        metrics = evaluate_perplexity(base_model, tokenizer, examples, max_length)
                    else:
                        metrics = evaluate_with_adapter(base_model_name, tokenizer, job.adapter_path, examples, max_length, dtype=dtype)
                    metrics["eval_time_s"] = round(time.time() - t0, 2)
                    seen_results[cache_key] = metrics
                except Exception as e:
                    logger.error("Eval failed for %s on %s: %s", label, es_id, e, exc_info=True)
                    metrics = {
                        "perplexity": float("nan"), "token_mean_loss": float("nan"),
                        "example_mean_loss": float("nan"), "example_std_loss": float("nan"),
                        "n_examples": 0, "total_tokens": 0, "scored_tokens": 0,
                        "truncated_examples": 0, "per_example_losses": [],
                        "per_example_tokens": [], "per_example_scored": [], "eval_time_s": 0.0,
                    }

            if verbose and not np.isnan(metrics["perplexity"]):
                print(f"    PPL: {metrics['perplexity']:.2f}  (loss: {metrics['token_mean_loss']:.4f})")

            result = EvalResult(
                adapter_label=label, adapter_type=atype, pair_id=job.pair_id,
                eval_set_id=es_id, eval_task=eval_set.task, side=job.side,
                perplexity=metrics["perplexity"], token_mean_loss=metrics["token_mean_loss"],
                example_mean_loss=metrics["example_mean_loss"], example_std_loss=metrics["example_std_loss"],
                n_examples=metrics["n_examples"], total_tokens=metrics["total_tokens"],
                scored_tokens=metrics["scored_tokens"], truncated_examples=metrics["truncated_examples"],
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


def print_summary(results: list[EvalResult]) -> None:
    """Print human-readable summary table."""
    print(f"\n{'=' * 110}")
    print("  STUDY 17 — PERPLEXITY RESULTS (token-weighted)")
    print(f"{'=' * 110}")
    print(
        f"  {'Adapter':<40s}  {'Type':<24s}  {'EvalSet':<12s}  "
        f"{'PPL':>8s}  {'TokLoss':>8s}  {'Scored':>7s}  {'Side':>4s}"
    )
    print(f"  {'─' * 106}")

    for r in sorted(results, key=lambda x: (x.eval_set_id, x.pair_id or "", x.adapter_type, x.adapter_label)):
        ppl_str = f"{r.perplexity:.2f}" if not np.isnan(r.perplexity) else "FAILED"
        loss_str = f"{r.token_mean_loss:.4f}" if not np.isnan(r.token_mean_loss) else "—"
        side_str = r.side or "—"
        print(
            f"  {r.adapter_label:<40s}  {r.adapter_type:<24s}  {r.eval_set_id:<12s}  "
            f"{ppl_str:>8s}  {loss_str:>8s}  {r.scored_tokens:>7d}  {side_str:>4s}"
        )

    print(f"{'=' * 110}\n")


def print_behavioral_comparison(results: list[EvalResult], pairs: list[PairDef]) -> None:
    """Print condition-wise behavioral comparison table for each pair."""
    print(f"\n{'=' * 100}")
    print("  STUDY 17 — BEHAVIORAL COMPARISON (worst-task loss change)")
    print(f"{'=' * 100}")

    for pair in pairs:
        pair_results = [r for r in results if r.pair_id == pair.pair_id]
        if not pair_results:
            continue

        boundary_flag = " [BOUNDARY]" if pair.is_boundary else ""
        print(f"\n  {pair.adapter_a.label} × {pair.adapter_b.label}{boundary_flag}")
        print(f"  {'Condition':<28s}  {'Side A loss':>11s}  {'Side B loss':>11s}  {'Worst':>8s}")
        print(f"  {'-' * 65}")

        for cond_name in ["A_fullrank_naive", "B_fullrank_normeq", "C_fullrank_recommended", "D_compress_normeq", "E_compress_recommended"]:
            cond_results = [r for r in pair_results if r.adapter_type == cond_name]
            loss_a = next((r.token_mean_loss for r in cond_results if r.side == "a"), float("nan"))
            loss_b = next((r.token_mean_loss for r in cond_results if r.side == "b"), float("nan"))
            worst = max(loss_a, loss_b) if not (np.isnan(loss_a) or np.isnan(loss_b)) else float("nan")

            la_str = f"{loss_a:.4f}" if not np.isnan(loss_a) else "—"
            lb_str = f"{loss_b:.4f}" if not np.isnan(loss_b) else "—"
            w_str = f"{worst:.4f}" if not np.isnan(worst) else "—"
            print(f"  {cond_name:<28s}  {la_str:>11s}  {lb_str:>11s}  {w_str:>8s}")

    print(f"\n{'=' * 100}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 17: Compress-Then-Merge — Phase 2 (GPU)")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output-dir", type=Path, default=Path("results/study17"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "gradience" / "adapters")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pairs", type=str, nargs="*", default=None)
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    pairs_to_run = PAIRS
    if args.pairs:
        pairs_to_run = [p for p in PAIRS if p.pair_id in args.pairs]
        if not pairs_to_run:
            print(f"No matching pairs. Available: {[p.pair_id for p in PAIRS]}")
            sys.exit(1)

    # Determine needed source adapters
    needed_sources = set()
    for pair in pairs_to_run:
        needed_sources.add(pair.adapter_a.label)
        needed_sources.add(pair.adapter_b.label)
    source_adapters_to_eval = [a for a in SOURCE_ADAPTERS if a.label in needed_sources]

    print("Study 17 — Compress-Then-Merge Perplexity Validation")
    print(f"  Base model:     {args.base_model}")
    print(f"  Pairs:          {len(pairs_to_run)}")
    print(f"  Source adapters: {len(source_adapters_to_eval)}")
    print(f"  Max examples:   {args.max_examples}")
    print("  Conditions:     A (naive), B (norm-eq), C (rec), D (comp+norm-eq), E (comp+rec)")
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
            pair_merged: dict[str, Path] = {}
            for cond_name, _, _, _ in CONDITION_DEFS:
                cond_dir = merge_base / pair.pair_id / cond_name
                if merged_adapter_has_weights(cond_dir):
                    pair_merged[cond_name] = cond_dir
            if len(pair_merged) >= 2:
                merged_paths[pair.pair_id] = pair_merged
            else:
                logger.warning("Missing weights for %s — will skip", pair.pair_id)
    else:
        print("Phase 1b: Executing merges ...")
        merged_paths = run_merges(pairs_to_run, args.cache_dir, args.output_dir, verbose=args.verbose)

    print(f"\n  {len(merged_paths)} pairs merged successfully\n")

    # ── Dry-run exit ───────────────────────────────────────────────────
    if args.dry_run:
        print("Dry-run complete. Merge phase done. Skipping GPU evaluation.")
        schedule = build_eval_schedule(pairs_to_run, merged_paths, source_adapter_dirs)
        total_evals = sum(len(jobs) for jobs in schedule.values())
        print(f"  Would run {total_evals} evaluations:")
        for es_id, jobs in schedule.items():
            print(f"    {es_id}: {len(jobs)} evals")
            for job in jobs:
                print(f"      - {job.adapter_label} ({job.adapter_type})")
        sys.exit(0)

    # ── Phase 2: GPU evaluation ────────────────────────────────────────
    print("Phase 2: GPU-side perplexity evaluation")

    import torch

    dtype = torch.float16
    base_model, tokenizer = load_base_model(args.base_model, dtype=dtype)

    schedule = build_eval_schedule(pairs_to_run, merged_paths, source_adapter_dirs)
    total_evals = sum(len(jobs) for jobs in schedule.values())
    print(f"\n  Total eval jobs: {total_evals}")

    results = run_evaluation(
        base_model, args.base_model, tokenizer, schedule,
        max_examples=args.max_examples, max_length=args.max_length,
        verbose=args.verbose, dtype=dtype,
    )

    # ── Output ─────────────────────────────────────────────────────────
    total_time = time.time() - t_start

    metadata = {
        "study_id": "study17_perplexity",
        "date": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "max_examples": args.max_examples,
        "max_length": args.max_length,
        "seed": SEED,
        "energy_threshold": ENERGY_THRESHOLD,
        "n_pairs": len(pairs_to_run),
        "n_evaluations": len(results),
        "total_time_s": round(total_time, 2),
        "conditions": ["A_fullrank_naive", "B_fullrank_normeq", "C_fullrank_recommended", "D_compress_normeq", "E_compress_recommended"],
        "scoring": "completion-only token-weighted NLL",
    }

    output_path = args.output_dir / "study17_perplexity.json"
    payload = {"metadata": metadata, "results": [asdict(r) for r in results]}
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print_summary(results)
    print_behavioral_comparison(results, pairs_to_run)

    print(f"Results saved to {output_path}")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")


if __name__ == "__main__":
    main()
