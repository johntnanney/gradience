#!/usr/bin/env python3
"""
Public Ecosystem Spectral Census
=================================

CPU-only spectral census of public decoder-only LoRA adapters from
HuggingFace Hub. Tests whether spectral fingerprints separate
architecture effects from task effects at population scale, using
found artifacts rather than controlled training.

Reuses download/audit infrastructure from ``broader_benchmarks.py``
and extends it with:
- Architecture-family and task-category labeling
- 10-dimensional fingerprint vector extraction
- Census-specific analysis (confound assessment, ANOVA, clustering,
  kNN purity, module-type replication)

Usage:
    python scripts/ecosystem_census.py \
        --output-dir field_trials/public_ecosystem_census \
        --cache-dir .cache/census_adapters \
        --tier pilot

The script is idempotent: re-running skips already-audited adapters.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure gradience is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gradience.vnext.audit.lora_audit import LoRAAuditResult, audit_lora_peft_dir

# ============================================================================
# Constants
# ============================================================================

# Architecture family → base model IDs (decoder-only only)
ARCHITECTURE_FAMILIES: dict[str, list[str]] = {
    "llama": [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ],
    "mistral": [
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ],
    "qwen": [
        "Qwen/Qwen2-7B",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-7B-Instruct",
    ],
    "phi": [
        "microsoft/phi-2",
        "microsoft/Phi-3-mini-4k-instruct",
    ],
    "gemma": [
        "google/gemma-7b",
        "google/gemma-2b",
        "google/gemma-2-2b",
        "google/gemma-2-9b",
    ],
}

# Flat lookup: base_model_id → family name
_BASE_MODEL_TO_FAMILY: dict[str, str] = {}
for _fam, _models in ARCHITECTURE_FAMILIES.items():
    for _m in _models:
        _BASE_MODEL_TO_FAMILY[_m] = _fam
        # Also register the short name (after /)
        _BASE_MODEL_TO_FAMILY[_m.split("/")[-1]] = _fam

# All base models to search (flattened)
ALL_BASE_MODELS: list[str] = [m for models in ARCHITECTURE_FAMILIES.values() for m in models]

# Tier → cohort size
TIER_CONFIGS: dict[str, dict[str, int]] = {
    "pilot": {"max_total": 50, "max_per_base": 8, "min_families": 2},
    "core": {"max_total": 150, "max_per_base": 12, "min_families": 3},
    "extended": {"max_total": 300, "max_per_base": 20, "min_families": 4},
}

# Census task categories (maps inferred task → census category)
TASK_CATEGORY_MAP: dict[str, str] = {
    "chat": "chat_instruct",
    "text-generation": "chat_instruct",
    "conversational": "chat_instruct",
    "code": "code",
    "math": "math_reasoning",
    "qa": "chat_instruct",
    "summarization": "chat_instruct",
    "translation": "chat_instruct",
    "text2text": "chat_instruct",
    "medical": "domain_specialist",
    "legal": "domain_specialist",
    "data": "domain_specialist",
    "classification": "classification",
    "local": "general_unknown",
    "general": "general_unknown",
}


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class CensusAdapterRecord:
    """Extended adapter record for census study."""

    repo_id: str
    base_model: str
    architecture_family: str
    inferred_task: str
    task_category: str

    # From adapter_config.json
    rank: int | None = None
    alpha: float | None = None
    lora_dropout: float | None = None
    target_modules: list[str] | None = None
    peft_type: str | None = None

    # From Hub API
    downloads: int | None = None
    tags: list[str] | None = None

    # Computed during pipeline
    n_layers: int | None = None
    total_params: int | None = None
    download_size_mb: float | None = None
    audit_time_s: float | None = None
    status: str = "pending"  # pending | downloaded | audited | failed | excluded
    issues: list[str] = field(default_factory=list)
    exclusion_reason: str | None = None


@dataclass
class SpectralFingerprint:
    """10-dimensional fingerprint vector per adapter."""

    repo_id: str
    architecture_family: str
    task_category: str
    nominal_rank: int

    # 10 fingerprint dimensions
    stable_rank_mean: float
    stable_rank_std: float
    utilization_mean: float
    energy_rank_90_p50: float
    entropy_erank_mean: float
    attn_stable_rank_mean: float
    mlp_stable_rank_mean: float
    attn_utilization_mean: float
    mlp_utilization_mean: float
    edge_gap_mean: float

    def to_vector(self) -> list[float]:
        """Return the 10-dimensional fingerprint as a flat list."""
        return [
            self.stable_rank_mean,
            self.stable_rank_std,
            self.utilization_mean,
            self.energy_rank_90_p50,
            self.entropy_erank_mean,
            self.attn_stable_rank_mean,
            self.mlp_stable_rank_mean,
            self.attn_utilization_mean,
            self.mlp_utilization_mean,
            self.edge_gap_mean,
        ]

    @staticmethod
    def dimension_names() -> list[str]:
        return [
            "stable_rank_mean",
            "stable_rank_std",
            "utilization_mean",
            "energy_rank_90_p50",
            "entropy_erank_mean",
            "attn_stable_rank_mean",
            "mlp_stable_rank_mean",
            "attn_utilization_mean",
            "mlp_utilization_mean",
            "edge_gap_mean",
        ]


# ============================================================================
# Architecture-family inference
# ============================================================================


def infer_architecture_family(base_model: str) -> str:
    """Infer architecture family from base model name."""
    if base_model in _BASE_MODEL_TO_FAMILY:
        return _BASE_MODEL_TO_FAMILY[base_model]

    # Fuzzy match on model name
    lower = base_model.lower()
    if "llama" in lower:
        return "llama"
    if "mistral" in lower or "mixtral" in lower:
        return "mistral"
    if "qwen" in lower:
        return "qwen"
    if "phi" in lower:
        return "phi"
    if "gemma" in lower:
        return "gemma"
    return "unknown"


# ============================================================================
# Task inference (reuses broader_benchmarks heuristic, maps to census categories)
# ============================================================================


def _infer_task(tags: set[str], repo_id: str) -> str:
    """Heuristic task inference from model tags and repo name."""
    name_lower = repo_id.lower()

    tag_map = {
        "text-generation": "text-generation",
        "text-classification": "classification",
        "question-answering": "qa",
        "summarization": "summarization",
        "translation": "translation",
        "conversational": "chat",
        "text2text-generation": "text2text",
    }
    for tag, task in tag_map.items():
        if tag in tags:
            return task

    if any(k in name_lower for k in ("code", "coder", "starcoder")):
        return "code"
    if any(k in name_lower for k in ("math", "gsm", "arithmetic")):
        return "math"
    if any(k in name_lower for k in ("chat", "instruct", "sft", "rlhf", "dpo")):
        return "chat"
    if any(k in name_lower for k in ("medical", "bio", "clinical")):
        return "medical"
    if any(k in name_lower for k in ("legal", "law", "contract")):
        return "legal"
    if any(k in name_lower for k in ("sql", "data", "table")):
        return "data"
    if any(k in name_lower for k in ("ner", "classify", "sentiment")):
        return "classification"

    return "general"


def infer_task_category(inferred_task: str) -> str:
    """Map inferred task to census task category."""
    return TASK_CATEGORY_MAP.get(inferred_task, "general_unknown")


# ============================================================================
# Discovery
# ============================================================================


def discover_census_adapters(
    families: list[str],
    max_per_base: int = 8,
    max_total: int = 50,
    min_downloads: int = 10,
) -> list[CensusAdapterRecord]:
    """
    Search HuggingFace Hub for decoder-only LoRA adapters across
    specified architecture families.

    Interleaves families round-robin to ensure architecture diversity
    before hitting max_total.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    records: list[CensusAdapterRecord] = []
    seen_repos: set[str] = set()

    # Build per-family base model lists
    family_base_models: dict[str, list[str]] = {}
    for fam in families:
        if fam in ARCHITECTURE_FAMILIES:
            family_base_models[fam] = list(ARCHITECTURE_FAMILIES[fam])
        else:
            print(f"  Warning: unknown family '{fam}', skipping")

    # Per-family budget: divide max_total across families, then
    # allow overflow into remaining budget
    per_family_target = max_total // len(family_base_models) if family_base_models else max_total

    # Interleave: process one base model per family in round-robin
    # Build interleaved list of (family, base_model)
    interleaved: list[tuple[str, str]] = []
    max_bases = max(len(models) for models in family_base_models.values()) if family_base_models else 0
    for i in range(max_bases):
        for fam in families:
            if fam in family_base_models and i < len(family_base_models[fam]):
                interleaved.append((fam, family_base_models[fam][i]))

    family_counts: dict[str, int] = {fam: 0 for fam in families}

    for family, base_model in interleaved:
        if len(records) >= max_total:
            break

        # Soft cap per family (can overflow if other families are exhausted)
        if family_counts[family] >= per_family_target * 2:
            continue

        short_name = base_model.split("/")[-1]
        print(f"\n  Searching for LoRA adapters: {base_model} (family={family})")

        try:
            models = api.list_models(
                filter="peft",
                search=short_name,
                sort="downloads",
                direction=-1,
                limit=80,
                full=True,
            )

            count_for_base = 0
            for model in models:
                if count_for_base >= max_per_base or len(records) >= max_total:
                    break

                repo_id = model.id
                if repo_id in seen_repos:
                    continue

                # Skip base models themselves
                if repo_id in {m for ms in ARCHITECTURE_FAMILIES.values() for m in ms}:
                    continue

                downloads = getattr(model, "downloads", 0) or 0
                if downloads < min_downloads:
                    continue

                # Check for required PEFT files
                try:
                    siblings = model.siblings or []
                    filenames = {s.rfilename for s in siblings}
                except Exception:
                    filenames = set()

                has_config = any(f.startswith("adapter_config") for f in filenames)
                has_weights = any(f.startswith("adapter_model") or f == "pytorch_model.bin" for f in filenames)

                if not (has_config and has_weights):
                    continue

                # Pre-download size estimate from siblings metadata
                try:
                    weight_size_bytes = 0
                    has_size_info = False
                    for s in model.siblings or []:
                        if s.rfilename.startswith("adapter_model") or s.rfilename == "pytorch_model.bin":
                            sz = getattr(s, "size", None)
                            if sz is not None and sz > 0:
                                weight_size_bytes += sz
                                has_size_info = True
                    if has_size_info and weight_size_bytes / (1024 * 1024) > 500.0:
                        continue  # Skip oversized adapters before download
                except Exception:
                    pass  # If size unavailable, proceed to download

                # Validate that this adapter is plausibly for the target base model
                # by checking tags or card_data for base model reference
                tags = set(getattr(model, "tags", []) or [])
                # Quick heuristic: repo name or tags should contain part of base model name
                base_short_lower = short_name.lower()
                repo_lower = repo_id.lower()
                tag_str = " ".join(tags).lower()
                if base_short_lower not in repo_lower and base_short_lower not in tag_str:
                    # Also check partial matches for common patterns
                    family_keywords = {
                        "llama": ["llama"],
                        "mistral": ["mistral"],
                        "qwen": ["qwen"],
                        "phi": ["phi"],
                        "gemma": ["gemma"],
                    }
                    keywords = family_keywords.get(family, [])
                    if not any(kw in repo_lower or kw in tag_str for kw in keywords):
                        continue

                inferred_task = _infer_task(tags, repo_id)
                task_category = infer_task_category(inferred_task)

                seen_repos.add(repo_id)
                records.append(
                    CensusAdapterRecord(
                        repo_id=repo_id,
                        base_model=base_model,
                        architecture_family=family,
                        inferred_task=inferred_task,
                        task_category=task_category,
                        downloads=downloads,
                        tags=sorted(tags) if tags else [],
                    )
                )
                count_for_base += 1
                family_counts[family] = family_counts.get(family, 0) + 1
                print(f"    + {repo_id} (dl={downloads}, task={task_category})")

        except Exception as e:
            print(f"    Error searching {base_model}: {e}")
            continue

    # Summary
    fam_counts = {}
    task_counts = {}
    for r in records:
        fam_counts[r.architecture_family] = fam_counts.get(r.architecture_family, 0) + 1
        task_counts[r.task_category] = task_counts.get(r.task_category, 0) + 1

    print(f"\n  Discovered {len(records)} adapters")
    print(f"  By family: {fam_counts}")
    print(f"  By task: {task_counts}")
    return records


# ============================================================================
# Download (reuses broader_benchmarks pattern)
# ============================================================================


def download_adapter(
    repo_id: str,
    cache_dir: Path,
    max_size_mb: float = 500.0,
) -> tuple[Path, float]:
    """Download a PEFT adapter from HuggingFace Hub. Returns (local_path, size_mb)."""
    from huggingface_hub import snapshot_download

    local_dir = cache_dir / repo_id.replace("/", "--")

    config_candidates = list(local_dir.glob("adapter_config*"))
    weight_candidates = list(local_dir.glob("adapter_model*"))
    if config_candidates and weight_candidates:
        size_mb = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file()) / (1024 * 1024)
        return local_dir, size_mb

    local_path = snapshot_download(
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

    size_mb = sum(f.stat().st_size for f in Path(local_path).rglob("*") if f.is_file()) / (1024 * 1024)

    if size_mb > max_size_mb:
        raise ValueError(f"Adapter {repo_id} is {size_mb:.1f} MB (limit: {max_size_mb} MB)")

    return Path(local_path), size_mb


def read_adapter_config(adapter_path: Path) -> dict[str, Any]:
    """Read adapter_config.json and extract metadata."""
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception:
        return {}


# ============================================================================
# Audit
# ============================================================================


def audit_adapter(
    repo_id: str,
    adapter_path: Path,
    record: CensusAdapterRecord,
) -> tuple[dict[str, Any] | None, CensusAdapterRecord]:
    """Run spectral audit on one adapter. Returns (audit_dict, updated_record)."""
    t0 = time.time()
    try:
        result: LoRAAuditResult = audit_lora_peft_dir(
            adapter_path,
            compute_udr=False,
            include_top_singular_values=4,
            rank_policies=["energy@0.90", "knee", "erank"],
        )

        elapsed = time.time() - t0
        record.audit_time_s = elapsed
        record.n_layers = result.n_layers
        record.total_params = result.total_lora_params

        if result.issues:
            record.issues.extend(result.issues)

        if result.n_layers == 0:
            record.status = "excluded"
            record.exclusion_reason = "No LoRA layers found"
            return None, record

        # Extract config metadata
        cfg = read_adapter_config(adapter_path)
        record.peft_type = cfg.get("peft_type")
        record.rank = cfg.get("r") or (result.layers[0].r if result.layers else None)
        record.alpha = cfg.get("lora_alpha") or (result.layers[0].alpha if result.layers else None)
        record.lora_dropout = cfg.get("lora_dropout")
        target_modules = cfg.get("target_modules")
        if isinstance(target_modules, list):
            record.target_modules = target_modules

        # Re-validate architecture family from actual config base_model
        config_base = cfg.get("base_model_name_or_path", "")
        if config_base:
            actual_family = infer_architecture_family(config_base)
            if actual_family != "unknown" and actual_family != record.architecture_family:
                record.architecture_family = actual_family
            record.base_model = config_base

        # Verify LORA peft_type
        if record.peft_type and record.peft_type.upper() != "LORA":
            record.status = "excluded"
            record.exclusion_reason = f"peft_type is {record.peft_type}, not LORA"
            return None, record

        audit_dict = result.to_summary_dict(include_layers=True)
        record.status = "audited"
        return audit_dict, record

    except Exception as e:
        elapsed = time.time() - t0
        record.audit_time_s = elapsed
        record.status = "failed"
        record.issues.append(f"Audit error: {e}")
        traceback.print_exc()
        return None, record


# ============================================================================
# Fingerprint extraction
# ============================================================================


def extract_fingerprint(
    record: CensusAdapterRecord,
    audit_dict: dict[str, Any],
) -> SpectralFingerprint | None:
    """Extract 10-dimensional fingerprint vector from audit results."""
    layer_rows = audit_dict.get("layer_data", {}).get("layer_rows", [])
    if not layer_rows:
        return None

    # Per-layer metrics
    stable_ranks: list[float] = []
    utilizations: list[float] = []
    energy_rank_90s: list[float] = []
    effective_ranks: list[float] = []
    edge_gaps: list[float] = []

    attn_stable_ranks: list[float] = []
    mlp_stable_ranks: list[float] = []
    attn_utilizations: list[float] = []
    mlp_utilizations: list[float] = []

    for layer in layer_rows:
        sr = layer.get("stable_rank")
        ut = layer.get("utilization")
        er90 = layer.get("energy_rank_90")
        erank = layer.get("effective_rank")
        mt = layer.get("module_type", "other")

        if sr is not None:
            stable_ranks.append(float(sr))
        if ut is not None:
            utilizations.append(float(ut))
        if er90 is not None:
            energy_rank_90s.append(float(er90))
        if erank is not None:
            effective_ranks.append(float(erank))

        # Edge gap: sigma_1 / sigma_2
        top_sv = layer.get("top_singular_values")
        if top_sv and len(top_sv) >= 2 and top_sv[1] > 0:
            edge_gaps.append(float(top_sv[0]) / float(top_sv[1]))

        # Module-type breakdown
        if mt == "attn":
            if sr is not None:
                attn_stable_ranks.append(float(sr))
            if ut is not None:
                attn_utilizations.append(float(ut))
        elif mt == "mlp":
            if sr is not None:
                mlp_stable_ranks.append(float(sr))
            if ut is not None:
                mlp_utilizations.append(float(ut))

    if not stable_ranks:
        return None

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def _std(vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        m = _mean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

    def _median(vals: list[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2

    return SpectralFingerprint(
        repo_id=record.repo_id,
        architecture_family=record.architecture_family,
        task_category=record.task_category,
        nominal_rank=record.rank or 0,
        stable_rank_mean=_mean(stable_ranks),
        stable_rank_std=_std(stable_ranks),
        utilization_mean=_mean(utilizations),
        energy_rank_90_p50=_median(energy_rank_90s),
        entropy_erank_mean=_mean(effective_ranks),
        attn_stable_rank_mean=_mean(attn_stable_ranks),
        mlp_stable_rank_mean=_mean(mlp_stable_ranks),
        attn_utilization_mean=_mean(attn_utilizations),
        mlp_utilization_mean=_mean(mlp_utilizations),
        edge_gap_mean=_mean(edge_gaps),
    )


# ============================================================================
# Pilot validation gate
# ============================================================================


def evaluate_pilot_gate(
    records: list[CensusAdapterRecord],
    fingerprints: list[SpectralFingerprint],
) -> dict[str, Any]:
    """Evaluate pilot validation criteria. Returns gate report dict."""
    audited = [r for r in records if r.status == "audited"]
    attempted = [r for r in records if r.status in ("audited", "failed", "excluded")]
    success_rate = len(audited) / len(attempted) if attempted else 0.0

    # Criterion 1: Pipeline viability (>=80% success)
    c1_pass = success_rate >= 0.80

    # Criterion 2: Metric sanity
    c2_pass = True
    issues: list[str] = []
    for fp in fingerprints:
        if fp.stable_rank_mean < 1.0:
            c2_pass = False
            issues.append(f"{fp.repo_id}: stable_rank_mean={fp.stable_rank_mean:.3f} < 1.0")
        if fp.utilization_mean <= 0.0 or fp.utilization_mean > 1.0:
            c2_pass = False
            issues.append(f"{fp.repo_id}: utilization_mean={fp.utilization_mean:.3f} out of (0,1]")

    # Criterion 3: Architecture coverage (>=2 families with >=8 adapters)
    fam_counts: dict[str, int] = {}
    for fp in fingerprints:
        fam_counts[fp.architecture_family] = fam_counts.get(fp.architecture_family, 0) + 1
    families_with_8plus = sum(1 for c in fam_counts.values() if c >= 8)
    c3_pass = families_with_8plus >= 2

    # Criterion 4: Task coverage (>=3 categories with >=5 adapters)
    task_counts: dict[str, int] = {}
    for fp in fingerprints:
        task_counts[fp.task_category] = task_counts.get(fp.task_category, 0) + 1
    tasks_with_5plus = sum(1 for c in task_counts.values() if c >= 5)
    c4_pass = tasks_with_5plus >= 3

    # Criterion 5: Visible signal (any metric differs between families)
    c5_pass = False
    c5_details = ""
    if len(fam_counts) >= 2:
        # Check stable_rank_mean difference between families
        by_family: dict[str, list[float]] = {}
        for fp in fingerprints:
            by_family.setdefault(fp.architecture_family, []).append(fp.stable_rank_mean)

        families_sorted = sorted(by_family.keys())
        for i, f1 in enumerate(families_sorted):
            for f2 in families_sorted[i + 1 :]:
                m1 = sum(by_family[f1]) / len(by_family[f1])
                m2 = sum(by_family[f2]) / len(by_family[f2])
                diff = abs(m1 - m2)
                pooled_mean = (m1 + m2) / 2
                if pooled_mean > 0 and diff / pooled_mean > 0.05:
                    c5_pass = True
                    c5_details = (
                        f"stable_rank_mean: {f1}={m1:.3f} vs {f2}={m2:.3f} "
                        f"(diff={diff:.3f}, {diff / pooled_mean:.1%} relative)"
                    )
                    break
            if c5_pass:
                break

    overall = c1_pass and c2_pass and c3_pass and c4_pass
    # c5 failure doesn't block — just documented

    return {
        "overall_pass": overall,
        "criteria": {
            "c1_pipeline_viability": {
                "pass": c1_pass,
                "success_rate": round(success_rate, 3),
                "threshold": 0.80,
                "attempted": len(attempted),
                "audited": len(audited),
            },
            "c2_metric_sanity": {
                "pass": c2_pass,
                "issues": issues[:10],
            },
            "c3_architecture_coverage": {
                "pass": c3_pass,
                "family_counts": fam_counts,
                "families_with_8_plus": families_with_8plus,
                "threshold": 2,
            },
            "c4_task_coverage": {
                "pass": c4_pass,
                "task_counts": task_counts,
                "tasks_with_5_plus": tasks_with_5plus,
                "threshold": 3,
            },
            "c5_visible_signal": {
                "pass": c5_pass,
                "note": "Non-blocking criterion",
                "details": c5_details if c5_pass else "No detectable difference between families",
            },
        },
        "recommendation": ("PROCEED to core cohort" if overall else "DIAGNOSE pipeline issues before expanding"),
    }


def evaluate_pilot_plus_gate(
    records: list[CensusAdapterRecord],
    fingerprints: list[SpectralFingerprint],
    phase2b: dict[str, Any],
    phase5: dict[str, Any],
) -> dict[str, Any]:
    """
    Pilot-plus validation gate (second gate after original pilot).

    Four success conditions:
    S1: Pipeline viability >= 70% (relaxed from 80% given observed adapter population)
    S2: >= 3 architecture families with >= 5 adapters each
    S3: At least 1 metric with residualized arch eta-sq > 0.05
    S4: Subtype breakdown has >= 3 distinct subtypes with >= 10 layers each
    """
    audited = [r for r in records if r.status == "audited"]
    attempted = [r for r in records if r.status in ("audited", "failed", "excluded")]
    success_rate = len(audited) / len(attempted) if attempted else 0.0

    # S1: Pipeline viability >= 70%
    s1_pass = success_rate >= 0.70
    s1 = {
        "pass": s1_pass,
        "success_rate": round(success_rate, 3),
        "threshold": 0.70,
        "attempted": len(attempted),
        "audited": len(audited),
    }

    # S2: >= 3 families with >= 5 adapters each
    fam_counts: dict[str, int] = {}
    for fp in fingerprints:
        fam_counts[fp.architecture_family] = fam_counts.get(fp.architecture_family, 0) + 1
    families_with_5plus = sum(1 for c in fam_counts.values() if c >= 5)
    s2_pass = families_with_5plus >= 3
    s2 = {
        "pass": s2_pass,
        "family_counts": fam_counts,
        "families_with_5_plus": families_with_5plus,
        "threshold": 3,
    }

    # S3: At least 1 metric with residualized arch eta-sq > 0.05
    variant_robustness = phase2b.get("variant_robustness", {})
    metrics_above_005 = variant_robustness.get("metrics_above_005", 0)
    s3_pass = metrics_above_005 >= 1
    s3 = {
        "pass": s3_pass,
        "metrics_above_005": metrics_above_005,
        "mean_residualized_arch_eta_sq": variant_robustness.get("mean_residualized_arch_eta_sq", 0.0),
        "threshold": ">=1 metric with residualized eta-sq > 0.05",
    }

    # S4: Subtype breakdown has >= 3 distinct subtypes with >= 10 layers each
    subtype_breakdown = phase5.get("subtype_breakdown", {})
    subtypes_with_10plus = sum(1 for d in subtype_breakdown.values() if d.get("n_layers", 0) >= 10)
    s4_pass = subtypes_with_10plus >= 3
    s4 = {
        "pass": s4_pass,
        "subtypes_with_10_plus_layers": subtypes_with_10plus,
        "total_subtypes": len(subtype_breakdown),
        "threshold": 3,
    }

    overall = s1_pass and s2_pass and s3_pass and s4_pass

    return {
        "overall_pass": overall,
        "criteria": {
            "s1_pipeline_viability": s1,
            "s2_family_coverage": s2,
            "s3_residualized_signal": s3,
            "s4_subtype_coverage": s4,
        },
        "recommendation": ("PROCEED to core cohort" if overall else "Address failing criteria before expanding"),
    }


# ============================================================================
# Statistical helpers (pure Python, no numpy/scipy dependency)
# ============================================================================


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = _mean(x), _mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx * dy == 0:
        return 0.0
    return num / (dx * dy)


def _distribution_stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    vals_sorted = sorted(vals)
    return {
        "n": len(vals_sorted),
        "mean": round(_mean(vals_sorted), 4),
        "median": round(_median(vals_sorted), 4),
        "std": round(_std(vals_sorted), 4),
        "min": round(vals_sorted[0], 4),
        "max": round(vals_sorted[-1], 4),
        "p25": round(_percentile(vals_sorted, 0.25), 4),
        "p75": round(_percentile(vals_sorted, 0.75), 4),
    }


# ============================================================================
# Analysis Phase 1: Distributional characterization
# ============================================================================


def phase1_distributional(fingerprints: list[SpectralFingerprint]) -> dict[str, Any]:
    """Empirical distributions of each core metric across the full cohort."""
    result: dict[str, Any] = {"phase": "distributional_characterization"}

    # Global distributions
    for dim_name in SpectralFingerprint.dimension_names():
        vals = [getattr(fp, dim_name) for fp in fingerprints]
        result[f"global_{dim_name}"] = _distribution_stats(vals)

    # By architecture family
    by_family: dict[str, dict[str, Any]] = {}
    families: dict[str, list[SpectralFingerprint]] = {}
    for fp in fingerprints:
        families.setdefault(fp.architecture_family, []).append(fp)
    for fam, fps in sorted(families.items()):
        fam_stats: dict[str, Any] = {"n": len(fps)}
        for dim_name in SpectralFingerprint.dimension_names():
            vals = [getattr(fp, dim_name) for fp in fps]
            fam_stats[dim_name] = _distribution_stats(vals)
        by_family[fam] = fam_stats
    result["by_architecture_family"] = by_family

    # By task category
    by_task: dict[str, dict[str, Any]] = {}
    tasks: dict[str, list[SpectralFingerprint]] = {}
    for fp in fingerprints:
        tasks.setdefault(fp.task_category, []).append(fp)
    for task, fps in sorted(tasks.items()):
        task_stats: dict[str, Any] = {"n": len(fps)}
        for dim_name in SpectralFingerprint.dimension_names():
            vals = [getattr(fp, dim_name) for fp in fps]
            task_stats[dim_name] = _distribution_stats(vals)
        by_task[task] = task_stats
    result["by_task_category"] = by_task

    return result


# ============================================================================
# Analysis Phase 2: Confound assessment
# ============================================================================


def phase2_confound_assessment(fingerprints: list[SpectralFingerprint]) -> dict[str, Any]:
    """
    Regress core metrics on confound variables (nominal_rank, alpha proxy, etc.).
    Uses simple R-squared from bivariate regression (no numpy needed).
    """
    result: dict[str, Any] = {"phase": "confound_assessment"}

    # Confound: nominal_rank vs each metric
    confounds: dict[str, list[float]] = {
        "nominal_rank": [float(fp.nominal_rank) for fp in fingerprints],
    }

    metrics_to_check = [
        "stable_rank_mean",
        "utilization_mean",
        "energy_rank_90_p50",
        "entropy_erank_mean",
        "edge_gap_mean",
    ]

    for confound_name, confound_vals in confounds.items():
        conf_results: dict[str, Any] = {}
        for metric_name in metrics_to_check:
            metric_vals = [getattr(fp, metric_name) for fp in fingerprints]
            if len(metric_vals) < 5:
                continue
            r = _pearson(confound_vals, metric_vals)
            r_sq = r * r
            conf_results[metric_name] = {
                "pearson_r": round(r, 4),
                "r_squared": round(r_sq, 4),
                "n": len(metric_vals),
                "interpretation": (
                    "strong confound (R^2 > 0.3)"
                    if r_sq > 0.3
                    else "moderate confound (R^2 > 0.1)"
                    if r_sq > 0.1
                    else "weak confound"
                ),
            }
        result[confound_name] = conf_results

    # Overall confound assessment
    max_r_sq = 0.0
    for conf_name in confounds:
        for metric_name in metrics_to_check:
            entry = result.get(conf_name, {}).get(metric_name, {})
            max_r_sq = max(max_r_sq, entry.get("r_squared", 0.0))

    result["overall"] = {
        "max_r_squared": round(max_r_sq, 4),
        "residualization_recommended": max_r_sq > 0.3,
        "note": (
            "Confound R^2 > 0.3 detected; residualize metrics before architecture/task analysis"
            if max_r_sq > 0.3
            else "Confound effects are moderate to weak; proceed with raw metrics"
        ),
    }

    return result


# ============================================================================
# Analysis Phase 2b: Analytic control subsets
# ============================================================================


def _residualize(values: list[float], confound: list[float]) -> list[float]:
    """
    Remove linear confound effect from values via OLS residuals.
    Returns residuals = values - (alpha + beta * confound).
    """
    n = len(values)
    if n < 3:
        return list(values)
    mx = _mean(confound)
    my = _mean(values)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(confound, values))
    denom = sum((xi - mx) ** 2 for xi in confound)
    if denom == 0:
        return list(values)
    beta = num / denom
    alpha = my - beta * mx
    return [yi - (alpha + beta * xi) for xi, yi in zip(confound, values)]


def phase2b_analytic_controls(
    fingerprints: list[SpectralFingerprint],
) -> dict[str, Any]:
    """
    Phase 2b: Three analytic control subsets.

    1. Rank-matched subset: restrict to most common nominal_rank to eliminate
       rank as a confound entirely.
    2. Task-label confidence subset: exclude general_unknown adapters to test
       whether task signal strengthens when ambiguous labels are removed.
    3. Variant-aware architecture robustness: re-run ANOVA on base-model
       variants within each family to check whether signal is family-level
       or variant-level.
    """
    result: dict[str, Any] = {"phase": "analytic_control_subsets"}

    metrics = [
        "stable_rank_mean",
        "utilization_mean",
        "energy_rank_90_p50",
        "entropy_erank_mean",
        "edge_gap_mean",
    ]

    # --- Control 1: Rank-matched subset ---
    rank_counts: dict[int, int] = {}
    for fp in fingerprints:
        rank_counts[fp.nominal_rank] = rank_counts.get(fp.nominal_rank, 0) + 1
    if rank_counts:
        modal_rank = max(rank_counts, key=lambda r: rank_counts[r])
        rank_matched = [fp for fp in fingerprints if fp.nominal_rank == modal_rank]
    else:
        modal_rank = 0
        rank_matched = []

    rank_matched_results: dict[str, Any] = {
        "modal_rank": modal_rank,
        "n_matched": len(rank_matched),
        "n_total": len(fingerprints),
    }

    if len(rank_matched) >= 5:
        # Architecture ANOVA on rank-matched subset
        arch_eta_sqs: dict[str, float] = {}
        task_eta_sqs: dict[str, float] = {}
        for metric_name in metrics:
            arch_groups: dict[str, list[float]] = {}
            task_groups: dict[str, list[float]] = {}
            for fp in rank_matched:
                arch_groups.setdefault(fp.architecture_family, []).append(getattr(fp, metric_name))
                task_groups.setdefault(fp.task_category, []).append(getattr(fp, metric_name))
            a_res = _one_way_anova_eta_sq(arch_groups)
            t_res = _one_way_anova_eta_sq(task_groups)
            arch_eta_sqs[metric_name] = a_res.get("eta_sq", 0.0)
            task_eta_sqs[metric_name] = t_res.get("eta_sq", 0.0)
        rank_matched_results["architecture_eta_sq"] = arch_eta_sqs
        rank_matched_results["task_eta_sq"] = task_eta_sqs
        rank_matched_results["mean_arch_eta_sq"] = round(_mean(list(arch_eta_sqs.values())), 4)
        rank_matched_results["mean_task_eta_sq"] = round(_mean(list(task_eta_sqs.values())), 4)
    else:
        rank_matched_results["note"] = "Insufficient rank-matched adapters for analysis"

    result["rank_matched_subset"] = rank_matched_results

    # --- Control 2: Task-label confidence subset ---
    confident = [fp for fp in fingerprints if fp.task_category != "general_unknown"]
    task_confidence_results: dict[str, Any] = {
        "n_confident": len(confident),
        "n_excluded": len(fingerprints) - len(confident),
    }

    if len(confident) >= 5:
        arch_eta_sqs_c: dict[str, float] = {}
        task_eta_sqs_c: dict[str, float] = {}
        for metric_name in metrics:
            arch_groups = {}
            task_groups = {}
            for fp in confident:
                arch_groups.setdefault(fp.architecture_family, []).append(getattr(fp, metric_name))
                task_groups.setdefault(fp.task_category, []).append(getattr(fp, metric_name))
            a_res = _one_way_anova_eta_sq(arch_groups)
            t_res = _one_way_anova_eta_sq(task_groups)
            arch_eta_sqs_c[metric_name] = a_res.get("eta_sq", 0.0)
            task_eta_sqs_c[metric_name] = t_res.get("eta_sq", 0.0)
        task_confidence_results["architecture_eta_sq"] = arch_eta_sqs_c
        task_confidence_results["task_eta_sq"] = task_eta_sqs_c
        task_confidence_results["mean_arch_eta_sq"] = round(_mean(list(arch_eta_sqs_c.values())), 4)
        task_confidence_results["mean_task_eta_sq"] = round(_mean(list(task_eta_sqs_c.values())), 4)
    else:
        task_confidence_results["note"] = "Insufficient confident-label adapters for analysis"

    result["task_label_confidence_subset"] = task_confidence_results

    # --- Control 3: Variant-aware architecture robustness ---
    # Group by base_model variant within each family, then check whether
    # variant explains more variance than family
    # Use repo_id prefix heuristic as variant proxy (since base_model stored on record, not fingerprint)
    # Re-run architecture ANOVA with residualized metrics (removing rank confound)
    confound_vals = [float(fp.nominal_rank) for fp in fingerprints]

    variant_results: dict[str, Any] = {}
    for metric_name in metrics:
        raw_vals = [getattr(fp, metric_name) for fp in fingerprints]
        residuals = _residualize(raw_vals, confound_vals)

        # ANOVA on residualized values by architecture
        arch_groups: dict[str, list[float]] = {}
        for fp, resid in zip(fingerprints, residuals):
            arch_groups.setdefault(fp.architecture_family, []).append(resid)
        res = _one_way_anova_eta_sq(arch_groups)
        variant_results[metric_name] = {
            "residualized_arch_eta_sq": res.get("eta_sq", 0.0),
            "f_statistic": res.get("f_statistic", 0.0),
        }

    residualized_eta_sqs = [v["residualized_arch_eta_sq"] for v in variant_results.values()]
    result["variant_robustness"] = {
        "metrics": variant_results,
        "mean_residualized_arch_eta_sq": round(_mean(residualized_eta_sqs), 4),
        "metrics_above_005": sum(1 for e in residualized_eta_sqs if e > 0.05),
        "note": "Architecture eta-sq after residualizing nominal_rank confound",
    }

    return result


# ============================================================================
# Analysis Phase 3: Architecture-vs-task decomposition (one-way ANOVA proxy)
# ============================================================================


def _one_way_anova_eta_sq(groups: dict[str, list[float]]) -> dict[str, Any]:
    """
    Compute one-way ANOVA eta-squared (effect size) for grouped values.
    Pure-Python implementation. Returns eta_sq and F-statistic proxy.
    """
    all_vals: list[float] = []
    for vals in groups.values():
        all_vals.extend(vals)

    if len(all_vals) < 3 or len(groups) < 2:
        return {"eta_sq": 0.0, "n_groups": len(groups), "n_total": len(all_vals), "computable": False}

    grand_mean = _mean(all_vals)

    # SS_between = sum of n_j * (mean_j - grand_mean)^2
    ss_between = 0.0
    for vals in groups.values():
        if vals:
            group_mean = _mean(vals)
            ss_between += len(vals) * (group_mean - grand_mean) ** 2

    # SS_total = sum of (x_i - grand_mean)^2
    ss_total = sum((x - grand_mean) ** 2 for x in all_vals)

    if ss_total == 0:
        return {"eta_sq": 0.0, "n_groups": len(groups), "n_total": len(all_vals), "computable": True}

    eta_sq = ss_between / ss_total

    # F-statistic
    k = len(groups)
    n = len(all_vals)
    ss_within = ss_total - ss_between
    df_between = k - 1
    df_within = n - k

    f_stat = 0.0
    if df_within > 0 and ss_within > 0:
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        f_stat = ms_between / ms_within

    return {
        "eta_sq": round(eta_sq, 4),
        "f_statistic": round(f_stat, 4),
        "n_groups": k,
        "n_total": n,
        "df_between": df_between,
        "df_within": df_within,
        "computable": True,
    }


def phase3_architecture_task_decomposition(fingerprints: list[SpectralFingerprint]) -> dict[str, Any]:
    """Architecture-family and task-category effect sizes via one-way ANOVA."""
    result: dict[str, Any] = {"phase": "architecture_task_decomposition"}

    metrics = [
        "stable_rank_mean",
        "stable_rank_std",
        "utilization_mean",
        "energy_rank_90_p50",
        "entropy_erank_mean",
        "edge_gap_mean",
    ]

    # Architecture effect
    arch_results: dict[str, Any] = {}
    for metric_name in metrics:
        groups: dict[str, list[float]] = {}
        for fp in fingerprints:
            groups.setdefault(fp.architecture_family, []).append(getattr(fp, metric_name))
        arch_results[metric_name] = _one_way_anova_eta_sq(groups)
    result["architecture_effect"] = arch_results

    # Task effect
    task_results: dict[str, Any] = {}
    for metric_name in metrics:
        groups = {}
        for fp in fingerprints:
            groups.setdefault(fp.task_category, []).append(getattr(fp, metric_name))
        task_results[metric_name] = _one_way_anova_eta_sq(groups)
    result["task_effect"] = task_results

    # Summary: which factor has larger eta_sq on average?
    arch_eta_sqs = [v.get("eta_sq", 0.0) for v in arch_results.values() if v.get("computable")]
    task_eta_sqs = [v.get("eta_sq", 0.0) for v in task_results.values() if v.get("computable")]

    result["summary"] = {
        "mean_architecture_eta_sq": round(_mean(arch_eta_sqs), 4) if arch_eta_sqs else 0.0,
        "mean_task_eta_sq": round(_mean(task_eta_sqs), 4) if task_eta_sqs else 0.0,
        "architecture_metrics_above_010": sum(1 for e in arch_eta_sqs if e > 0.10),
        "task_metrics_above_010": sum(1 for e in task_eta_sqs if e > 0.10),
        "dominant_factor": (
            "architecture"
            if (_mean(arch_eta_sqs) if arch_eta_sqs else 0) > (_mean(task_eta_sqs) if task_eta_sqs else 0)
            else "task"
        ),
    }

    return result


# ============================================================================
# Analysis Phase 4: Clustering validation (kNN purity)
# ============================================================================


def _euclidean_dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def phase4_clustering(fingerprints: list[SpectralFingerprint], k: int = 5) -> dict[str, Any]:
    """kNN purity test: do nearest neighbors share architecture/task labels?"""
    result: dict[str, Any] = {"phase": "clustering_validation", "k": k}

    if len(fingerprints) < k + 1:
        result["computable"] = False
        result["reason"] = f"Need at least {k + 1} fingerprints, have {len(fingerprints)}"
        return result

    vectors = [fp.to_vector() for fp in fingerprints]
    arch_labels = [fp.architecture_family for fp in fingerprints]
    task_labels = [fp.task_category for fp in fingerprints]

    arch_purity_scores: list[float] = []
    task_purity_scores: list[float] = []

    for i in range(len(fingerprints)):
        # Compute distances to all other points
        dists: list[tuple[float, int]] = []
        for j in range(len(fingerprints)):
            if i == j:
                continue
            d = _euclidean_dist(vectors[i], vectors[j])
            dists.append((d, j))
        dists.sort()

        # k nearest neighbors
        neighbors = [idx for _, idx in dists[:k]]

        arch_matches = sum(1 for n in neighbors if arch_labels[n] == arch_labels[i])
        task_matches = sum(1 for n in neighbors if task_labels[n] == task_labels[i])

        arch_purity_scores.append(arch_matches / k)
        task_purity_scores.append(task_matches / k)

    result["computable"] = True
    result["architecture_knn_purity"] = {
        "mean": round(_mean(arch_purity_scores), 4),
        "median": round(_median(arch_purity_scores), 4),
        "std": round(_std(arch_purity_scores), 4),
    }
    result["task_knn_purity"] = {
        "mean": round(_mean(task_purity_scores), 4),
        "median": round(_median(task_purity_scores), 4),
        "std": round(_std(task_purity_scores), 4),
    }

    # Baseline: expected purity under random assignment
    arch_counts = {}
    task_counts = {}
    for fp in fingerprints:
        arch_counts[fp.architecture_family] = arch_counts.get(fp.architecture_family, 0) + 1
        task_counts[fp.task_category] = task_counts.get(fp.task_category, 0) + 1
    n = len(fingerprints)
    arch_baseline = sum((c / n) ** 2 for c in arch_counts.values())
    task_baseline = sum((c / n) ** 2 for c in task_counts.values())

    result["baselines"] = {
        "architecture_random_purity": round(arch_baseline, 4),
        "task_random_purity": round(task_baseline, 4),
    }
    result["above_baseline"] = {
        "architecture": round(_mean(arch_purity_scores) - arch_baseline, 4),
        "task": round(_mean(task_purity_scores) - task_baseline, 4),
    }

    return result


# ============================================================================
# Analysis Phase 5: Module-type replication
# ============================================================================


def phase5_module_type_asymmetry(
    fingerprints: list[SpectralFingerprint],
    audit_data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Module-type asymmetry analysis (first-class output).

    Extends the original sign test with:
    - Subtype-level breakdown (q/k/v/o for attention, gate/up/down for MLP)
    - Per-metric asymmetry (stable_rank, utilization, edge_gap)
    - Architectural attribution (does asymmetry pattern differ by family?)
    - Effect-size quantification (mean difference, not just sign direction)
    """
    result: dict[str, Any] = {"phase": "module_type_asymmetry"}

    # --- Part 1: Aggregate sign test (backward-compatible) ---
    pairs: list[dict[str, Any]] = []
    attn_less_count = 0
    mlp_less_count = 0
    ties = 0

    for fp in fingerprints:
        if fp.attn_stable_rank_mean > 0 and fp.mlp_stable_rank_mean > 0:
            diff = fp.attn_stable_rank_mean - fp.mlp_stable_rank_mean
            pairs.append(
                {
                    "repo_id": fp.repo_id,
                    "architecture_family": fp.architecture_family,
                    "attn_sr": round(fp.attn_stable_rank_mean, 4),
                    "mlp_sr": round(fp.mlp_stable_rank_mean, 4),
                    "diff": round(diff, 4),
                }
            )
            if abs(diff) < 0.01:
                ties += 1
            elif diff < 0:
                attn_less_count += 1
            else:
                mlp_less_count += 1

    n_valid = attn_less_count + mlp_less_count
    result["sign_test"] = {
        "n_pairs": len(pairs),
        "n_valid": n_valid,
        "ties": ties,
        "attn_less_than_mlp": attn_less_count,
        "mlp_less_than_attn": mlp_less_count,
        "proportion_attn_less": round(attn_less_count / n_valid, 4) if n_valid > 0 else None,
        "replicates_encoder_pattern": (attn_less_count / n_valid > 0.5) if n_valid > 0 else None,
    }

    # Backward-compat top-level keys
    result["n_pairs"] = len(pairs)
    result["n_valid"] = n_valid
    result["ties"] = ties
    result["attn_less_than_mlp"] = attn_less_count
    result["mlp_less_than_attn"] = mlp_less_count
    if n_valid > 0:
        proportion_attn_less = attn_less_count / n_valid
        result["proportion_attn_less"] = round(proportion_attn_less, 4)
        result["replicates_encoder_pattern"] = proportion_attn_less > 0.5
        result["sign_test_interpretation"] = (
            f"Attention < MLP in {attn_less_count}/{n_valid} adapters ({proportion_attn_less:.0%})"
        )
    else:
        result["proportion_attn_less"] = None
        result["replicates_encoder_pattern"] = None
        result["sign_test_interpretation"] = "Insufficient data for sign test"

    # --- Part 2: By architecture family ---
    by_family: dict[str, dict[str, Any]] = {}
    for fp in fingerprints:
        if fp.attn_stable_rank_mean > 0 and fp.mlp_stable_rank_mean > 0:
            fam = fp.architecture_family
            if fam not in by_family:
                by_family[fam] = {"attn_less": 0, "mlp_less": 0, "ties": 0, "diffs": []}
            diff = fp.attn_stable_rank_mean - fp.mlp_stable_rank_mean
            by_family[fam]["diffs"].append(diff)
            if abs(diff) < 0.01:
                by_family[fam]["ties"] += 1
            elif diff < 0:
                by_family[fam]["attn_less"] += 1
            else:
                by_family[fam]["mlp_less"] += 1

    for _fam, counts in by_family.items():
        total = counts["attn_less"] + counts["mlp_less"]
        counts["proportion_attn_less"] = round(counts["attn_less"] / total, 4) if total > 0 else None
        counts["mean_diff"] = round(_mean(counts["diffs"]), 4)
        counts["direction"] = (
            "attn < MLP" if counts["mean_diff"] < -0.01 else "MLP < attn" if counts["mean_diff"] > 0.01 else "~equal"
        )
        del counts["diffs"]  # Don't serialize raw list
    result["by_architecture_family"] = by_family

    # --- Part 3: Subtype-level breakdown ---
    # Classify layer names into subtypes: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    _ATTN_SUBTYPES = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o"}
    _MLP_SUBTYPES = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}

    subtype_stable_ranks: dict[str, list[float]] = {}
    subtype_utilizations: dict[str, list[float]] = {}
    subtype_by_family: dict[str, dict[str, list[float]]] = {}

    for fp in fingerprints:
        ad = audit_data.get(fp.repo_id, {})
        layer_rows = ad.get("layer_data", {}).get("layer_rows", [])
        for layer in layer_rows:
            layer_name = layer.get("name", "") or layer.get("layer_name", "")
            sr = layer.get("stable_rank")
            ut = layer.get("utilization")
            if sr is None:
                continue

            # Determine subtype
            subtype = None
            for suffix, label in {**_ATTN_SUBTYPES, **_MLP_SUBTYPES}.items():
                if suffix in layer_name:
                    subtype = label
                    break
            if subtype is None:
                mt = layer.get("module_type", "other")
                subtype = f"{mt}_other"

            subtype_stable_ranks.setdefault(subtype, []).append(float(sr))
            if ut is not None:
                subtype_utilizations.setdefault(subtype, []).append(float(ut))

            # Per-family subtype tracking
            fam_key = fp.architecture_family
            if fam_key not in subtype_by_family:
                subtype_by_family[fam_key] = {}
            subtype_by_family[fam_key].setdefault(subtype, []).append(float(sr))

    subtype_summary: dict[str, Any] = {}
    for subtype in sorted(subtype_stable_ranks.keys()):
        sr_vals = subtype_stable_ranks[subtype]
        ut_vals = subtype_utilizations.get(subtype, [])
        subtype_summary[subtype] = {
            "n_layers": len(sr_vals),
            "stable_rank_mean": round(_mean(sr_vals), 4),
            "stable_rank_std": round(_std(sr_vals), 4),
            "utilization_mean": round(_mean(ut_vals), 4) if ut_vals else None,
        }
    result["subtype_breakdown"] = subtype_summary

    # Per-family subtype summary
    family_subtype_summary: dict[str, dict[str, Any]] = {}
    for fam in sorted(subtype_by_family.keys()):
        fam_subtypes: dict[str, Any] = {}
        for subtype in sorted(subtype_by_family[fam].keys()):
            vals = subtype_by_family[fam][subtype]
            fam_subtypes[subtype] = {
                "n_layers": len(vals),
                "stable_rank_mean": round(_mean(vals), 4),
            }
        family_subtype_summary[fam] = fam_subtypes
    result["subtype_by_family"] = family_subtype_summary

    # --- Part 4: Architectural attribution ---
    # Does asymmetry magnitude differ across families?
    family_asymmetry_magnitudes: dict[str, list[float]] = {}
    for fp in fingerprints:
        if fp.attn_stable_rank_mean > 0 and fp.mlp_stable_rank_mean > 0:
            magnitude = abs(fp.attn_stable_rank_mean - fp.mlp_stable_rank_mean)
            family_asymmetry_magnitudes.setdefault(fp.architecture_family, []).append(magnitude)

    arch_attribution: dict[str, Any] = {}
    for fam, mags in sorted(family_asymmetry_magnitudes.items()):
        arch_attribution[fam] = {
            "n": len(mags),
            "mean_asymmetry_magnitude": round(_mean(mags), 4),
            "std": round(_std(mags), 4),
        }

    # ANOVA on asymmetry magnitude by family
    if len(family_asymmetry_magnitudes) >= 2:
        anova_res = _one_way_anova_eta_sq(family_asymmetry_magnitudes)
        arch_attribution["family_effect_eta_sq"] = anova_res.get("eta_sq", 0.0)
        arch_attribution["family_effect_f"] = anova_res.get("f_statistic", 0.0)
    result["architectural_attribution"] = arch_attribution

    return result


def generate_module_type_asymmetry_md(phase5: dict[str, Any]) -> str:
    """Generate module_type_asymmetry.md report."""
    lines = [
        "# Module-Type Asymmetry Analysis",
        "",
        "## Aggregate Sign Test",
        "",
    ]

    st = phase5.get("sign_test", {})
    lines.append(f"- Pairs tested: {st.get('n_pairs', 'N/A')}")
    lines.append(f"- Valid (non-tied): {st.get('n_valid', 'N/A')}")
    lines.append(f"- Attention < MLP: {st.get('attn_less_than_mlp', 'N/A')}")
    lines.append(f"- MLP < Attention: {st.get('mlp_less_than_attn', 'N/A')}")
    prop = st.get("proportion_attn_less")
    lines.append(f"- Proportion attn < MLP: {prop if prop is not None else 'N/A'}")
    lines.append(f"- Replicates encoder pattern: {st.get('replicates_encoder_pattern', 'N/A')}")
    lines.append("")

    # By family
    lines += ["## By Architecture Family", ""]
    by_fam = phase5.get("by_architecture_family", {})
    if by_fam:
        lines += [
            "| Family | Attn<MLP | MLP<Attn | Ties | Prop | Mean diff | Direction |",
            "|--------|---------|---------|------|------|-----------|-----------|",
        ]
        for fam, d in sorted(by_fam.items()):
            lines.append(
                f"| {fam} | {d.get('attn_less', 0)} | {d.get('mlp_less', 0)} | {d.get('ties', 0)} "
                f"| {d.get('proportion_attn_less', 'N/A')} | {d.get('mean_diff', 'N/A')} "
                f"| {d.get('direction', 'N/A')} |"
            )
        lines.append("")

    # Subtype breakdown
    lines += ["## Subtype-Level Breakdown", ""]
    subtypes = phase5.get("subtype_breakdown", {})
    if subtypes:
        lines += [
            "| Subtype | N layers | SR mean | SR std | Util mean |",
            "|---------|----------|---------|--------|-----------|",
        ]
        for sub, d in sorted(subtypes.items()):
            ut = d.get("utilization_mean")
            ut_str = f"{ut:.4f}" if ut is not None else "N/A"
            lines.append(
                f"| {sub} | {d.get('n_layers', 0)} | {d.get('stable_rank_mean', 'N/A')} "
                f"| {d.get('stable_rank_std', 'N/A')} | {ut_str} |"
            )
        lines.append("")

    # Architectural attribution
    lines += ["## Architectural Attribution", ""]
    attr = phase5.get("architectural_attribution", {})
    for fam in sorted(k for k in attr if k not in ("family_effect_eta_sq", "family_effect_f")):
        d = attr[fam]
        lines.append(
            f"- **{fam}**: mean asymmetry magnitude = {d.get('mean_asymmetry_magnitude', 'N/A')} "
            f"(std = {d.get('std', 'N/A')}, n = {d.get('n', 0)})"
        )
    if "family_effect_eta_sq" in attr:
        lines.append(f"- Family effect on asymmetry magnitude: eta-sq = {attr['family_effect_eta_sq']}")
    lines.append("")

    lines += [
        "## Guardrails",
        "",
        "- Subtype labels (q/k/v/o, gate/up/down) inferred from layer name patterns; may not match all architectures.",
        "- Asymmetry direction is an empirical finding, not a causal claim about architecture design.",
        "- Small sample sizes per family limit statistical power for architectural attribution.",
        "",
    ]

    return "\n".join(lines)


# ============================================================================
# Success/failure assessment
# ============================================================================


def assess_outcome(
    phase3: dict[str, Any],
    phase4: dict[str, Any],
    phase5: dict[str, Any],
) -> dict[str, Any]:
    """Classify study outcome as success / partial / negative."""
    summary = phase3.get("summary", {})
    arch_above_010 = summary.get("architecture_metrics_above_010", 0)
    task_above_010 = summary.get("task_metrics_above_010", 0)

    replicates = phase5.get("replicates_encoder_pattern")

    arch_knn = phase4.get("architecture_knn_purity", {}).get("mean", 0)
    arch_baseline = phase4.get("baselines", {}).get("architecture_random_purity", 0)
    arch_above_random = arch_knn > arch_baseline + 0.05

    # Success: arch eta^2 > 0.10 for >=2 metrics, module asymmetry replicates
    if arch_above_010 >= 2 and replicates and arch_above_random:
        outcome = "success"
        interpretation = (
            f"Architecture clustering detected (eta^2 > 0.10 for {arch_above_010} metrics). "
            f"Module-type asymmetry replicates on decoders. "
            f"Task signal is {'present' if task_above_010 > 0 else 'absent'} as secondary effect."
        )
    # Partial: some signal but incomplete
    elif arch_above_010 >= 1 or (replicates is True) or arch_above_random:
        outcome = "partial_success"
        parts = []
        if arch_above_010 >= 1:
            parts.append(f"architecture signal detectable for {arch_above_010} metric(s)")
        if replicates:
            parts.append("module asymmetry replicates")
        if arch_above_random:
            parts.append("kNN purity above random baseline")
        interpretation = "Partial signal: " + "; ".join(parts) + "."
    else:
        outcome = "negative_completion"
        interpretation = (
            "No stable architecture or task signal survives analysis. "
            "Spectral variation likely dominated by confounds. "
            "Controlled GPU study becomes strictly necessary."
        )

    return {
        "outcome": outcome,
        "interpretation": interpretation,
        "metrics": {
            "architecture_metrics_above_010": arch_above_010,
            "task_metrics_above_010": task_above_010,
            "module_asymmetry_replicates": replicates,
            "arch_knn_above_random": arch_above_random,
        },
    }


# ============================================================================
# Report generation
# ============================================================================


def generate_study_memo(
    records: list[CensusAdapterRecord],
    fingerprints: list[SpectralFingerprint],
    phase1: dict[str, Any],
    phase2: dict[str, Any],
    phase3: dict[str, Any],
    phase4: dict[str, Any],
    phase5: dict[str, Any],
    outcome: dict[str, Any],
    pilot_gate: dict[str, Any] | None,
) -> str:
    """Generate interpretive study memo as markdown."""
    audited = [r for r in records if r.status == "audited"]
    excluded = [r for r in records if r.status == "excluded"]
    failed = [r for r in records if r.status == "failed"]

    lines = [
        "# Public Ecosystem Spectral Census -- Study Memo",
        "",
        f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        f"**Outcome**: {outcome['outcome'].replace('_', ' ').title()}",
        "",
        "## Cohort",
        "",
        f"- Adapters attempted: {len(records)}",
        f"- Successfully audited: {len(audited)}",
        f"- Excluded: {len(excluded)}",
        f"- Failed: {len(failed)}",
        f"- Fingerprints extracted: {len(fingerprints)}",
        "",
    ]

    # Architecture/task breakdown
    fam_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    for fp in fingerprints:
        fam_counts[fp.architecture_family] = fam_counts.get(fp.architecture_family, 0) + 1
        task_counts[fp.task_category] = task_counts.get(fp.task_category, 0) + 1

    lines += [
        "### By architecture family",
        "",
        "| Family | Count |",
        "|--------|-------|",
    ]
    for fam, count in sorted(fam_counts.items()):
        lines.append(f"| {fam} | {count} |")
    lines.append("")

    lines += [
        "### By task category",
        "",
        "| Category | Count |",
        "|----------|-------|",
    ]
    for task, count in sorted(task_counts.items()):
        lines.append(f"| {task} | {count} |")
    lines.append("")

    # Confound assessment
    confound_overall = phase2.get("overall", {})
    lines += [
        "## Confound Assessment",
        "",
        f"- Max confound R-squared: {confound_overall.get('max_r_squared', 'N/A')}",
        f"- Residualization recommended: {confound_overall.get('residualization_recommended', 'N/A')}",
        f"- {confound_overall.get('note', '')}",
        "",
    ]

    # Architecture vs task decomposition
    summary = phase3.get("summary", {})
    lines += [
        "## Architecture-vs-Task Decomposition",
        "",
        f"- Mean architecture eta-squared: {summary.get('mean_architecture_eta_sq', 'N/A')}",
        f"- Mean task eta-squared: {summary.get('mean_task_eta_sq', 'N/A')}",
        f"- Architecture metrics with eta-squared > 0.10: {summary.get('architecture_metrics_above_010', 0)}",
        f"- Task metrics with eta-squared > 0.10: {summary.get('task_metrics_above_010', 0)}",
        f"- Dominant factor: {summary.get('dominant_factor', 'N/A')}",
        "",
    ]

    # Per-metric table
    lines += [
        "### Effect sizes by metric",
        "",
        "| Metric | Arch eta-sq | Task eta-sq |",
        "|--------|-------------|-------------|",
    ]
    arch_effects = phase3.get("architecture_effect", {})
    task_effects = phase3.get("task_effect", {})
    for metric in ["stable_rank_mean", "utilization_mean", "energy_rank_90_p50", "entropy_erank_mean", "edge_gap_mean"]:
        a_eta = arch_effects.get(metric, {}).get("eta_sq", "N/A")
        t_eta = task_effects.get(metric, {}).get("eta_sq", "N/A")
        lines.append(f"| {metric} | {a_eta} | {t_eta} |")
    lines.append("")

    # Clustering
    if phase4.get("computable"):
        arch_purity = phase4.get("architecture_knn_purity", {})
        task_purity = phase4.get("task_knn_purity", {})
        baselines = phase4.get("baselines", {})
        lines += [
            "## Clustering Validation (k=5 NN purity)",
            "",
            f"- Architecture purity: {arch_purity.get('mean', 'N/A')} "
            f"(random baseline: {baselines.get('architecture_random_purity', 'N/A')})",
            f"- Task purity: {task_purity.get('mean', 'N/A')} "
            f"(random baseline: {baselines.get('task_random_purity', 'N/A')})",
            "",
        ]

    # Module-type replication
    lines += [
        "## Module-Type Replication",
        "",
        f"- {phase5.get('sign_test_interpretation', 'N/A')}",
        f"- Replicates encoder-era pattern: {phase5.get('replicates_encoder_pattern', 'N/A')}",
        "",
    ]

    # Outcome
    lines += [
        "## Outcome Assessment",
        "",
        f"**{outcome['outcome'].replace('_', ' ').title()}**",
        "",
        outcome["interpretation"],
        "",
        "### Guardrails",
        "",
        "- These are observational findings from found artifacts, not causal claims.",
        "- Confound assessment must be considered before interpreting architecture/task effects.",
        "- Census findings do not replace the controlled GPU-return study.",
        "- Hub download count and popularity are not quality evidence.",
        "",
    ]

    return "\n".join(lines)


def generate_decomposition_summary(
    phase3: dict[str, Any],
    fingerprints: list[SpectralFingerprint],
) -> str:
    """Generate architecture_task_decomposition.md summary."""
    lines = [
        "# Architecture-vs-Task Decomposition",
        "",
        "One-way ANOVA effect sizes (eta-squared) for spectral fingerprint metrics.",
        "",
        "## Architecture Effect",
        "",
        "| Metric | eta-sq | F | n_groups | n_total |",
        "|--------|--------|---|----------|---------|",
    ]
    for metric, vals in sorted(phase3.get("architecture_effect", {}).items()):
        if vals.get("computable"):
            lines.append(
                f"| {metric} | {vals['eta_sq']} | {vals['f_statistic']} | {vals['n_groups']} | {vals['n_total']} |"
            )
    lines.append("")

    lines += [
        "## Task Effect",
        "",
        "| Metric | eta-sq | F | n_groups | n_total |",
        "|--------|--------|---|----------|---------|",
    ]
    for metric, vals in sorted(phase3.get("task_effect", {}).items()):
        if vals.get("computable"):
            lines.append(
                f"| {metric} | {vals['eta_sq']} | {vals['f_statistic']} | {vals['n_groups']} | {vals['n_total']} |"
            )
    lines.append("")

    summary = phase3.get("summary", {})
    lines += [
        "## Summary",
        "",
        f"- Dominant factor: **{summary.get('dominant_factor', 'N/A')}**",
        f"- Mean architecture eta-sq: {summary.get('mean_architecture_eta_sq', 'N/A')}",
        f"- Mean task eta-sq: {summary.get('mean_task_eta_sq', 'N/A')}",
        "",
        "Note: These are observational effect sizes from found artifacts. "
        "They do not establish causal architecture or task effects.",
        "",
    ]

    return "\n".join(lines)


def generate_fingerprint_table_md(fingerprints: list[SpectralFingerprint]) -> str:
    """Generate human-readable fingerprint summary table."""
    lines = [
        "# Spectral Fingerprint Table",
        "",
        "| Adapter | Family | Task | Rank | SR mean | SR std | Util | E90 p50 | Erank | Attn SR | MLP SR | Edge gap |",
        "|---------|--------|------|------|---------|--------|------|---------|-------|---------|--------|----------|",
    ]
    for fp in sorted(fingerprints, key=lambda x: (x.architecture_family, x.repo_id)):
        short_id = fp.repo_id
        if len(short_id) > 40:
            short_id = "..." + short_id[-37:]
        lines.append(
            f"| {short_id} | {fp.architecture_family} | {fp.task_category} "
            f"| {fp.nominal_rank} "
            f"| {fp.stable_rank_mean:.2f} | {fp.stable_rank_std:.2f} "
            f"| {fp.utilization_mean:.3f} | {fp.energy_rank_90_p50:.1f} "
            f"| {fp.entropy_erank_mean:.2f} "
            f"| {fp.attn_stable_rank_mean:.2f} | {fp.mlp_stable_rank_mean:.2f} "
            f"| {fp.edge_gap_mean:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# Main orchestrator
# ============================================================================


def run_census(
    output_dir: Path,
    cache_dir: Path,
    tier: str = "pilot",
    families: list[str] | None = None,
    min_downloads: int = 10,
) -> None:
    """Run the full public ecosystem spectral census."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tier_cfg = TIER_CONFIGS.get(tier, TIER_CONFIGS["pilot"])
    max_total = tier_cfg["max_total"]
    max_per_base = tier_cfg["max_per_base"]

    if families is None:
        # Default: llama, mistral, qwen (core 3)
        families = ["llama", "mistral", "qwen"]

    print("=" * 72)
    print(f"PUBLIC ECOSYSTEM SPECTRAL CENSUS -- Tier: {tier}")
    print(f"Families: {families}")
    print(f"Max adapters: {max_total}, Max per base: {max_per_base}")
    print("=" * 72)

    # Check for existing results (idempotent resume)
    audit_cache_path = output_dir / "audit_cache.json"
    existing_audit_data: dict[str, dict[str, Any]] = {}

    if audit_cache_path.exists():
        print(f"\n  Loading existing audit cache from {audit_cache_path}")
        with open(audit_cache_path) as f:
            existing_audit_data = json.load(f)
        print(f"  {len(existing_audit_data)} adapters already audited")

    # ------------------------------------------------------------------
    # STEP 1: Discovery
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 1: DISCOVER ADAPTERS")
    print("=" * 72)

    records = discover_census_adapters(
        families=families,
        max_per_base=max_per_base,
        max_total=max_total,
        min_downloads=min_downloads,
    )

    if not records:
        print("  No adapters found. Check network connectivity and HuggingFace Hub access.")
        return

    # ------------------------------------------------------------------
    # STEP 2: Download
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 2: DOWNLOAD ADAPTERS")
    print("=" * 72)

    adapter_paths: dict[str, Path] = {}
    for i, record in enumerate(records):
        if record.repo_id in existing_audit_data:
            print(f"  [{i + 1}/{len(records)}] (cached) {record.repo_id}")
            record.status = "audited"
            continue

        print(f"  [{i + 1}/{len(records)}] Downloading: {record.repo_id}")
        try:
            adapter_path, size_mb = download_adapter(record.repo_id, cache_dir)
            record.download_size_mb = size_mb
            record.status = "downloaded"
            adapter_paths[record.repo_id] = adapter_path
            print(f"    OK ({size_mb:.1f} MB)")
        except Exception as e:
            record.status = "failed"
            record.issues.append(f"Download error: {e}")
            print(f"    FAILED: {e}")

    # ------------------------------------------------------------------
    # STEP 3: Audit
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 3: AUDIT ADAPTERS")
    print("=" * 72)

    audit_data: dict[str, dict[str, Any]] = dict(existing_audit_data)
    total_audit_time = 0.0

    for i, record in enumerate(records):
        if record.repo_id in existing_audit_data:
            continue
        if record.status != "downloaded":
            continue

        ap = adapter_paths.get(record.repo_id, cache_dir / record.repo_id.replace("/", "--"))

        print(f"  [{i + 1}/{len(records)}] Auditing: {record.repo_id}")
        result_dict, record = audit_adapter(record.repo_id, ap, record)

        if result_dict is not None:
            audit_data[record.repo_id] = result_dict
            total_audit_time += record.audit_time_s or 0
            print(
                f"    OK: {record.n_layers} layers, rank={record.rank}, "
                f"util={result_dict.get('utilization_mean', 0):.3f}, "
                f"time={record.audit_time_s:.1f}s"
            )
        else:
            reason = record.exclusion_reason or "; ".join(record.issues)
            print(f"    {record.status.upper()}: {reason}")

    # Save audit cache incrementally
    with open(audit_cache_path, "w") as f:
        json.dump(audit_data, f, indent=2, default=str)

    # Fill in metadata for cached records
    for r in records:
        if r.repo_id in existing_audit_data and r.status != "failed":
            data = existing_audit_data[r.repo_id]
            r.status = "audited"
            r.n_layers = data.get("n_layers")
            r.total_params = data.get("total_lora_params")
            layer_rows = data.get("layer_data", {}).get("layer_rows", [])
            if layer_rows:
                r.rank = layer_rows[0].get("r")
                r.alpha = layer_rows[0].get("alpha")

    audited = [r for r in records if r.status == "audited"]
    excluded = [r for r in records if r.status == "excluded"]
    failed = [r for r in records if r.status == "failed"]
    print(f"\n  Audited: {len(audited)}, Excluded: {len(excluded)}, Failed: {len(failed)}")
    print(f"  Total audit time: {total_audit_time:.1f}s")

    if not audit_data:
        print("  No successful audits. Cannot proceed.")
        return

    # ------------------------------------------------------------------
    # STEP 4: Extract fingerprints
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 4: EXTRACT FINGERPRINTS")
    print("=" * 72)

    fingerprints: list[SpectralFingerprint] = []
    for r in records:
        if r.repo_id not in audit_data:
            continue
        fp = extract_fingerprint(r, audit_data[r.repo_id])
        if fp is not None:
            fingerprints.append(fp)
    print(f"  Extracted {len(fingerprints)} fingerprints")

    if len(fingerprints) < 5:
        print("  Too few fingerprints for analysis. Stopping.")
        return

    # ------------------------------------------------------------------
    # STEP 5: Analysis (Phases 1-5)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 5: ANALYSIS")
    print("=" * 72)

    print("  Phase 1: Distributional characterization...")
    p1 = phase1_distributional(fingerprints)

    print("  Phase 2: Confound assessment...")
    p2 = phase2_confound_assessment(fingerprints)
    overall_confound = p2.get("overall", {})
    print(f"    Max confound R^2: {overall_confound.get('max_r_squared', 'N/A')}")

    print("  Phase 2b: Analytic control subsets...")
    p2b = phase2b_analytic_controls(fingerprints)
    rm = p2b.get("rank_matched_subset", {})
    print(f"    Rank-matched subset: n={rm.get('n_matched', 0)} (modal rank={rm.get('modal_rank', '?')})")
    vr = p2b.get("variant_robustness", {})
    print(f"    Residualized arch eta^2: {vr.get('mean_residualized_arch_eta_sq', 'N/A')}")
    print(f"    Metrics with residualized eta^2 > 0.05: {vr.get('metrics_above_005', 0)}")

    print("  Phase 3: Architecture-vs-task decomposition...")
    p3 = phase3_architecture_task_decomposition(fingerprints)
    summary = p3.get("summary", {})
    print(f"    Mean arch eta^2: {summary.get('mean_architecture_eta_sq', 'N/A')}")
    print(f"    Mean task eta^2: {summary.get('mean_task_eta_sq', 'N/A')}")

    print("  Phase 4: Clustering validation...")
    p4 = phase4_clustering(fingerprints, k=5)
    if p4.get("computable"):
        arch_pur = p4.get("architecture_knn_purity", {}).get("mean", "N/A")
        task_pur = p4.get("task_knn_purity", {}).get("mean", "N/A")
        print(f"    Arch kNN purity: {arch_pur}")
        print(f"    Task kNN purity: {task_pur}")

    print("  Phase 5: Module-type asymmetry analysis...")
    p5 = phase5_module_type_asymmetry(fingerprints, audit_data)
    print(f"    {p5.get('sign_test_interpretation', 'N/A')}")
    subtype_count = len(p5.get("subtype_breakdown", {}))
    print(f"    Subtypes identified: {subtype_count}")

    # Outcome assessment
    outcome = assess_outcome(p3, p4, p5)
    print(f"\n  Outcome: {outcome['outcome']}")
    print(f"  {outcome['interpretation']}")

    # Pilot gate (if pilot tier)
    pilot_gate = None
    pilot_plus_gate = None
    if tier == "pilot":
        print("\n  Evaluating pilot validation gate...")
        pilot_gate = evaluate_pilot_gate(records, fingerprints)
        print(f"    Overall: {'PASS' if pilot_gate['overall_pass'] else 'FAIL'}")
        for crit_name, crit in pilot_gate.get("criteria", {}).items():
            status = "PASS" if crit.get("pass") else "FAIL"
            print(f"    {crit_name}: {status}")

        print("\n  Evaluating pilot-plus gate...")
        pilot_plus_gate = evaluate_pilot_plus_gate(records, fingerprints, p2b, p5)
        print(f"    Overall: {'PASS' if pilot_plus_gate['overall_pass'] else 'FAIL'}")
        for crit_name, crit in pilot_plus_gate.get("criteria", {}).items():
            status = "PASS" if crit.get("pass") else "FAIL"
            print(f"    {crit_name}: {status}")

    # ------------------------------------------------------------------
    # STEP 6: Save outputs
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 6: SAVE OUTPUTS")
    print("=" * 72)

    # manifest.json
    manifest = {
        "study": "public_ecosystem_spectral_census",
        "date": datetime.now(timezone.utc).isoformat(),
        "tier": tier,
        "families": families,
        "n_discovered": len(records),
        "n_audited": len(audited),
        "n_excluded": len(excluded),
        "n_failed": len(failed),
        "n_fingerprints": len(fingerprints),
        "outcome": outcome["outcome"],
    }
    _write_json(output_dir / "manifest.json", manifest)

    # adapter_records.json
    adapter_records_out = [
        {
            "repo_id": r.repo_id,
            "base_model": r.base_model,
            "architecture_family": r.architecture_family,
            "inferred_task": r.inferred_task,
            "task_category": r.task_category,
            "rank": r.rank,
            "alpha": r.alpha,
            "lora_dropout": r.lora_dropout,
            "target_modules": r.target_modules,
            "peft_type": r.peft_type,
            "downloads": r.downloads,
            "n_layers": r.n_layers,
            "total_params": r.total_params,
            "download_size_mb": r.download_size_mb,
            "audit_time_s": r.audit_time_s,
            "status": r.status,
            "issues": r.issues,
            "exclusion_reason": r.exclusion_reason,
        }
        for r in records
    ]
    _write_json(output_dir / "adapter_records.json", adapter_records_out)

    # fingerprint_table.json
    fp_table = [
        {
            "repo_id": fp.repo_id,
            "architecture_family": fp.architecture_family,
            "task_category": fp.task_category,
            "nominal_rank": fp.nominal_rank,
            "vector": fp.to_vector(),
            "dimensions": SpectralFingerprint.dimension_names(),
            **{name: getattr(fp, name) for name in SpectralFingerprint.dimension_names()},
        }
        for fp in fingerprints
    ]
    _write_json(output_dir / "fingerprint_table.json", fp_table)

    # fingerprint_table.md
    fp_md = generate_fingerprint_table_md(fingerprints)
    _write_text(output_dir / "fingerprint_table.md", fp_md)

    # confound_assessment.json
    _write_json(output_dir / "confound_assessment.json", p2)

    # architecture_task_decomposition.json
    _write_json(output_dir / "architecture_task_decomposition.json", p3)

    # architecture_task_decomposition.md
    decomp_md = generate_decomposition_summary(p3, fingerprints)
    _write_text(output_dir / "architecture_task_decomposition.md", decomp_md)

    # clustering_results.json
    _write_json(output_dir / "clustering_results.json", p4)

    # module_type_replication.json (backward-compat name)
    _write_json(output_dir / "module_type_replication.json", p5)

    # Phase 2b deliverables
    _write_json(output_dir / "rank_matched_analysis.json", p2b.get("rank_matched_subset", {}))
    _write_json(output_dir / "task_label_confidence.json", p2b.get("task_label_confidence_subset", {}))
    _write_json(output_dir / "variant_robustness.json", p2b.get("variant_robustness", {}))

    # Module-type asymmetry markdown report
    asymmetry_md = generate_module_type_asymmetry_md(p5)
    _write_text(output_dir / "module_type_asymmetry.md", asymmetry_md)

    # excluded_adapters.json
    excluded_records = [
        {
            "repo_id": r.repo_id,
            "base_model": r.base_model,
            "status": r.status,
            "exclusion_reason": r.exclusion_reason,
            "issues": r.issues,
        }
        for r in records
        if r.status in ("excluded", "failed")
    ]
    _write_json(output_dir / "excluded_adapters.json", excluded_records)

    # pilot_gate_report.md (if pilot)
    if pilot_gate is not None:
        _write_json(output_dir / "pilot_gate_report.json", pilot_gate)
        gate_md = _generate_pilot_gate_md(pilot_gate)
        _write_text(output_dir / "pilot_gate_report.md", gate_md)

    # pilot_plus_gate_report (if pilot)
    if pilot_plus_gate is not None:
        _write_json(output_dir / "pilot_plus_gate_report.json", pilot_plus_gate)
        pp_md = _generate_pilot_plus_gate_md(pilot_plus_gate)
        _write_text(output_dir / "pilot_plus_gate_report.md", pp_md)

    # study_memo.md
    memo = generate_study_memo(records, fingerprints, p1, p2, p3, p4, p5, outcome, pilot_gate)
    _write_text(output_dir / "study_memo.md", memo)

    print("\n  All outputs written to:", output_dir)
    print("\n" + "=" * 72)
    print("CENSUS COMPLETE")
    print("=" * 72)


# ============================================================================
# File I/O helpers
# ============================================================================


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"    {path.name}")


def _write_text(path: Path, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)
    print(f"    {path.name}")


def _generate_pilot_gate_md(gate: dict[str, Any]) -> str:
    lines = [
        "# Pilot Validation Gate Report",
        "",
        f"**Overall**: {'PASS' if gate['overall_pass'] else 'FAIL'}",
        f"**Recommendation**: {gate['recommendation']}",
        "",
    ]
    for name, crit in gate.get("criteria", {}).items():
        status = "PASS" if crit.get("pass") else "FAIL"
        lines.append(f"## {name}: {status}")
        lines.append("")
        for k, v in crit.items():
            if k == "pass":
                continue
            lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def _generate_pilot_plus_gate_md(gate: dict[str, Any]) -> str:
    lines = [
        "# Pilot-Plus Validation Gate Report",
        "",
        f"**Overall**: {'PASS' if gate['overall_pass'] else 'FAIL'}",
        f"**Recommendation**: {gate['recommendation']}",
        "",
    ]
    for name, crit in gate.get("criteria", {}).items():
        status = "PASS" if crit.get("pass") else "FAIL"
        lines.append(f"## {name}: {status}")
        lines.append("")
        for k, v in crit.items():
            if k == "pass":
                continue
            lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Public Ecosystem Spectral Census -- CPU-only decoder adapter survey")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("field_trials/public_ecosystem_census"),
        help="Directory for census outputs (default: field_trials/public_ecosystem_census)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/census_adapters"),
        help="Directory to cache downloaded adapters (default: .cache/census_adapters)",
    )
    parser.add_argument(
        "--tier",
        choices=["pilot", "core", "extended"],
        default="pilot",
        help="Cohort tier (default: pilot)",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Architecture families to include (default: llama mistral qwen)",
    )
    parser.add_argument(
        "--min-downloads",
        type=int,
        default=10,
        help="Minimum downloads to consider an adapter (default: 10)",
    )
    args = parser.parse_args()

    run_census(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        tier=args.tier,
        families=args.families,
        min_downloads=args.min_downloads,
    )


if __name__ == "__main__":
    main()
