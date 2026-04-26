"""Configuration loading, canonicalization, and hashing.

Implements the config system per SPEC_CPU_v0_2.md §4. The canonical
config hash ties every condition manifest row to a specific config state
and is the primary reproducibility anchor for the pipeline.

Key public functions:
    load_config(study_config_path)   → MergedConfig
    canonicalize(merged_dict)        → str (canonical YAML)
    compute_config_hash(merged_dict) → (full_hex, short_hex)

Key exception types (per SPEC §11.2):
    ConfigValidationError
    ConfigHashMismatchError
    FewshotLeakageError
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# -- Exception types per SPEC §11.2 ------------------------------------------


class ConfigValidationError(Exception):
    """Raised when config validation fails (structural or semantic)."""


class ConfigHashMismatchError(Exception):
    """Raised when a stored config hash does not match the computed hash."""


class FewshotLeakageError(Exception):
    """Raised when few-shot exemplar IDs overlap the evaluation split.

    The leakage check (SPEC §3.7, §8.3) is a hard acceptance criterion:
    any intersection between the drawn fewshot pool and the eval-split
    item set for the same benchmark aborts the pipeline with exit code 3.
    """


class SchemaValidationError(Exception):
    """Raised when a raw artifact fails JSON Schema validation (SPEC §11.2).

    Used by ``04_normalize_outputs.py`` when ``run_metadata.json``,
    ``item_outputs.jsonl`` rows, or ``item_scores.jsonl`` rows do not
    conform to their declared schema. Triggers exit code 3 (data
    contract violation) per SPEC §11.1.
    """


class ConditionCountMismatch(Exception):
    """Raised when the raw run's item count disagrees with the manifest.

    The manifest's ``expected_num_items`` is the contract; if the raw
    JSONL or the run metadata reports a different number, the run is
    rejected (no silent truncation, no silent padding).
    """


class PartialRunCompletion(Exception):
    """Raised when ``num_items_completed != num_items_expected`` in a run.

    Partial completion is treated as a hard failure by the normalizer:
    the GPU-side backend must mark partial runs as ``status="partial"``
    or ``"failed"``, not as ``"complete"``. A complete-status run with
    a partial item count is a contract violation.
    """


class ConvergenceFailureAllLevels(Exception):
    """Raised by ``06_variance_components.py`` when every level of the
    pre-registered mixed-effects cascade (§10.1) fails to converge and
    even the aggregate-score G-theory fallback is unavailable.

    This is a strict signal that the statistical core could not produce
    variance components — distinct from the softer Level-4 fallback,
    which succeeds via aggregate G-theory without item-level mixed
    effects. Exit code 4 per SPEC §11.1.
    """


# -- Dataclasses for typed config access -------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    hf_name: str
    hf_revision: str
    model_family: str
    model_type: str  # "base" | "instruct"
    parameter_count_b: float
    primary: bool
    optional_extension: bool


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    display_name: str
    tier: str  # "primary" | "secondary"
    hf_dataset: str
    hf_config: str | None
    dataset_version_hash: str
    task_type: str  # "constrained_choice" | "open_generation"
    fewshot_k: int
    seed_facet: bool
    fewshot_source_split: str | None
    eval_split: str
    scoring_rules: tuple[str, ...]
    item_id_field: str
    answer_field: str
    expected_num_items: int | None = None
    subjects: tuple[str, ...] | None = None
    expected_num_items_per_subject: int | None = None


@dataclass(frozen=True)
class PromptSpec:
    benchmark_id: str
    prompt_id: str
    prompt_path: str
    content_hash: str
    source_type: str
    source_url_or_ref: str
    source_commit: str | None
    admissibility_status: str  # "locked" | "draft"
    placeholders_used: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class ScoringRuleSpec:
    scoring_rule_id: str
    display_name: str
    applies_to_task_types: tuple[str, ...]
    normalization: str | None = None
    selection_rule: str | None = None
    generation_params: dict[str, Any] | None = None
    parse_strategy: str | None = None


@dataclass(frozen=True)
class BootstrapConfig:
    n_resamples: int
    random_seed: int
    ci_lower_percentile: float
    ci_upper_percentile: float


@dataclass(frozen=True)
class MixedEffectsCascadeConfig:
    levels: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ConvergenceTriggersConfig:
    max_singular_warning: str
    hessian_non_pd: str
    iteration_limit_exceeded: str
    gradient_norm_above: float


@dataclass(frozen=True)
class ToleranceConfig:
    decision_rule_threshold_h1: float
    benchmarks_required_for_h1: int
    # v1.1.2: cells with median parseability_rate below this threshold
    # are routed to sample-SD-based tolerance (D-07 pattern) instead of
    # variance-components decomposition. See D-09 v1.1.2 resolution.
    parse_failure_threshold: float = 0.30


@dataclass(frozen=True)
class AnalysisConfig:
    bootstrap: BootstrapConfig
    mixed_effects_cascade: MixedEffectsCascadeConfig
    convergence_triggers: ConvergenceTriggersConfig
    tolerance: ToleranceConfig
    h2_generalizability_threshold: float
    h3_ranking_reversal_threshold: float
    h4_mmlu_interaction_threshold: float


@dataclass(frozen=True)
class StudyConfig:
    study_id: str
    prereg_version: str
    tier_primary_enabled: bool
    tier_secondary_enabled: bool
    extension_7b_enabled: bool
    paths: dict[str, str]
    includes: tuple[str, ...]


@dataclass(frozen=True)
class MergedConfig:
    """All six YAML files merged into a single typed object.

    The `raw_dict` field is the merged dict in the exact form used for
    canonicalization and hashing. Typed fields above are derived views for
    convenience; do not rehydrate them back into YAML and rehash — use
    `raw_dict` as the canonical source.
    """

    study_config: StudyConfig
    models: tuple[ModelSpec, ...]
    benchmarks: tuple[BenchmarkSpec, ...]
    prompts: tuple[PromptSpec, ...]
    scoring_rules: tuple[ScoringRuleSpec, ...]
    analysis_config: AnalysisConfig
    raw_dict: dict[str, Any]


# -- Canonicalization and hashing per SPEC §4.4 ------------------------------


def canonicalize(merged: dict[str, Any]) -> str:
    """Canonical YAML serialization for hashing.

    Deterministic: keys sorted, block style, no line wrapping, UTF-8.
    Any bit-level change to the input dict produces a different output
    string. This is the preimage for compute_config_hash().
    """
    return yaml.safe_dump(
        merged,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        width=10**9,  # prevent line-wrapping differences
    )


def compute_config_hash(merged: dict[str, Any]) -> tuple[str, str]:
    """Compute the config hash per SPEC §4.4.

    Returns:
        (full, short) where full is the 64-char SHA-256 hex digest and
        short is the first 8 chars for human-readable use in condition
        manifests.
    """
    canonical = canonicalize(merged).encode("utf-8")
    full = hashlib.sha256(canonical).hexdigest()
    return full, full[:8]


# -- Loading -----------------------------------------------------------------


_INCLUDE_FILENAME_TO_SECTION: dict[str, str] = {
    "models.yaml": "models",
    "benchmarks.yaml": "benchmarks",
    "prompts.yaml": "prompts",
    "scoring_rules.yaml": "scoring_rules",
    "analysis_config.yaml": "analysis_config",
}


def load_raw_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dict. Raises FileNotFoundError if missing."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(study_config_path: Path | str) -> MergedConfig:
    """Load and merge all six config files into a typed MergedConfig.

    The study_config.yaml's `includes` field lists paths (relative to the
    paper-root directory, i.e. the parent of the `configs/` directory)
    to the other five config files. All six are merged into the
    MergedConfig.raw_dict under the fixed section keys defined by
    _INCLUDE_FILENAME_TO_SECTION.

    Raises:
        FileNotFoundError: if any config file is missing.
        ConfigValidationError: if structural requirements are violated
            (unknown include filename, missing section in merged dict).
        KeyError: if a required field is missing from a loaded section.
    """
    study_config_path = Path(study_config_path)
    study_dict = load_raw_yaml(study_config_path)

    # configs/study_config.yaml lives in <paper_root>/configs/, so paper_root
    # is the parent of the configs directory.
    paper_root = study_config_path.parent.parent

    merged_raw: dict[str, Any] = {"study_config": study_dict}
    for include_path in study_dict.get("includes", []):
        resolved = paper_root / include_path
        if not resolved.exists():
            raise ConfigValidationError(
                f"Include not found at {resolved} (declared as '{include_path}')"
            )
        filename = Path(include_path).name
        section_key = _INCLUDE_FILENAME_TO_SECTION.get(filename)
        if section_key is None:
            raise ConfigValidationError(
                f"Unknown include file: {filename}. "
                f"Expected one of: {sorted(_INCLUDE_FILENAME_TO_SECTION)}"
            )
        merged_raw[section_key] = load_raw_yaml(resolved)

    # Verify all expected sections are present.
    for section in _INCLUDE_FILENAME_TO_SECTION.values():
        if section not in merged_raw:
            raise ConfigValidationError(
                f"Section '{section}' missing from merged config. "
                f"Check study_config.yaml's 'includes' list."
            )

    return MergedConfig(
        study_config=_build_study_config(study_dict),
        models=tuple(_build_model_spec(m) for m in merged_raw["models"]["models"]),
        benchmarks=tuple(
            _build_benchmark_spec(b) for b in merged_raw["benchmarks"]["benchmarks"]
        ),
        prompts=tuple(_build_prompt_spec(p) for p in merged_raw["prompts"]["prompts"]),
        scoring_rules=tuple(
            _build_scoring_rule_spec(s)
            for s in merged_raw["scoring_rules"]["scoring_rules"]
        ),
        analysis_config=_build_analysis_config(merged_raw["analysis_config"]),
        raw_dict=merged_raw,
    )


def _build_study_config(d: dict[str, Any]) -> StudyConfig:
    return StudyConfig(
        study_id=d["study_id"],
        prereg_version=d["prereg_version"],
        tier_primary_enabled=d["tier_primary_enabled"],
        tier_secondary_enabled=d["tier_secondary_enabled"],
        extension_7b_enabled=d.get("extension_7b_enabled", False),
        paths=dict(d["paths"]),
        includes=tuple(d["includes"]),
    )


def _build_model_spec(d: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        model_id=d["model_id"],
        hf_name=d["hf_name"],
        hf_revision=d["hf_revision"],
        model_family=d["model_family"],
        model_type=d["model_type"],
        parameter_count_b=float(d["parameter_count_b"]),
        primary=d["primary"],
        optional_extension=d["optional_extension"],
    )


def _build_benchmark_spec(d: dict[str, Any]) -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id=d["benchmark_id"],
        display_name=d["display_name"],
        tier=d["tier"],
        hf_dataset=d["hf_dataset"],
        hf_config=d.get("hf_config"),
        dataset_version_hash=d["dataset_version_hash"],
        task_type=d["task_type"],
        fewshot_k=d["fewshot_k"],
        seed_facet=d["seed_facet"],
        fewshot_source_split=d.get("fewshot_source_split"),
        eval_split=d["eval_split"],
        scoring_rules=tuple(d["scoring_rules"]),
        item_id_field=d["item_id_field"],
        answer_field=d["answer_field"],
        expected_num_items=d.get("expected_num_items"),
        subjects=tuple(d["subjects"]) if d.get("subjects") else None,
        expected_num_items_per_subject=d.get("expected_num_items_per_subject"),
    )


def _build_prompt_spec(d: dict[str, Any]) -> PromptSpec:
    return PromptSpec(
        benchmark_id=d["benchmark_id"],
        prompt_id=d["prompt_id"],
        prompt_path=d["prompt_path"],
        content_hash=d["content_hash"],
        source_type=d["source_type"],
        source_url_or_ref=d["source_url_or_ref"],
        source_commit=d.get("source_commit"),
        admissibility_status=d["admissibility_status"],
        placeholders_used=tuple(d["placeholders_used"]),
        notes=d.get("notes", ""),
    )


def _build_scoring_rule_spec(d: dict[str, Any]) -> ScoringRuleSpec:
    return ScoringRuleSpec(
        scoring_rule_id=d["scoring_rule_id"],
        display_name=d["display_name"],
        applies_to_task_types=tuple(d["applies_to_task_types"]),
        normalization=d.get("normalization"),
        selection_rule=d.get("selection_rule"),
        generation_params=d.get("generation_params"),
        parse_strategy=d.get("parse_strategy"),
    )


def _build_analysis_config(d: dict[str, Any]) -> AnalysisConfig:
    boot = d["bootstrap"]
    cascade = d["mixed_effects_cascade"]
    trig = d["convergence_triggers"]
    tol = d["tolerance"]
    return AnalysisConfig(
        bootstrap=BootstrapConfig(
            n_resamples=int(boot["n_resamples"]),
            random_seed=int(boot["random_seed"]),
            ci_lower_percentile=float(boot["ci_lower_percentile"]),
            ci_upper_percentile=float(boot["ci_upper_percentile"]),
        ),
        mixed_effects_cascade=MixedEffectsCascadeConfig(levels=dict(cascade)),
        convergence_triggers=ConvergenceTriggersConfig(
            max_singular_warning=trig["max_singular_warning"],
            hessian_non_pd=trig["hessian_non_pd"],
            iteration_limit_exceeded=trig["iteration_limit_exceeded"],
            gradient_norm_above=float(trig["gradient_norm_above"]),
        ),
        tolerance=ToleranceConfig(
            decision_rule_threshold_h1=float(tol["decision_rule_threshold_h1"]),
            benchmarks_required_for_h1=int(tol["benchmarks_required_for_h1"]),
            parse_failure_threshold=float(tol.get("parse_failure_threshold", 0.30)),
        ),
        h2_generalizability_threshold=float(d["h2_generalizability_threshold"]),
        h3_ranking_reversal_threshold=float(d["h3_ranking_reversal_threshold"]),
        h4_mmlu_interaction_threshold=float(d["h4_mmlu_interaction_threshold"]),
    )
