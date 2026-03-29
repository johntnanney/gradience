"""
gradience.api

Thin, stable Python wrappers around Gradience's *stable surfaces*.

Design goals:
- Avoid entangling internal modules (e.g., gradience.bench.protocol).
- Prefer calling the canonical CLI/module entrypoints for reproducibility.
- Return paths to canonical artifacts (bench.json, bench_aggregate.json, etc.).
- Keep advanced helpers explicit and opt-in (diagnostic-first, non-default).

Stability policy:
- This module is part of Gradience's "Stable (public API)" tier.
- If you need programmatic access, use this instead of importing internals.

Note:
- These helpers are intentionally minimal. They orchestrate runs; they don't
  re-implement Bench/Audit logic.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
    from gradience.vnext.inventory.neighborhoods import MergeNeighborhoodReport
    from gradience.vnext.inventory.summary import InventorySummary
    from gradience.vnext.merge.qa_report import MergeQAReport

logger = logging.getLogger(__name__)


# -----------------------------
# Data containers (lightweight)
# -----------------------------


@dataclass(frozen=True)
class BenchRunArtifacts:
    """Paths produced by a single `run_bench` invocation."""

    output_dir: Path
    bench_json: Path
    bench_md: Path


@dataclass(frozen=True)
class BenchAggregateArtifacts:
    """Paths produced by `aggregate_bench_runs`."""

    output_dir: Path
    aggregate_json: Path
    aggregate_md: Path


@dataclass(frozen=True)
class AuditResult:
    """Result from running ``gradience audit``."""

    returncode: int
    log_path: Path | None = None

    @property
    def success(self) -> bool:
        """True when the audit process exited cleanly."""
        return self.returncode == 0


@dataclass(frozen=True)
class MonitorResult:
    """Result from running ``gradience monitor``."""

    returncode: int
    log_path: Path | None = None

    @property
    def success(self) -> bool:
        """True when the monitor process exited cleanly."""
        return self.returncode == 0


# -----------------------------
# Helpers
# -----------------------------


def _pyexe(python: str | None = None) -> str:
    """Resolve which Python executable to use."""
    return python or sys.executable


def _run(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    log_path: Path | None = None,
) -> int:
    """
    Run a command with optional logging, returning the exit code.

    If *log_path* is provided, stdout/stderr are written to that file.
    Otherwise, the process inherits the current stdout/stderr.
    """
    logger.debug("Running command: %s", argv)
    merged_env = os.environ.copy()
    if env:
        merged_env.update({k: str(v) for k, v in env.items()})

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            proc = subprocess.run(
                list(argv),
                cwd=str(cwd) if cwd else None,
                env=merged_env,
                text=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=check,
            )
            return proc.returncode

    proc = subprocess.run(
        list(argv),
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
        check=check,
    )
    return proc.returncode


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


def _load_schema_artifacts(
    *,
    files: Sequence[Path],
    schema_prefix: str,
    from_dict: Any,
    strict_input: bool,
    malformed_label: str,
) -> list[Any]:
    """Load typed artifacts from a path list with schema-prefixed filtering."""
    import sys

    loaded: list[Any] = []
    for fp in files:
        try:
            data = _read_json(fp)
        except Exception:
            if strict_input:
                raise
            print(f"WARNING: skipping unreadable file: {fp}", file=sys.stderr)
            continue
        schema = data.get("schema", "")
        if not isinstance(schema, str) or not schema.startswith(schema_prefix):
            continue
        try:
            loaded.append(from_dict(data))
        except Exception:
            if strict_input:
                raise
            print(f"WARNING: skipping malformed {malformed_label}: {fp}", file=sys.stderr)
    return loaded


# -----------------------------
# Public API: Bench
# -----------------------------


def run_bench(
    *,
    config: str | Path,
    output: str | Path,
    smoke: bool = False,
    ci: bool = False,
    python: str | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    check: bool = True,
) -> BenchRunArtifacts:
    """
    Run the Bench protocol using the stable module entrypoint.

    Equivalent to:
      python -m gradience.bench.run_bench --config <yaml> --output <dir> [--smoke] [--ci]
    """
    logger.debug("Starting bench run: config=%s output=%s smoke=%s ci=%s", config, output, smoke, ci)
    config_p = Path(config)
    output_p = Path(output)
    output_p.mkdir(parents=True, exist_ok=True)

    argv = [
        _pyexe(python),
        "-m",
        "gradience.bench.run_bench",
        "--config",
        str(config_p),
        "--output",
        str(output_p),
    ]
    if smoke:
        argv.append("--smoke")
    if ci:
        argv.append("--ci")

    _run(
        argv,
        env=env,
        check=check,
        log_path=Path(log_path) if log_path else None,
    )

    return BenchRunArtifacts(
        output_dir=output_p,
        bench_json=output_p / "bench.json",
        bench_md=output_p / "bench.md",
    )


def aggregate_bench_runs(
    *,
    runs: Sequence[str | Path],
    output: str | Path,
    include_smoke: bool = False,
    python: str | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    check: bool = True,
) -> BenchAggregateArtifacts:
    """
    Aggregate multiple Bench runs using the stable module entrypoint.

    Equivalent to:
      python -m gradience.bench.aggregate <run_dir>... --output <dir> [--include-smoke]
    """
    logger.debug("Aggregating %d bench runs to %s", len(runs), output)
    run_paths = [Path(r) for r in runs]
    output_p = Path(output)
    output_p.mkdir(parents=True, exist_ok=True)

    argv = [_pyexe(python), "-m", "gradience.bench.aggregate"]
    argv.extend(str(p) for p in run_paths)
    argv.extend(["--output", str(output_p)])
    if include_smoke:
        argv.append("--include-smoke")

    _run(
        argv,
        env=env,
        check=check,
        log_path=Path(log_path) if log_path else None,
    )

    return BenchAggregateArtifacts(
        output_dir=output_p,
        aggregate_json=output_p / "bench_aggregate.json",
        aggregate_md=output_p / "bench_aggregate.md",
    )


# -----------------------------
# Public API: Audit + Monitor
# -----------------------------


def audit(
    *,
    peft_dir: str | Path,
    layers: bool = True,
    base_model: str | None = None,
    base_norms_cache: str | Path | None = None,
    no_udr: bool = False,
    extra_args: Sequence[str] | None = None,
    python: str | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    check: bool = True,
) -> AuditResult:
    """
    Run `gradience audit` via the stable CLI entrypoint.

    Equivalent to (typical):
      python -m gradience audit --peft-dir <dir> --layers [--base-model ...] [...]

    We intentionally treat `gradience` CLI as the stable surface and avoid importing internals.
    """
    logger.debug("Running audit: peft_dir=%s", peft_dir)
    peft_p = Path(peft_dir)

    argv = [
        _pyexe(python),
        "-m",
        "gradience",
        "audit",
        "--peft-dir",
        str(peft_p),
    ]
    if layers:
        argv.append("--layers")
    if base_model:
        argv.extend(["--base-model", base_model])
    if base_norms_cache:
        argv.extend(["--base-norms-cache", str(Path(base_norms_cache))])
    if no_udr:
        argv.append("--no-udr")
    if extra_args:
        argv.extend(list(extra_args))

    log_p = Path(log_path) if log_path else None
    returncode = _run(argv, env=env, check=check, log_path=log_p)
    return AuditResult(returncode=returncode, log_path=log_p)


def monitor(
    *,
    run_jsonl: str | Path,
    verbose: bool = False,
    extra_args: Sequence[str] | None = None,
    python: str | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    check: bool = True,
) -> MonitorResult:
    """
    Run `gradience monitor` via the stable CLI entrypoint.

    Equivalent to:
      python -m gradience monitor <run.jsonl> [--verbose] [...]
    """
    logger.debug("Running monitor: run_jsonl=%s", run_jsonl)
    run_p = Path(run_jsonl)

    argv = [
        _pyexe(python),
        "-m",
        "gradience",
        "monitor",
        str(run_p),
    ]
    if verbose:
        argv.append("--verbose")
    if extra_args:
        argv.extend(list(extra_args))

    log_p = Path(log_path) if log_path else None
    returncode = _run(argv, env=env, check=check, log_path=log_p)
    return MonitorResult(returncode=returncode, log_path=log_p)


def audit_adapter(
    *,
    peft_dir: str | Path,
    base_model: str | None = None,
    adapter_score: float | None = None,
    base_score: float | None = None,
    metric_name: str | None = None,
    lower_is_better: bool = True,
    eval_dataset: str | None = None,
    margin: float = 0.0,
    notes: list[str] | None = None,
    adapter_config_path: str | Path | None = None,
    adapter_weights_path: str | Path | None = None,
    base_norms_cache: str | Path | None = None,
    compute_udr: bool = True,
) -> AdapterQAArtifact:
    """Produce an AdapterQAArtifact from a PEFT adapter directory.

    This is the preferred stable Python entry point for producing adapter
    QA artifacts.  It runs the structural audit internally and builds the
    artifact through the same policy path as the CLI.

    Parameters
    ----------
    peft_dir
        Path to the PEFT adapter directory.
    base_model
        Base model identifier (e.g. ``meta-llama/Llama-2-7b-hf``).
    adapter_score, base_score
        Behavioral evaluation scores.  Both must be provided for
        behavioral eligibility classification.
    metric_name
        Name of the evaluation metric.
    lower_is_better
        True for metrics like perplexity where lower = better.
    eval_dataset
        Dataset used for evaluation.
    margin
        Tolerance margin for eligibility classification (symmetric).
    notes
        Optional list of caveats or annotations.
    adapter_config_path, adapter_weights_path
        Override default adapter config/weights file detection.
    base_norms_cache
        Path to cached base model norms for UDR computation.
    compute_udr
        Whether to compute Update Dominance Ratio metrics.

    Returns
    -------
    AdapterQAArtifact
        The canonical QA artifact for this adapter.
    """
    from gradience.vnext.audit import audit_lora_peft_dir
    from gradience.vnext.audit.qa_artifact import build_qa_artifact

    result = audit_lora_peft_dir(
        str(peft_dir),
        adapter_config_path=str(adapter_config_path) if adapter_config_path else None,
        adapter_weights_path=str(adapter_weights_path) if adapter_weights_path else None,
        map_location="cpu",
        base_model_id=base_model,
        base_norms_cache=str(base_norms_cache) if base_norms_cache else None,
        compute_udr=compute_udr,
    )

    return build_qa_artifact(
        result,
        adapter_path=str(peft_dir),
        base_model=base_model or "",
        adapter_score=adapter_score,
        base_score=base_score,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
        eval_dataset=eval_dataset,
        margin=margin,
        notes=notes,
    )


# -----------------------------
# Convenience: load canonical artifacts
# -----------------------------


def merge_risk_report(
    *,
    adapter_a: str | Path,
    adapter_b: str | Path,
    source_a_qa: str | Path | None = None,
    source_b_qa: str | Path | None = None,
    thresholds: str = "default",
    compute_core_space: bool = False,
    python: str | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    check: bool = True,
) -> MergeQAReport:
    """Run merge-audit and return the pair-level MergeQAReport.

    This is the stable Python wrapper for the ``merge-audit --qa-report
    --emit-report`` workflow. It delegates report generation to the CLI,
    then loads the resulting JSON as a ``MergeQAReport``.

    Parameters
    ----------
    adapter_a, adapter_b
        Paths to the two PEFT adapter directories.
    source_a_qa, source_b_qa
        Optional paths to adapter QA artifact JSON files.
    thresholds
        Threshold preset: ``"default"``, ``"conservative"``, or ``"permissive"``.
    compute_core_space
        When ``True``, enable optional shared-basis diagnostics and include
        the optional ``core_space`` block in the loaded ``MergeQAReport``.

    Returns
    -------
    MergeQAReport
        The canonical pair-level merge risk report.
    """
    import tempfile

    adapter_a_p = Path(adapter_a)
    adapter_b_p = Path(adapter_b)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        emit_path = Path(tmp.name)

    try:
        argv = [
            _pyexe(python),
            "-m",
            "gradience",
            "merge-audit",
            "--adapter-a",
            str(adapter_a_p),
            "--adapter-b",
            str(adapter_b_p),
            "--qa-report",
            "--emit-report",
            str(emit_path),
            "--thresholds",
            thresholds,
        ]
        if source_a_qa:
            argv.extend(["--source-a-qa", str(Path(source_a_qa))])
        if source_b_qa:
            argv.extend(["--source-b-qa", str(Path(source_b_qa))])
        if compute_core_space:
            argv.append("--compute-core-space")

        _run(
            argv,
            env=env,
            check=check,
            log_path=Path(log_path) if log_path else None,
        )

        report_data = _read_json(emit_path)
        from gradience.vnext.merge.qa_report import MergeQAReport as _MergeQAReport

        return _MergeQAReport.from_dict(report_data)
    finally:
        emit_path.unlink(missing_ok=True)


def compute_core_space_diagnostic(
    *,
    adapter_a: str | Path,
    adapter_b: str | Path,
    source_a_qa: str | Path | None = None,
    source_b_qa: str | Path | None = None,
    thresholds: str = "default",
    python: str | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    check: bool = True,
) -> Any:
    """Compute the advanced optional core-space diagnostic for one adapter pair.

    This is a thin wrapper over :func:`merge_risk_report` with
    ``compute_core_space=True``. It returns the report's ``core_space``
    section as a typed object when available.

    Notes
    -----
    - Diagnostic-only: this call does not alter default merge recommendation
      behavior.
    - Optional advanced tier: keep usage explicit in ambiguous pair analysis.
    """
    report = merge_risk_report(
        adapter_a=adapter_a,
        adapter_b=adapter_b,
        source_a_qa=source_a_qa,
        source_b_qa=source_b_qa,
        thresholds=thresholds,
        compute_core_space=True,
        python=python,
        env=env,
        log_path=log_path,
        check=check,
    )
    core_space = getattr(report, "core_space", None)
    if core_space is None:
        raise RuntimeError("Expected core_space diagnostic block, but none was returned")
    return core_space


def summarize_inventory(
    *,
    qa_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    qa_paths: list[str | Path] | None = None,
    report_paths: list[str | Path] | None = None,
    strict_input: bool = False,
) -> InventorySummary:
    """Build an :class:`~gradience.vnext.inventory.summary.InventorySummary`.

    Scans directories and/or explicit file paths for adapter QA artifacts
    and merge risk reports, then aggregates them into a single summary.

    Parameters
    ----------
    qa_dir
        Directory to glob for ``*.json`` adapter QA artifacts.
    report_dir
        Directory to glob for ``*.json`` merge risk reports.
    qa_paths
        Explicit paths to adapter QA artifact JSON files.
    report_paths
        Explicit paths to merge risk report JSON files.
    strict_input
        When ``False`` (default), malformed files are skipped with a
        warning to stderr.  When ``True``, exceptions from ``from_dict()``
        are re-raised immediately.

    Returns
    -------
    InventorySummary
        The aggregated inventory summary.

    Raises
    ------
    ValueError
        If no input source is provided.
    """
    from gradience.vnext.audit.qa_artifact import AdapterQAArtifact as _AdapterQAArtifact
    from gradience.vnext.inventory.summary import build_inventory_summary as _build_inventory_summary
    from gradience.vnext.merge.qa_report import MergeQAReport as _MergeQAReport

    if qa_dir is None and report_dir is None and not qa_paths and not report_paths:
        raise ValueError("At least one of qa_dir, report_dir, qa_paths, or report_paths must be provided")

    # --- Collect file paths ---
    qa_files: list[Path] = []
    report_files: list[Path] = []

    if qa_dir is not None:
        qa_files.extend(sorted(Path(qa_dir).glob("*.json")))
    if qa_paths:
        qa_files.extend(Path(p) for p in qa_paths)

    if report_dir is not None:
        report_files.extend(sorted(Path(report_dir).glob("*.json")))
    if report_paths:
        report_files.extend(Path(p) for p in report_paths)

    # --- Load artifacts ---
    artifacts = _load_schema_artifacts(
        files=qa_files,
        schema_prefix="gradience.adapter_qa/",
        from_dict=_AdapterQAArtifact.from_dict,
        strict_input=strict_input,
        malformed_label="QA artifact",
    )

    reports = _load_schema_artifacts(
        files=report_files,
        schema_prefix="gradience.merge_qa_report/",
        from_dict=_MergeQAReport.from_dict,
        strict_input=strict_input,
        malformed_label="merge report",
    )

    return _build_inventory_summary(artifacts, reports)


def suggest_neighborhoods(
    *,
    qa_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    qa_paths: list[str | Path] | None = None,
    report_paths: list[str | Path] | None = None,
    strict_input: bool = False,
    strict_qa: bool = False,
    min_compatibility: float = 0.0,
    exclude_unknown: bool = False,
) -> MergeNeighborhoodReport:
    """Build a ``MergeNeighborhoodReport`` from QA artifacts and pair reports.

    This is the advanced inventory workflow wrapper for neighborhood
    suggestions. It is additive and diagnostic-first; it does not alter any
    default merge recommendation logic.
    """
    from gradience.vnext.audit.qa_artifact import AdapterQAArtifact as _AdapterQAArtifact
    from gradience.vnext.inventory.neighborhoods import suggest_merge_neighborhoods as _suggest_merge_neighborhoods
    from gradience.vnext.merge.qa_report import MergeQAReport as _MergeQAReport

    if report_dir is None and not report_paths:
        raise ValueError("At least one of report_dir or report_paths must be provided")

    qa_files: list[Path] = []
    report_files: list[Path] = []

    if qa_dir is not None:
        qa_files.extend(sorted(Path(qa_dir).glob("*.json")))
    if qa_paths:
        qa_files.extend(Path(p) for p in qa_paths)

    if report_dir is not None:
        report_files.extend(sorted(Path(report_dir).glob("*.json")))
    if report_paths:
        report_files.extend(Path(p) for p in report_paths)

    artifacts = _load_schema_artifacts(
        files=qa_files,
        schema_prefix="gradience.adapter_qa/",
        from_dict=_AdapterQAArtifact.from_dict,
        strict_input=strict_input,
        malformed_label="QA artifact",
    )
    reports = _load_schema_artifacts(
        files=report_files,
        schema_prefix="gradience.merge_qa_report/",
        from_dict=_MergeQAReport.from_dict,
        strict_input=strict_input,
        malformed_label="merge report",
    )
    if not reports:
        raise ValueError("No valid merge reports were found from the provided input")

    return _suggest_merge_neighborhoods(
        artifacts,
        reports,
        strict_qa=strict_qa,
        min_compatibility=min_compatibility,
        exclude_unknown=exclude_unknown,
    )


def scan_portfolio(
    directory: str | Path,
    strict_input: bool = False,
) -> Any:
    """Scan *directory* for inventory summary artifacts and return a portfolio view.

    Thin stable wrapper around
    :func:`gradience.vnext.inventory.portfolio.scan_portfolio`.

    Parameters
    ----------
    directory
        Root directory to scan for ``gradience.inventory_summary/v1`` JSON files.
    strict_input
        When ``True``, re-raises on malformed artifacts instead of warning.

    Returns
    -------
    PortfolioView
    """
    from gradience.vnext.inventory.portfolio import scan_portfolio as _scan_portfolio

    return _scan_portfolio(directory, strict_input=strict_input)


# -----------------------------
# Convenience: load canonical artifacts
# -----------------------------


def load_bench_report(output_dir: str | Path) -> dict[str, Any]:
    """Load <output_dir>/bench.json."""
    logger.debug("Loading bench report from %s", output_dir)
    p = Path(output_dir) / "bench.json"
    return _read_json(p)


def load_bench_aggregate(output_dir: str | Path) -> dict[str, Any]:
    """Load <output_dir>/bench_aggregate.json."""
    logger.debug("Loading bench aggregate from %s", output_dir)
    p = Path(output_dir) / "bench_aggregate.json"
    return _read_json(p)
