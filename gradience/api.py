"""
gradience.api

Thin, stable Python wrappers around Gradience's *stable surfaces*.

Design goals:
- Avoid entangling internal modules (e.g., gradience.bench.protocol).
- Prefer calling the canonical CLI/module entrypoints for reproducibility.
- Return paths to canonical artifacts (bench.json, bench_aggregate.json, etc.).

Stability policy:
- This module is part of Gradience's "Stable (public API)" tier.
- If you need programmatic access, use this instead of importing internals.

Note:
- These helpers are intentionally minimal. They orchestrate runs; they don't
  re-implement Bench/Audit logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Optional
import json
import os
import subprocess
import sys


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


# -----------------------------
# Helpers
# -----------------------------

def _pyexe(python: Optional[str] = None) -> str:
    """Resolve which Python executable to use."""
    return python or sys.executable


def _run(
    argv: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    check: bool = True,
    log_path: Optional[Path] = None,
) -> subprocess.CompletedProcess[str]:
    """
    Run a command with optional logging.

    If log_path is provided, stdout/stderr are written to that file.
    Otherwise, the process inherits the current stdout/stderr.
    """
    merged_env = os.environ.copy()
    if env:
        merged_env.update({k: str(v) for k, v in env.items()})

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            return subprocess.run(
                list(argv),
                cwd=str(cwd) if cwd else None,
                env=merged_env,
                text=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=check,
            )

    return subprocess.run(
        list(argv),
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        check=check,
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Public API: Bench
# -----------------------------

def run_bench(
    *,
    config: str | Path,
    output: str | Path,
    smoke: bool = False,
    ci: bool = False,
    python: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    log_path: Optional[str | Path] = None,
    check: bool = True,
) -> BenchRunArtifacts:
    """
    Run the Bench protocol using the stable module entrypoint.

    Equivalent to:
      python -m gradience.bench.run_bench --config <yaml> --output <dir> [--smoke] [--ci]
    """
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
    python: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    log_path: Optional[str | Path] = None,
    check: bool = True,
) -> BenchAggregateArtifacts:
    """
    Aggregate multiple Bench runs using the stable module entrypoint.

    Equivalent to:
      python -m gradience.bench.aggregate <run_dir>... --output <dir> [--include-smoke]
    """
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
    base_model: Optional[str] = None,
    base_norms_cache: Optional[str | Path] = None,
    no_udr: bool = False,
    extra_args: Optional[Sequence[str]] = None,
    python: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    log_path: Optional[str | Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """
    Run `gradience audit` via the stable CLI entrypoint.

    Equivalent to (typical):
      python -m gradience audit --peft-dir <dir> --layers [--base-model ...] [...]

    We intentionally treat `gradience` CLI as the stable surface and avoid importing internals.
    """
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

    return _run(
        argv,
        env=env,
        check=check,
        log_path=Path(log_path) if log_path else None,
    )


def monitor(
    *,
    run_jsonl: str | Path,
    verbose: bool = False,
    extra_args: Optional[Sequence[str]] = None,
    python: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    log_path: Optional[str | Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """
    Run `gradience monitor` via the stable CLI entrypoint.

    Equivalent to:
      python -m gradience monitor <run.jsonl> [--verbose] [...]
    """
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

    return _run(
        argv,
        env=env,
        check=check,
        log_path=Path(log_path) if log_path else None,
    )


# -----------------------------
# Convenience: load canonical artifacts
# -----------------------------

def load_bench_report(output_dir: str | Path) -> dict[str, Any]:
    """Load <output_dir>/bench.json."""
    p = Path(output_dir) / "bench.json"
    return _read_json(p)


def load_bench_aggregate(output_dir: str | Path) -> dict[str, Any]:
    """Load <output_dir>/bench_aggregate.json."""
    p = Path(output_dir) / "bench_aggregate.json"
    return _read_json(p)