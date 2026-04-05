"""Shared CLI utilities used across command modules.

Contains config loading, normalization, formatting helpers, and
analysis functions shared by multiple command handlers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gradience.exceptions import (
    ConfigError,
    QASchemaError,
)

# ---------------------------------------------------------------------------
# Config pipeline
# ---------------------------------------------------------------------------


def _load_config_file(path: str) -> dict[str, Any]:
    """Load a JSON/YAML config file into a dict."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()

    # Prefer explicit extension
    if suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        # Try JSON first, then YAML
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            import yaml

            data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping/object at the top level. Got: {type(data).__name__}")

    return data


def _autodetect_file_in_dir(
    dir_path: str,
    *,
    candidates: list[str],
    label: str,
) -> str:
    """Auto-detect a config file inside a directory.

    We first check for exact filenames at the directory root (common HF/PEFT
    outputs). If not found, we fall back to a recursive search for any of the
    candidate filenames.

    Args:
        dir_path: Directory containing the file.
        candidates: Filenames to look for, in priority order.
        label: Human-friendly label for error messages.

    Returns:
        The detected file path as a string.

    Raises:
        FileNotFoundError: if no candidate file is found.
        NotADirectoryError: if dir_path is not a directory.
        ValueError: if multiple candidates are found in recursive search.
    """

    p = Path(dir_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if not p.is_dir():
        raise NotADirectoryError(str(p))

    # Fast path: expected files at directory root.
    for name in candidates:
        f = p / name
        if f.exists() and f.is_file():
            return str(f)

    # Fallback: recursive search (in case caller points at a run folder).
    matches: list[Path] = []
    for name in candidates:
        matches.extend(list(p.rglob(name)))

    # De-duplicate (rglob can return duplicates on some filesystems)
    matches = sorted(set(matches))

    if not matches:
        tried = ", ".join(candidates)
        raise FileNotFoundError(f"No {label} file found in '{p}'. Tried: {tried}")

    if len(matches) > 1:
        # If multiple found, prefer the shallowest path.
        matches_sorted = sorted(matches, key=lambda m: (len(m.parts), str(m)))
        best = matches_sorted[0]
        # If there is ambiguity at the same depth, force the user to be explicit.
        same_depth = [m for m in matches_sorted if len(m.parts) == len(best.parts)]
        if len(same_depth) > 1:
            rendered = "\n".join([f"  - {m}" for m in same_depth[:10]])
            more = "" if len(same_depth) <= 10 else f"\n  ... and {len(same_depth) - 10} more"
            raise ValueError(
                f"Multiple {label} files found in '{p}'. Please pass an explicit path.\n"
                f"Candidates at same depth:\n{rendered}{more}"
            )
        return str(best)

    return str(matches[0])


def _first(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _parse_targets(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        out: list[str] = []
        for item in v:
            if item is None:
                continue
            s = str(item).strip()
            if not s:
                continue
            # split comma-separated values inside lists
            if "," in s:
                out.extend([t.strip() for t in s.split(",") if t.strip()])
            else:
                out.append(s)
        return out
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        # space-separated
        if " " in s:
            return [t.strip() for t in s.split() if t.strip()]
        return [s]
    return []


def _normalize_to_vnext_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Best-effort normalization of common config formats to vNext ConfigSnapshot dict."""

    # If it already looks like canonical vNext, just ensure nested dicts exist.
    if any(k in raw for k in ("optimizer", "lora", "training", "task_profile")):
        out = dict(raw)
        out["optimizer"] = dict(out.get("optimizer") or {})
        out["lora"] = dict(out.get("lora") or {})
        out["training"] = dict(out.get("training") or {})
        return out

    # Otherwise treat as "flat" or PEFT-like.
    d: dict[str, Any] = {
        "model_name": _first(raw, ["model_name", "model", "base_model", "base_model_name_or_path"]),
        "dataset_name": _first(raw, ["dataset_name", "dataset", "task", "data"]),
        "task_profile": _first(raw, ["task_profile", "taskProfile", "profile"]),
        "optimizer": {},
        "lora": {},
        "training": {},
        "extras": {"source_format": "flat_or_peft"},
    }

    # Optimizer-ish
    opt = d["optimizer"]
    opt["name"] = _first(raw, ["optimizer", "optim", "optimizer_name", "optim_name"])
    opt["lr"] = _first(raw, ["lr", "learning_rate", "learningRate"])
    opt["weight_decay"] = _first(raw, ["weight_decay", "wd", "weightDecay"])

    # LoRA-ish (PEFT adapter_config.json keys)
    lora = d["lora"]
    lora["r"] = _first(raw, ["r", "rank", "lora_r"])
    lora["alpha"] = _first(raw, ["alpha", "lora_alpha", "loraAlpha"])
    lora["target_modules"] = _parse_targets(_first(raw, ["target_modules", "targets", "modules"]))
    lora["dropout"] = _first(raw, ["lora_dropout", "dropout"])
    lora["bias"] = _first(raw, ["bias"])

    # Training-ish
    tr = d["training"]
    tr["seed"] = _first(raw, ["seed"])
    tr["batch_size"] = _first(raw, ["batch_size", "per_device_train_batch_size"])
    tr["gradient_accumulation"] = _first(raw, ["gradient_accumulation", "gradient_accumulation_steps"])
    tr["max_steps"] = _first(raw, ["max_steps"])
    tr["epochs"] = _first(raw, ["epochs", "num_train_epochs"])
    tr["dtype"] = _first(raw, ["dtype", "torch_dtype"])

    # Keep original keys for debugging
    d["extras"]["raw_keys"] = sorted(list(raw.keys()))

    return d


def _blank_vnext_dict() -> dict[str, Any]:
    """Return an empty (canonical-ish) vNext config dict.

    We keep this as a plain dict so we can merge multiple source files
    (canonical config, PEFT adapter_config.json, HF training_args.json) before
    constructing a :class:`~gradience.vnext.types.ConfigSnapshot`.
    """

    return {
        "model_name": None,
        "dataset_name": None,
        "task_profile": None,
        "optimizer": {},
        "lora": {},
        "training": {},
        "notes": None,
        "extras": {},
    }


def _merge_fill_missing(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge `overlay` into `base`, filling only missing/empty values.

    Precedence rule:
      * base (positional CONFIG) wins
      * `--peft` and `--training` fill gaps
      * CLI overrides apply last (handled separately)

    This keeps behavior predictable: explicit configs aren't silently overwritten
    by convenience wrapper files.
    """

    def _is_missing(v: Any) -> bool:
        return v is None or v == "" or v == [] or v == {}

    for k, v in (overlay or {}).items():
        if v is None:
            continue

        if k in ("optimizer", "lora", "training"):
            base.setdefault(k, {})
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if vv is None:
                        continue
                    if kk == "extras" and isinstance(base[k].get(kk), dict) and isinstance(vv, dict):
                        # merge extras dicts
                        base[k][kk].update(vv)
                        continue
                    if kk not in base[k] or _is_missing(base[k].get(kk)):
                        base[k][kk] = vv
            continue

        if k == "extras":
            base.setdefault("extras", {})
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if vv is None:
                        continue
                    if isinstance(base["extras"].get(kk), dict) and isinstance(vv, dict):
                        base["extras"][kk].update(vv)
                    elif kk not in base["extras"] or _is_missing(base["extras"].get(kk)):
                        base["extras"][kk] = vv
            continue

        if k not in base or _is_missing(base.get(k)):
            base[k] = v

    return base


def _apply_overrides(d: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply CLI overrides onto the normalized vNext dict."""

    if args.model is not None:
        d["model_name"] = args.model
    if args.dataset is not None:
        d["dataset_name"] = args.dataset
    if args.task_profile is not None:
        d["task_profile"] = args.task_profile
    if args.notes is not None:
        d["notes"] = args.notes

    opt = d.setdefault("optimizer", {})
    if args.optimizer is not None:
        opt["name"] = args.optimizer
    if args.lr is not None:
        opt["lr"] = args.lr
    if args.weight_decay is not None:
        opt["weight_decay"] = args.weight_decay

    lora = d.setdefault("lora", {})
    if args.r is not None:
        lora["r"] = args.r
    if args.alpha is not None:
        lora["alpha"] = args.alpha
    if args.targets:
        # Allow comma-separated inside args.targets too.
        merged: list[str] = []
        for t in args.targets:
            merged.extend(_parse_targets(t))
        lora["target_modules"] = merged



# ---------------------------------------------------------------------------
# Formatting utilities — canonical definitions in cli_format.py
# ---------------------------------------------------------------------------
from gradience.cli_format import _fmt as _fmt  # noqa: F401, E402
from gradience.cli_format import _fmt_params as _fmt_params  # noqa: F401, E402
from gradience.cli_format import _severity_rank as _severity_rank  # noqa: F401, E402

# ---------------------------------------------------------------------------
# QA loading
# ---------------------------------------------------------------------------


def _load_source_qa(path_str: str | None) -> Any:
    """Load an AdapterQAResult from a JSON file path, or return None.

    Three-way routing:
    - schema key present: strict v1 loader (AdapterQAArtifact.from_dict)
    - schema key absent: legacy flat-format parser (AdapterQAResult.from_dict)
    - schema key present but wrong: hard fail, no fallback to legacy
    """
    if path_str is None:
        return None
    import json as jsonlib

    p = Path(path_str)
    if not p.is_file():
        raise ConfigError(f"--source-*-qa path does not exist: {p}")
    try:
        with open(p) as f:
            data = jsonlib.load(f)
    except Exception as e:
        raise QASchemaError(f"Failed to parse QA file {p}: {e}") from e

    # Three-way routing based on schema key presence
    schema_key = data.get("schema") if isinstance(data, dict) else None

    if schema_key is not None:
        # Schema present — must go through strict v1 loader
        try:
            from gradience.vnext.audit.qa_artifact import AdapterQAArtifact

            return AdapterQAArtifact.from_dict(data).to_qa_result()
        except Exception as e:
            raise QASchemaError(f"Invalid QA artifact {p}: {e}") from e

    # Schema absent — legacy flat format
    try:
        from gradience.vnext.merge.eligibility import AdapterQAResult

        return AdapterQAResult.from_dict(data)
    except Exception as e:
        raise QASchemaError(f"Failed to load legacy QA file {p}: {e}") from e



# ---------------------------------------------------------------------------
# Rank policy parsing
# ---------------------------------------------------------------------------


def _parse_rank_policies(policies_arg: str | None) -> list[str] | None:
    """Parse user-friendly rank policy names to internal policy names."""
    if not policies_arg:
        return None

    # Handle both comma-separated and space-separated
    if "," in policies_arg:
        policies = [p.strip() for p in policies_arg.split(",")]
    else:
        policies = policies_arg.split()

    # Map user-friendly names to internal policy names
    parsed_policies = []
    for policy in policies:
        if policy in ["energy@0.90", "energy@0.95"]:
            parsed_policies.append("energy_threshold")
        elif policy == "knee":
            parsed_policies.append("knee_elbow")
        elif policy == "erank":
            parsed_policies.append("entropy_effective")
        elif policy == "oht":
            parsed_policies.append("optimal_hard_threshold")
        elif policy in ["energy_threshold", "knee_elbow", "entropy_effective", "optimal_hard_threshold"]:
            # Support internal names directly
            parsed_policies.append(policy)
        else:
            print(f"Warning: Unknown policy '{policy}'. Available: energy@0.90, energy@0.95, knee, erank, oht")

    return parsed_policies if parsed_policies else None



# ---------------------------------------------------------------------------
# Version info
# ---------------------------------------------------------------------------


def _get_version_info():
    """Extract Gradience version and git SHA if available."""
    version_info = {}

    try:
        # Try to get package version (prefer importlib.metadata over deprecated pkg_resources)
        try:
            from importlib.metadata import PackageNotFoundError, version

            version_info["gradience_version"] = version("gradience")
        except (ImportError, PackageNotFoundError):
            try:
                import pkg_resources

                version_info["gradience_version"] = pkg_resources.get_distribution("gradience").version
            except Exception:  # Intentionally broad: pkg_resources exceptions vary by platform
                version_info["gradience_version"] = "development"
    except Exception:  # Intentionally broad: outermost fallback for version detection
        version_info["gradience_version"] = "unknown"

    # Try to get git SHA
    try:
        import os
        import subprocess

        # Look for git in the current directory
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        version_info["git_sha"] = git_sha[:12]  # Short SHA
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        version_info["git_sha"] = "unknown"

    return version_info



# ---------------------------------------------------------------------------
# Guard activity extraction
# ---------------------------------------------------------------------------


def _extract_guard_activity(reader: Any) -> dict[str, Any]:
    """Extract Guard activity summary from telemetry."""
    guard_info = {
        "present": False,
        "last_action": None,
        "rollback_count": 0,
        "snapshot_count": 0,
        "memory_mb": 0.0,
        "last_trigger_code": None,
        "aborted": False,
        "rollback_occurred": False,
    }

    try:
        # Check for Guard alerts
        for event in reader.iter_events(event_type="alert"):
            code = event.get("code", "")
            if code.startswith("GUARD_"):
                guard_info["present"] = True

                if code == "GUARD_TRIGGERED":
                    guard_info["last_trigger_code"] = code
                elif code == "GUARD_ROLLBACK":
                    guard_info["rollback_occurred"] = True
                elif code in ("GUARD_ABORT", "GUARD_ABORT_NO_SNAPSHOT"):
                    guard_info["aborted"] = True

        # Check Guard metrics for latest state and rollback count
        for event in reader.iter_events(event_type="metrics"):
            if event.get("kind") == "guard":
                guard_info["present"] = True
                metrics = event.get("metrics", {})
                action = metrics.get("action")

                if action:
                    guard_info["last_action"] = action

                # Track rollback count from any metrics (rollback or abort can have n_rollbacks)
                if "n_rollbacks" in metrics:
                    guard_info["rollback_count"] = max(guard_info["rollback_count"], metrics.get("n_rollbacks", 0))

                # Latest snapshot info
                if "snapshot_count" in metrics:
                    guard_info["snapshot_count"] = metrics["snapshot_count"]
                if "memory_mb" in metrics:
                    guard_info["memory_mb"] = metrics["memory_mb"]

        # If we found any rollback count > 0, mark rollback as occurred
        if guard_info["rollback_count"] is not None and guard_info["rollback_count"] > 0:
            guard_info["rollback_occurred"] = True

    except Exception:  # Intentionally broad: guard info is best-effort diagnostic
        # If anything fails, return minimal guard_info
        pass

    return guard_info



# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _analyze_policy_disagreements(
    layers: list[Any],
    name_mapping: dict[str, str],
    importance_config: dict[str, Any] | None = None,
    rationale_verbosity: str = "flagged_only",
) -> dict[str, Any]:
    """Analyze policy disagreements and return structured data for JSON output.

    Args:
        layers: List of layer objects with rank suggestions
        name_mapping: Mapping of internal policy names to user-friendly names
        importance_config: Configuration for importance thresholds
        rationale_verbosity: "full" or "flagged_only" - controls detail level for non-flagged layers

    Returns detailed flagging rationale for each layer, suitable for machine consumption.
    """
    from gradience.policy_analysis import (
        compute_energy_distribution,
        compute_layer_importance_scores,
    )

    # Extract importance configuration with defaults
    if importance_config is None:
        importance_config = {}

    quantile_threshold = importance_config.get("quantile_threshold", 0.75)
    min_uniform_mult = importance_config.get("uniform_mult_gate", 1.5)
    importance_metric = importance_config.get("metric", "energy_share")

    if not layers:
        return {
            "schema_version": 1,
            "computed_with": _get_version_info(),
            "analysis_performed": False,
            "reason": "no_layers",
            "layers": [],
        }

    # Shared computation steps 1-3
    layer_analysis, importance_scores = compute_layer_importance_scores(layers, name_mapping, importance_config)

    if not layer_analysis:
        return {
            "schema_version": 1,
            "computed_with": _get_version_info(),
            "analysis_performed": False,
            "reason": "no_disagreements",
            "layers": [],
        }

    total_energy, uniform_share, max_uniform_mult, distribution_is_flat = compute_energy_distribution(
        layer_analysis, min_uniform_mult
    )

    n_layers = len(layer_analysis)
    quantile_pct = quantile_threshold * 100

    # Step 5: Apply smart filtering and generate flagging rationale
    # (JSON output needs detailed per-layer rationale, so we build it here
    #  rather than using filter_disagreement_layers directly)
    flagged_layers = []
    all_layers = []

    for analysis in layer_analysis:
        layer_name = analysis["layer_name"]
        energy_share = analysis["energy_share"]
        uniform_mult = analysis["uniform_mult"]
        policy_spread = analysis["policy_spread"]
        max_k = analysis["max_k"]

        # Check spread filter
        spread_threshold = max(3, 0.5 * max_k)
        meets_spread_threshold = policy_spread >= spread_threshold

        # Calculate priority score for Bench ordering
        spread_norm = max(0.0, policy_spread / spread_threshold) if spread_threshold > 0 else 0.0
        priority_score = spread_norm * uniform_mult

        # Build flagging rationale - full version first (will be condensed later if needed)
        flagging_rationale = {
            "spread": int(policy_spread),
            "spread_threshold": float(spread_threshold),
            "meets_spread_threshold": bool(meets_spread_threshold),
            "importance_share": float(energy_share),
            "uniform_mult": float(uniform_mult),
            "uniform_mult_threshold": float(min_uniform_mult),
            "meets_uniform_mult_threshold": bool(uniform_mult >= min_uniform_mult),
            "priority_score": float(priority_score),
            "is_flat_distribution": bool(distribution_is_flat),
            "quantile_threshold": float(quantile_threshold),
            "meets_quantile_threshold": False,
            "passed_gate": False,
            "flagged_as_high_impact": False,
            "k_values": [int(k) for k in analysis["k_values"]],
            "policies": analysis["policies"],
        }

        layer_data = {"layer_name": layer_name, "flagging_rationale": flagging_rationale}

        # Only consider layers with significant disagreement
        if meets_spread_threshold:
            if not distribution_is_flat:
                import numpy as np  # lazy: only needed for percentile in this branch

                energy_shares = [a["energy_share"] for a in layer_analysis]
                energy_quantile = np.percentile(energy_shares, quantile_pct)
                meets_quantile_threshold = energy_share >= energy_quantile
                flagging_rationale["meets_quantile_threshold"] = bool(meets_quantile_threshold)
                flagging_rationale["energy_quantile_threshold"] = float(energy_quantile)

                passed_gate = meets_quantile_threshold and uniform_mult >= min_uniform_mult
                flagging_rationale["passed_gate"] = bool(passed_gate)
                flagging_rationale["flagged_as_high_impact"] = bool(passed_gate)

                if passed_gate:
                    flagged_layers.append(layer_data)
            else:
                flagging_rationale["meets_quantile_threshold"] = None
                flagging_rationale["passed_gate"] = False
                flagging_rationale["flagged_as_high_impact"] = False

        # Condense rationale for non-flagged layers if verbosity is "flagged_only"
        if rationale_verbosity == "flagged_only" and not flagging_rationale.get("flagged_as_high_impact", False):
            failed_reasons = []
            if not meets_spread_threshold:
                failed_reasons.append("insufficient_spread")
            if flagging_rationale.get("meets_quantile_threshold") is False:
                failed_reasons.append("below_quantile_threshold")
            if not flagging_rationale.get("meets_uniform_mult_threshold", False):
                failed_reasons.append("below_uniform_mult_threshold")
            if distribution_is_flat:
                failed_reasons.append("flat_distribution")

            flagging_rationale = {
                "spread": int(policy_spread),
                "importance_share": float(energy_share),
                "uniform_mult": float(uniform_mult),
                "priority_score": float(priority_score),
                "failed_reasons": failed_reasons,
            }
            layer_data["flagging_rationale"] = flagging_rationale

        all_layers.append(layer_data)

    # Get version and timestamp information
    version_info = _get_version_info()

    import time

    timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Return structured analysis with complete schema
    return {
        "schema_version": 1,
        "computed_at": timestamp_iso,
        "computed_with": version_info,
        "disagreement_config": {
            "quantile_threshold": float(quantile_threshold),
            "uniform_mult_gate": float(min_uniform_mult),
            "importance_metric": importance_metric,
            "quantile_pct": float(quantile_pct),
            "min_uniform_mult_threshold": float(min_uniform_mult),
            "spread_base_threshold": 3,
            "spread_dynamic_factor": 0.5,
            "spread_calculation_formula": "max(3, 0.5 * max_k_for_layer)",
            "flat_detection_enabled": True,
            "spread_filter_enabled": True,
            "quantile_filter_enabled": True,
            "config_capture_version": "1.0",
            "algorithm_name": "energy_share_uniform_multiplier_gate",
        },
        # Backward compatibility alias for older test expectations
        "config": {
            "quantile_threshold": float(quantile_threshold),
            "uniform_mult_gate": float(min_uniform_mult),
            "importance_metric": importance_metric,
        },
        "analysis_performed": True,
        "distribution": {
            "total_layers": int(n_layers),
            "total_energy": float(total_energy),
            "max_uniform_mult": float(max_uniform_mult),
            "is_flat": bool(distribution_is_flat),
            "uniform_share": float(uniform_share),
            "flatness_witness": {
                "threshold": float(min_uniform_mult),
                "max_observed": float(max_uniform_mult),
                "is_below_threshold": bool(max_uniform_mult < min_uniform_mult),
                "mathematical_proof": f"max_uniform_mult={max_uniform_mult:.3f} {'<' if max_uniform_mult < min_uniform_mult else '≥'} {min_uniform_mult:.3f}=threshold → {'flat' if max_uniform_mult < min_uniform_mult else 'hierarchical'}",
            },
        },
        "summary": {"layers_with_disagreement": len(all_layers), "layers_flagged_as_high_impact": len(flagged_layers)},
        "disagreement_focus_set": _build_focus_set(
            flagged_layers, all_layers, distribution_is_flat, min_uniform_mult, max_uniform_mult, uniform_share
        ),
        "flagged_layers": sorted(flagged_layers, key=lambda x: x["flagging_rationale"]["priority_score"], reverse=True),
        "all_layers_with_disagreement": sorted(
            all_layers, key=lambda x: x["flagging_rationale"]["priority_score"], reverse=True
        ),
    }


def _build_focus_set(
    flagged_layers, all_layers, distribution_is_flat, min_uniform_mult, max_uniform_mult, uniform_share
):
    """Build disagreement focus set for Bench consumption.

    Returns a structured focus set that Bench can use directly to restrict
    per-layer validation to only the most critical layers.
    """
    # Sort flagged layers by priority_score (highest first)
    sorted_flagged = sorted(flagged_layers, key=lambda x: x["flagging_rationale"]["priority_score"], reverse=True)

    # Extract layer names for Bench consumption
    high_impact_layer_names = [layer["layer_name"] for layer in sorted_flagged]

    # Determine recommended focus count
    if distribution_is_flat:
        # For flat distributions, no clear high-impact layers
        # Recommend focusing on top disagreement layers by spread
        sorted_all = sorted(all_layers, key=lambda x: x["flagging_rationale"]["priority_score"], reverse=True)
        top_disagreement_names = [layer["layer_name"] for layer in sorted_all[:3]]  # Top 3 by priority

        recommended_focus_n = min(3, len(all_layers))
        message = f"Energy distribution is flat (max={max_uniform_mult:.1f}× < {min_uniform_mult:.1f}× threshold, uniform_share={uniform_share:.3f}). Consider Bench validation on top {recommended_focus_n} disagreement layers by priority_score."

        return {
            "high_impact_layers": top_disagreement_names,
            "recommended_focus_n": recommended_focus_n,
            "focus_strategy": "top_disagreement_priority",
            "message": message,
            "distribution_type": "flat",
        }
    else:
        # Clear importance hierarchy - focus on flagged layers
        recommended_focus_n = len(high_impact_layer_names)
        total_disagreement_layers = len(all_layers)

        if recommended_focus_n == 0:
            # No high-impact layers (e.g., all below thresholds)
            message = f"No layers meet high-impact criteria. All {total_disagreement_layers} disagreement layers are below importance thresholds."
            return {
                "high_impact_layers": [],
                "recommended_focus_n": 0,
                "focus_strategy": "none",
                "message": message,
                "distribution_type": "hierarchical",
            }
        elif recommended_focus_n == 1:
            # Single high-impact layer
            top_layer = high_impact_layer_names[0]
            message = f"Focus Bench validation on 1 energy-significant layer: {top_layer}"
            return {
                "high_impact_layers": high_impact_layer_names,
                "recommended_focus_n": recommended_focus_n,
                "focus_strategy": "single_layer",
                "message": message,
                "distribution_type": "hierarchical",
            }
        else:
            # Multiple high-impact layers
            message = f"Focus Bench validation on {recommended_focus_n} energy-significant layers (highest priority: {high_impact_layer_names[0]})"
            return {
                "high_impact_layers": high_impact_layer_names,
                "recommended_focus_n": recommended_focus_n,
                "focus_strategy": "multiple_layers",
                "message": message,
                "distribution_type": "hierarchical",
            }


# ---------------------------------------------------------------------------
# Output formatting — canonical definitions in cli_format.py
# ---------------------------------------------------------------------------
from gradience.cli_format import _print_audit_summary as _print_audit_summary  # noqa: F401, E402
from gradience.cli_format import _print_monitor_result as _print_monitor_result  # noqa: F401, E402
from gradience.cli_format import (  # noqa: F401, E402
    _print_policy_disagreement_summary as _print_policy_disagreement_summary,
)
from gradience.cli_format import _print_qa_summary as _print_qa_summary  # noqa: F401, E402
from gradience.cli_format import _print_recommendations as _print_recommendations  # noqa: F401, E402
