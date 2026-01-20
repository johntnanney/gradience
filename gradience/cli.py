"""Gradience CLI

Simple command-line interface for common operations.

Usage:
    gradience verify
    gradience report FILE
    gradience check CONFIG [--json] [--verbose] [--overrides ...]

    # Monitor a vNext telemetry run and emit alerts/recommendations
    gradience monitor RUN.jsonl [--gap-threshold 1.5] [--json] [--verbose]

    # Convenience wrapper (two-file merge):
    gradience check --task gsm8k --peft adapter_config.json --training training_args.json

    # Convenience wrapper using output directories (auto-detect files):
    gradience check --task gsm8k --peft-dir ./peft_out --training-dir ./trainer_out

    # Audit a PEFT LoRA adapter directory for rank/utilization waste:
    gradience audit --peft-dir ./peft_out [--top-wasteful 10] [--json]

Notes:
  * `check` consumes a Gradience vNext `ConfigSnapshot` (JSON/YAML) and emits
    `Recommendation[]` using the restraint-first policy.
  * Config files may be in canonical vNext form (nested optimizer/lora/training)
    or common "flat" / PEFT-style forms (e.g. adapter_config.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> None:
    """Run installation verification."""
    from gradience.verify import main as verify_main

    verify_main()


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> None:
    """Generate report from telemetry file."""
    import statistics

    telemetry_path = Path(args.file)
    if not telemetry_path.exists():
        print(f"Error: File not found: {telemetry_path}")
        sys.exit(1)

    # Load telemetry
    events: List[Dict[str, Any]] = []
    with open(telemetry_path) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                # Skip malformed lines
                pass

    if not events:
        print("Error: No events found in telemetry file")
        sys.exit(1)

    # Basic report
    print("=" * 60)
    print("GRADIENCE TELEMETRY REPORT")
    print("=" * 60)
    print(f"\nFile: {telemetry_path}")
    print(f"Total events: {len(events)}")

    # Event breakdown
    event_types: Dict[str, int] = {}
    for e in events:
        t = e.get("event", "unknown")
        event_types[t] = event_types.get(t, 0) + 1

    print("\nEvent breakdown:")
    for t, count in sorted(event_types.items()):
        print(f"  {t}: {count}")

    # Spectral summary
    spectral = [e for e in events if e.get("event") == "spectral"]
    if spectral:
        kappas: List[float] = []
        for e in spectral:
            # Support both per-matrix and aggregate spectral events.
            k = e.get("kappa_mean")
            if k is None:
                k = e.get("kappa")
            if isinstance(k, (int, float)):
                kappas.append(float(k))

        if kappas:
            print("\nSpectral metrics:")
            print(f"  κ mean: {statistics.mean(kappas):.1f}")
            print(f"  κ std: {statistics.stdev(kappas) if len(kappas) > 1 else 0:.1f}")
            print(f"  κ range: {min(kappas):.1f} - {max(kappas):.1f}")

    # Eval summary
    evals = [e for e in events if e.get("event") == "eval"]
    if evals:
        accs: List[float] = []
        for e in evals:
            a = e.get("accuracy")
            if a is None:
                a = e.get("acc")
            if isinstance(a, (int, float)):
                accs.append(float(a))

        if accs:
            print("\nEvaluation metrics:")
            print(f"  Final accuracy: {accs[-1]:.1%}")
            print(f"  Best accuracy: {max(accs):.1%}")

    # Guard events
    guard_events = [
        e
        for e in events
        if e.get("event")
        in ("corruption_detected", "rollback_started", "rollback_succeeded", "would_rollback")
    ]
    if guard_events:
        print("\nGuard events:")
        for e in guard_events:
            print(f"  Step {e.get('step', '?')}: {e.get('event')}")

    print()


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def _load_config_file(path: str) -> Dict[str, Any]:
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
        except Exception:
            import yaml

            data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping/object at the top level. Got: {type(data).__name__}")

    return data


def _autodetect_file_in_dir(
    dir_path: str,
    *,
    candidates: List[str],
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
    matches: List[Path] = []
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


def _first(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _parse_targets(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        out: List[str] = []
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


def _normalize_to_vnext_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort normalization of common config formats to vNext ConfigSnapshot dict."""

    # If it already looks like canonical vNext, just ensure nested dicts exist.
    if any(k in raw for k in ("optimizer", "lora", "training", "task_profile")):
        d = dict(raw)
        d["optimizer"] = dict(d.get("optimizer") or {})
        d["lora"] = dict(d.get("lora") or {})
        d["training"] = dict(d.get("training") or {})
        return d

    # Otherwise treat as "flat" or PEFT-like.
    d: Dict[str, Any] = {
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


def _blank_vnext_dict() -> Dict[str, Any]:
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


def _merge_fill_missing(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
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


def _apply_overrides(d: Dict[str, Any], args: argparse.Namespace) -> None:
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
        merged: List[str] = []
        for t in args.targets:
            merged.extend(_parse_targets(t))
        lora["target_modules"] = merged


def _severity_rank(sev: str) -> int:
    order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
    return order.get(sev.lower(), 99)


def _print_recommendations(config: Any, recs: List[Any], *, verbose: bool = False) -> None:
    print("=" * 72)
    print("GRADIENCE CHECK")
    print("=" * 72)

    # Config summary
    model_name = getattr(config, "model_name", None)
    dataset_name = getattr(config, "dataset_name", None)
    task_profile = getattr(getattr(config, "task_profile", None), "value", None) or str(getattr(config, "task_profile", "unknown"))

    opt = getattr(config, "optimizer", None)
    lora = getattr(config, "lora", None)

    lr = getattr(opt, "lr", None)
    wd = getattr(opt, "weight_decay", None)
    r = getattr(lora, "r", None)
    alpha = getattr(lora, "alpha", None)
    a_over_r = getattr(lora, "alpha_over_r", None)
    targets = list(getattr(lora, "target_modules", []) or [])

    print(f"Model:   {model_name or '-'}")
    print(f"Dataset: {dataset_name or '-'}")
    print(f"Profile: {task_profile}")

    print("\nKey knobs:")
    print(f"  LR:          {lr if lr is not None else '-'}")
    print(f"  Weight decay:{wd if wd is not None else '-'}")
    print(f"  Targets:     {', '.join(targets) if targets else '-'}")
    print(f"  Rank r:      {r if r is not None else '-'}")
    if alpha is None or a_over_r is None:
        print(f"  Alpha:       {alpha if alpha is not None else '-'}")
    else:
        print(f"  Alpha:       {alpha} (α/r={a_over_r:.3g})")

    if not recs:
        print("\nNo recommendations. Config looks reasonable.")
        return

    # Sort by severity then action
    recs_sorted = sorted(
        recs,
        key=lambda r: (
            _severity_rank(getattr(getattr(r, "severity", None), "value", str(getattr(r, "severity", "info")))),
            str(getattr(r, "action", "")),
        ),
    )

    print(f"\nRecommendations ({len(recs_sorted)}):")
    for i, rec in enumerate(recs_sorted, 1):
        sev = getattr(getattr(rec, "severity", None), "value", str(getattr(rec, "severity", "info"))).upper()
        action = getattr(rec, "action", "")
        msg = getattr(rec, "message", "")
        print(f"  {i:02d}. [{sev}] {action}: {msg}")

        if verbose:
            rationale = getattr(rec, "rationale", None)
            confidence = getattr(rec, "confidence", None)
            scope = getattr(rec, "scope", None)
            evidence = getattr(rec, "evidence", None)
            if rationale:
                print(f"       why: {rationale}")
            if confidence is not None:
                print(f"       confidence: {confidence:.2f}")
            if scope:
                print(f"       scope: {scope}")
            if evidence:
                try:
                    ev_json = json.dumps(evidence, sort_keys=True)
                except Exception:
                    ev_json = str(evidence)
                print(f"       evidence: {ev_json}")


def cmd_check(args: argparse.Namespace) -> None:
    """Validate a LoRA/training config and emit restraint-first recommendations."""

    # Convenience alias: --task behaves like --dataset
    if getattr(args, "task", None) is not None and args.dataset is None:
        args.dataset = args.task

    # Auto-detect convenience wrapper inputs if directories are provided.
    # Explicit file paths (--peft/--training) take precedence over directories.
    if args.peft is None and getattr(args, "peft_dir", None):
        try:
            args.peft = _autodetect_file_in_dir(
                args.peft_dir,
                candidates=["adapter_config.json", "adapter_config.yaml", "adapter_config.yml"],
                label="PEFT adapter_config",
            )
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    if args.training is None and getattr(args, "training_dir", None):
        try:
            args.training = _autodetect_file_in_dir(
                args.training_dir,
                candidates=["training_args.json", "training_args.yaml", "training_args.yml"],
                label="training_args",
            )
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Build merged config dict from up to three sources:
    #   1) positional CONFIG (canonical or flat)
    #   2) --peft adapter_config.json (or --peft-dir)
    #   3) --training training_args.json (or --training-dir)
    # Then apply explicit CLI overrides.

    if args.config is None and args.peft is None and args.training is None:
        print("Error: Please provide either a CONFIG file or --peft/--training inputs.")
        print("Examples:")
        print("  gradience check config.yaml")
        print("  gradience check --task gsm8k --peft adapter_config.json --training training_args.json")
        print("  gradience check --task gsm8k --peft-dir ./peft_out --training-dir ./trainer_out")
        sys.exit(1)

    merged = _blank_vnext_dict()
    sources: List[Dict[str, str]] = []

    def _load_and_norm(path: str) -> Dict[str, Any]:
        raw = _load_config_file(path)
        return _normalize_to_vnext_dict(raw)

    # (1) CONFIG
    if args.config is not None:
        try:
            d0 = _load_and_norm(args.config)
            merged = _merge_fill_missing(merged, d0)
            sources.append({"type": "config", "path": str(args.config)})
        except FileNotFoundError:
            print(f"Error: File not found: {args.config}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to parse config file '{args.config}': {e}")
            sys.exit(1)

    # (2) PEFT adapter_config
    if args.peft is not None:
        try:
            dp = _load_and_norm(args.peft)
            merged = _merge_fill_missing(merged, dp)
            sources.append({"type": "peft", "path": str(args.peft)})
        except FileNotFoundError:
            print(f"Error: File not found: {args.peft}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to parse PEFT config '{args.peft}': {e}")
            sys.exit(1)

    # (3) Training args
    if args.training is not None:
        try:
            dt = _load_and_norm(args.training)
            merged = _merge_fill_missing(merged, dt)
            sources.append({"type": "training", "path": str(args.training)})
        except FileNotFoundError:
            print(f"Error: File not found: {args.training}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to parse training args '{args.training}': {e}")
            sys.exit(1)

    # Attach sources for debugging
    merged.setdefault("extras", {})
    if sources:
        merged["extras"].setdefault("sources", [])
        # Don't explode existing sources if present
        if isinstance(merged["extras"].get("sources"), list):
            merged["extras"]["sources"].extend(sources)
        else:
            merged["extras"]["sources"] = sources

    # Apply explicit CLI overrides
    d = merged
    _apply_overrides(d, args)

    try:
        from gradience.vnext import ConfigSnapshot, check_config

        config = ConfigSnapshot.from_dict(d)
        recs = check_config(config)
    except Exception as e:
        print(f"Error: Failed to build ConfigSnapshot or run policy: {e}")
        sys.exit(1)

    if args.json:
        payload = {
            "config": config.to_dict(),
            "recommendations": [r.to_dict() for r in recs],
        }
        print(json.dumps(payload, indent=2))
        return

    _print_recommendations(config, recs, verbose=args.verbose)


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------


def _fmt(x: Any, *, pct: bool = False) -> str:
    if x is None:
        return "-"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if pct:
        return f"{xf * 100:.1f}%"
    # Use compact scientific for very small/large
    if abs(xf) != 0 and (abs(xf) < 1e-3 or abs(xf) >= 1e4):
        return f"{xf:.2e}"
    return f"{xf:.3g}"


def _fmt_params(n) -> str:
    """Format a parameter count into human-friendly units (K/M/B)."""
    if n is None:
        return "n/a"
    try:
        x = float(n)
    except Exception:
        return str(n)
    ax = abs(x)
    if ax >= 1e9:
        return f"{x/1e9:.1f}B"
    if ax >= 1e6:
        return f"{x/1e6:.1f}M"
    if ax >= 1e3:
        return f"{x/1e3:.1f}K"
    if x.is_integer():
        return str(int(x))
    return f"{x:.3g}"


def _extract_guard_activity(reader: Any) -> Dict[str, Any]:
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
                    guard_info["rollback_count"] = max(
                        guard_info["rollback_count"],
                        metrics.get("n_rollbacks", 0)
                    )
                
                # Latest snapshot info
                if "snapshot_count" in metrics:
                    guard_info["snapshot_count"] = metrics["snapshot_count"]
                if "memory_mb" in metrics:
                    guard_info["memory_mb"] = metrics["memory_mb"]
        
        # If we found any rollback count > 0, mark rollback as occurred
        if guard_info["rollback_count"] > 0:
            guard_info["rollback_occurred"] = True
    
    except Exception:
        # If anything fails, return minimal guard_info
        pass
    
    return guard_info

def _print_monitor_result(
    *,
    telemetry_path: Path,
    config: Any,
    signals: Any,
    alerts: List[Dict[str, Any]],
    recs: List[Any],
    issues: List[str],
    verbose: bool = False,
    guard_activity: Optional[Dict[str, Any]] = None,
) -> None:
    print("=" * 72)
    print("GRADIENCE MONITOR")
    print("=" * 72)
    print(f"File: {telemetry_path}")

    # Config summary (best-effort)
    model_name = getattr(config, "model_name", None) if config is not None else None
    dataset_name = getattr(config, "dataset_name", None) if config is not None else None
    task_profile = None
    if config is not None:
        task_profile = getattr(getattr(config, "task_profile", None), "value", None) or str(getattr(config, "task_profile", "unknown"))
    else:
        # Fall back to summarize() extras
        try:
            task_profile = (getattr(signals, "extras", {}) or {}).get("task_profile")
            model_name = model_name or (getattr(signals, "extras", {}) or {}).get("model_name")
            dataset_name = dataset_name or (getattr(signals, "extras", {}) or {}).get("dataset_name")
        except Exception:
            task_profile = task_profile or "unknown"

    print(f"Model:   {model_name or '-'}")
    print(f"Dataset: {dataset_name or '-'}")
    print(f"Profile: {task_profile or 'unknown'}")

    # Signals summary
    train = getattr(signals, "train", None)
    test = getattr(signals, "test", None)

    train_ppl = getattr(train, "ppl", None) if train is not None else None
    test_ppl = getattr(test, "ppl", None) if test is not None else None
    train_acc = getattr(train, "accuracy", None) if train is not None else None
    test_acc = getattr(test, "accuracy", None) if test is not None else None
    gap = getattr(signals, "gap", None)

    print("\nLatest eval signals:")
    print(f"  Train PPL: {_fmt(train_ppl)}")
    print(f"  Test  PPL: {_fmt(test_ppl)}")
    print(f"  Gap:       {_fmt(gap)}x")
    print(f"  Train Acc: {_fmt(train_acc, pct=True)}")
    print(f"  Test  Acc: {_fmt(test_acc, pct=True)}")

    # Optional structural signals
    sr = getattr(signals, "stable_rank_mean", None)
    util = getattr(signals, "utilization_mean", None)
    dom = getattr(signals, "dominance_act_mean", None)
    kap = getattr(signals, "kappa_mean", None)

    if any(v is not None for v in (sr, util, dom, kap)):
        print("\nDiagnostics:")
        if sr is not None:
            print(f"  Stable rank (mean): {_fmt(sr)}")
        if util is not None:
            print(f"  Utilization (mean): {_fmt(util, pct=True)}")
            # Dominance ingredients (scaled) — verbose only
            if verbose:
                _la = (getattr(signals, 'extras', None) or {}).get('lora_audit') or {}
                _s50 = _la.get('delta_sigma_max_scaled_p50')
                _s90 = _la.get('delta_sigma_max_scaled_p90')
                _f50 = _la.get('delta_frob_norm_scaled_p50')
                _f90 = _la.get('delta_frob_norm_scaled_p90')
                if any(v is not None for v in (_s50, _s90, _f50, _f90)):
                    _s50s = 'n/a' if _s50 is None else '{:.4g}'.format(float(_s50))
                    _s90s = 'n/a' if _s90 is None else '{:.4g}'.format(float(_s90))
                    _f50s = 'n/a' if _f50 is None else '{:.4g}'.format(float(_f50))
                    _f90s = 'n/a' if _f90 is None else '{:.4g}'.format(float(_f90))
                    print('  Dominance (scaled):')
                    print(f'    sigma_max_scaled (p50/p90): {_s50s}/{_s90s}')
                    print(f'    frob_norm_scaled (p50/p90): {_f50s}/{_f90s}')

        if dom is not None:
            print(f"  Activation dominance (mean): {_fmt(dom)}")
        if kap is not None:
            print(f"  Kappa (mean): {_fmt(kap)}")

    # Optional: surface LoRA audit stats if present in telemetry summary.
    audit = None
    try:
        audit = (getattr(signals, "extras", {}) or {}).get("lora_audit")
    except Exception:
        audit = None

    if isinstance(audit, dict) and audit:
        total_params = audit.get("total_lora_params")
        n_layers = audit.get("n_layers")
        e90_p50 = audit.get("energy_rank_90_p50")
        e90_p90 = audit.get("energy_rank_90_p90")

        def _fmt_dom_params(p: Any) -> str:
            try:
                pf = float(p)
            except Exception:
                return "-"
            if pf >= 1e6:
                return f"{pf/1e6:.1f}M"
            if pf >= 1e3:
                return f"{pf/1e3:.1f}K"
            return f"{pf:.0f}"

        print("\nLoRA audit:")
        print(f"  LoRA params: {_fmt_params(total_params)}")
        print(f"  Layers:      {_fmt(n_layers)}")
        if e90_p50 is not None or e90_p90 is not None:
            print(f"  Energy rank k@90% (p50/p90): {_fmt(e90_p50)}/{_fmt(e90_p90)}")

            # Suggested rank printout (global)
            try:
                s_med = summary.get("suggested_r_global_median")
                s_p90 = summary.get("suggested_r_global_90")
                p50 = summary.get("energy_rank_90_p50")
                p90 = summary.get("energy_rank_90_p90")
            except Exception:
                s_med = s_p90 = p50 = p90 = None

            if s_med:
                print(f"  Suggested rank (median): r={int(s_med)} likely sufficient for most layers (p50 k@90%={p50})")
            if s_p90:
                print(f"  Suggested rank (p90):    r={int(s_p90)} covers worst-case layers at 90% energy (p90 k@90%={p90})")

        by_type = audit.get("by_type")
        if isinstance(by_type, dict) and by_type:
            # Show a compact per-type breakdown.
            for t in ("attn", "mlp", "other"):
                row = by_type.get(t)
                if not isinstance(row, dict):
                    continue
                t_params = row.get("params")
                t_util = row.get("utilization_mean")
                t_sr = row.get("stable_rank_mean")
                print(
                    f"  {t:>5}: params={_fmt_params(t_params)}  util={_fmt(t_util, pct=True)}  sr={_fmt(t_sr)}"
                )

    # Guard activity
    if guard_activity and guard_activity.get("present"):
        # In verbose mode, always show Guard activity if present
        if verbose:
            print("\nGuard activity:")
            print(f"  Last action:    {guard_activity.get('last_action', '-')}")
            print(f"  Rollbacks:      {guard_activity.get('rollback_count', 0)}")
            if guard_activity.get("last_trigger_code"):
                print(f"  Last trigger:   {guard_activity['last_trigger_code']}")
            print(f"  Snapshots:      {guard_activity.get('snapshot_count', 0)}")
            print(f"  Memory usage:   {_fmt(guard_activity.get('memory_mb', 0))} MB")
        
        # In non-verbose mode, only show if rollback occurred or training aborted
        elif guard_activity.get("rollback_occurred") or guard_activity.get("aborted"):
            if guard_activity.get("rollback_occurred"):
                rollback_count = guard_activity.get("rollback_count", 1)
                print(f"\n⚠ Guard performed {rollback_count} rollback(s) during training")
            if guard_activity.get("aborted"):
                print("⚠ Guard aborted rollback attempts (anti-thrash protection)")

    # Issues
    if issues:
        print(f"\nTelemetry issues: {len(issues)}")
        if verbose:
            for line in issues[:10]:
                print(f"  - {line}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")

    # Alerts
    if alerts:
        print(f"\nAlerts ({len(alerts)}):")
        for i, a in enumerate(alerts, 1):
            sev = str(a.get("severity", "info")).upper()
            code = a.get("code", "")
            msg = a.get("message", "")
            print(f"  {i:02d}. [{sev}] {code}: {msg}")
            if verbose and a.get("context"):
                try:
                    ctx = json.dumps(a["context"], sort_keys=True)
                except Exception:
                    ctx = str(a["context"])
                print(f"       context: {ctx}")

    # Recommendations
    if recs:
        print(f"\nRecommendations ({len(recs)}):")
        for i, rec in enumerate(recs, 1):
            sev = getattr(getattr(rec, "severity", None), "value", str(getattr(rec, "severity", "info"))).upper()
            action = getattr(rec, "action", "")
            msg = getattr(rec, "message", "")
            print(f"  {i:02d}. [{sev}] {action}: {msg}")
            if verbose:
                rationale = getattr(rec, "rationale", None)
                confidence = getattr(rec, "confidence", None)
                scope = getattr(rec, "scope", None)
                evidence = getattr(rec, "evidence", None)
                if rationale:
                    print(f"       why: {rationale}")
                if confidence is not None:
                    try:
                        print(f"       confidence: {float(confidence):.2f}")
                    except Exception:
                        print(f"       confidence: {confidence}")
                if scope:
                    print(f"       scope: {scope}")
                if evidence:
                    try:
                        ev_json = json.dumps(evidence, sort_keys=True)
                    except Exception:
                        ev_json = str(evidence)
                    print(f"       evidence: {ev_json}")
    else:
        print("\nNo recommendations emitted.")


def cmd_monitor(args: argparse.Namespace) -> None:
    """Monitor a vNext telemetry run and emit alerts/recommendations."""

    telemetry_path = Path(args.file)
    if not telemetry_path.exists():
        print(f"Error: File not found: {telemetry_path}")
        sys.exit(1)

    try:
        from gradience.vnext import TelemetryReader
        from gradience.vnext.policy import check_run
    except Exception as e:
        print(f"Error: Failed to import Gradience vNext components: {e}")
        sys.exit(1)

    try:
        reader = TelemetryReader(
            telemetry_path,
            strict_schema=bool(getattr(args, "strict_schema", False)),
            normalize=True,
        )
    except Exception as e:
        print(f"Error: Failed to open telemetry file: {e}")
        sys.exit(1)

    # Pull typed summaries
    config = None
    try:
        config = reader.latest_config()
    except Exception:
        config = None

    try:
        signals = reader.summarize()
    except Exception as e:
        print(f"Error: Failed to summarize telemetry: {e}")
        sys.exit(1)

    # Emit alerts (simple, gap-first)
    alerts: List[Dict[str, Any]] = []
    gap = getattr(signals, "gap", None)
    if gap is not None:
        try:
            gap_f = float(gap)
            if gap_f >= float(args.gap_threshold):
                alerts.append(
                    {
                        "severity": "warning",
                        "code": "memorization_gap",
                        "message": f"Train/test PPL gap {gap_f:.2f}x >= {float(args.gap_threshold):.2f}x (memorization risk)",
                        "context": {"gap": gap_f, "threshold": float(args.gap_threshold)},
                    }
                )
        except Exception:
            pass
    else:
        alerts.append(
            {
                "severity": "info",
                "code": "gap_unavailable",
                "message": "Could not compute train/test gap (missing train/test PPL).",
                "context": {},
            }
        )


    # Policy-driven recommendations (config + signals)
    try:
        recs = check_run(config, signals, gap_threshold=float(args.gap_threshold))
    except Exception as e:
        print(f"Error: Policy evaluation failed: {e}")
        sys.exit(1)

    issues = [str(i) for i in getattr(reader, "issues", []) or []]

    # Extract Guard activity from telemetry
    guard_activity = _extract_guard_activity(reader)

    # Guard alerts (conservative triage advice)
    if guard_activity and guard_activity.get("present"):
        # Abort takes precedence over rollback as it's the more serious condition
        if guard_activity.get("aborted"):
            alerts.append(
                {
                    "severity": "error",
                    "code": "guard_abort",
                    "message": "⚠️ EXPERIMENTAL Guard stopped training due to repeated instability. Guard CANNOT fix root causes (data bugs, bad objectives). Investigate underlying issues before re-running. ALWAYS validate with eval.",
                    "context": {
                        "last_action": guard_activity.get("last_action"),
                        "snapshot_count": guard_activity.get("snapshot_count", 0),
                        "rollback_count": guard_activity.get("rollback_count", 0),
                        "note": "Guard is experimental and can stop training",
                    },
                }
            )
        elif guard_activity.get("rollback_occurred"):
            rollback_count = guard_activity.get("rollback_count", 1)
            alerts.append(
                {
                    "severity": "warning",
                    "code": "guard_intervention",
                    "message": f"⚠️ EXPERIMENTAL Guard rolled back adapter weights {rollback_count} time(s). This does NOT fix data bugs or bad objectives. Investigate triggers (grad explosion/NaN), check data pipeline, consider lowering LR. ALWAYS validate with eval.",
                    "context": {
                        "rollback_count": rollback_count,
                        "last_action": guard_activity.get("last_action"),
                        "snapshot_count": guard_activity.get("snapshot_count", 0),
                        "note": "Guard is experimental and rolls back weights",
                    },
                }
            )

    if args.json:
        payload = {
            "file": str(telemetry_path),
            "config": config.to_dict() if config is not None else None,
            "signals": signals.to_dict() if hasattr(signals, "to_dict") else {},
            "alerts": alerts,
            "recommendations": [r.to_dict() for r in recs],
            "telemetry_issues": issues,
        }
        print(json.dumps(payload, indent=2))
        return

    _print_monitor_result(
        telemetry_path=telemetry_path,
        config=config,
        signals=signals,
        alerts=alerts,
        recs=recs,
        issues=issues,
        verbose=args.verbose,
        guard_activity=guard_activity,
    )


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def _print_audit_summary(result: Any, *, top_wasteful: int = 0) -> None:
    """Pretty-print a compact LoRA audit summary."""

    def _fmt_params(p: Any) -> str:
        try:
            pf = float(p)
        except Exception:
            return "-"
        if pf >= 1e6:
            return f"{pf/1e6:.1f}M"
        if pf >= 1e3:
            return f"{pf/1e3:.1f}K"
        return f"{pf:.0f}"

    print("=" * 72)
    print("GRADIENCE LoRA AUDIT")
    print("=" * 72)

    peft_dir = getattr(result, "peft_dir", None)
    cfg_path = getattr(result, "adapter_config_path", None)
    w_path = getattr(result, "adapter_weights_path", None)

    if peft_dir:
        print(f"PEFT dir: {peft_dir}")
    if cfg_path:
        print(f"Config:   {cfg_path}")
    if w_path:
        print(f"Weights:  {w_path}")

    print("\nSummary:")
    print(f"  LoRA params: {_fmt_params(getattr(result, 'total_lora_params', None))}")
    print(f"  Layers:      {_fmt(getattr(result, 'n_layers', None))}")
    print(f"  Stable rank (mean):    {_fmt(getattr(result, 'stable_rank_mean', None))}")
    print(f"  Stable rank (median):  {_fmt(getattr(result, 'stable_rank_median', None))}")
    print(f"  Stable rank (w-mean):  {_fmt(getattr(result, 'stable_rank_weighted_mean', None))}")
    print(f"  Effective rank (mean): {_fmt(getattr(result, 'effective_rank_mean', None))}")
    print(f"  Utilization (mean):    {_fmt(getattr(result, 'utilization_mean', None), pct=True)}")

    e90_p50 = getattr(result, "energy_rank_90_p50", None)
    e90_p90 = getattr(result, "energy_rank_90_p90", None)
    if e90_p50 is not None or e90_p90 is not None:
        print(f"  Energy rank k@90% (p50/p90): {_fmt(e90_p50)}/{_fmt(e90_p90)}")
        # Suggested rank printout (audit)
        def _snap_rank(_k):
            if _k is None:
                return None
            try:
                _k = float(_k)
            except Exception:
                return None
            for _r in (1, 2, 4, 8, 16, 32):
                if _k <= _r:
                    return _r
            return 32

        try:
            _p50 = summary.get('energy_rank_90_p50')
            _p90 = summary.get('energy_rank_90_p90')
        except Exception:
            _p50 = _p90 = None
        _s_med = _snap_rank(_p50)
        _s_p90 = _snap_rank(_p90)
        if _s_med is not None:
            print(f"  Suggested rank (median): r={int(_s_med)} likely sufficient for most layers (p50 k@90%={_p50})")
        if _s_p90 is not None:
            print(f"  Suggested rank (p90):    r={int(_s_p90)} covers worst-case layers at 90% energy (p90 k@90%={_p90})")


    by_type = getattr(result, "by_type", None)
    if isinstance(by_type, dict) and by_type:
        print("\nBy module type:")
        for t in ("attn", "mlp", "other"):
            row = by_type.get(t)
            if not isinstance(row, dict):
                continue
            print(
                f"  {t:>5}: params={_fmt_params(row.get('params'))}  layers={_fmt(row.get('n_layers'))}  "
                f"util={_fmt(row.get('utilization_mean'), pct=True)}  sr={_fmt(row.get('stable_rank_mean'))}"
            )

    issues = getattr(result, "issues", None)
    if isinstance(issues, list) and issues:
        print(f"\nIssues ({len(issues)}):")
        for line in issues[:10]:
            print(f"  - {line}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")

    if top_wasteful and int(top_wasteful) > 0:
        layers = getattr(result, "layers", None)
        if isinstance(layers, list) and layers:
            # Most wasteful = lowest utilization
            ls = sorted(layers, key=lambda x: getattr(x, "utilization", 1e9))[: int(top_wasteful)]
            print(f"\nMost wasteful layers (lowest utilization, top {len(ls)}):")
            for i, layer in enumerate(ls, 1):
                name = getattr(layer, "name", "?")
                mtype = getattr(layer, "module_type", "?")
                r = getattr(layer, "r", None)
                sr = getattr(layer, "stable_rank", None)
                util = getattr(layer, "utilization", None)
                e90 = getattr(layer, "energy_rank_90", None)
                print(
                    f"  {i:02d}. {mtype:>4} r={r:<3} util={_fmt(util, pct=True):>6}  sr={_fmt(sr):>5}  k@90%={_fmt(e90):>3}  {name}"
                )


def cmd_audit(args: argparse.Namespace) -> None:
    import json as jsonlib
    """Audit a PEFT LoRA adapter directory and print a compact efficiency summary."""

    peft_dir = getattr(args, "peft_dir", None)
    if not peft_dir:
        print("Error: --peft-dir is required")
        sys.exit(1)

    try:
        from gradience.vnext.audit import audit_lora_peft_dir
    except Exception as e:
        print(f"Error: Failed to import LoRA audit module: {e}")
        sys.exit(1)

    try:
        result = audit_lora_peft_dir(
            peft_dir,
            adapter_config_path=getattr(args, "adapter_config", None),
            adapter_weights_path=getattr(args, "weights", None),
            map_location="cpu",
            include_top_singular_values=int(getattr(args, "top_singular_values", 0) or 0),
            base_model_id=getattr(args, "base_model", None),
            base_norms_cache=getattr(args, "base_norms_cache", None),
            compute_udr=not getattr(args, "no_udr", False),
        )
        # --- audit --append support ---
        if getattr(args, "append", None):
            import json, time
            from pathlib import Path
            append_path = Path(args.append)
            run_id = None
            last_step = None
            if append_path.exists():
                try:
                    with append_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                e = jsonlib.loads(line)
                            except Exception:
                                continue
                            if isinstance(e, dict):
                                if run_id is None and isinstance(e.get("run_id"), str):
                                    run_id = e.get("run_id")
                                if isinstance(e.get("step"), int):
                                    last_step = e.get("step")
                except Exception:
                    pass
            if run_id is None:
                run_id = f"audit_{int(time.time())}"
            # Prefer structured event helper if available
            try:
                event = result.to_metrics_event(run_id=run_id, step=last_step)
            except Exception:
                event = {
                    "schema": "gradience.vnext.telemetry/v1",
                    "ts": time.time(),
                    "run_id": run_id,
                    "event": "metrics",
                    "step": last_step,
                    "kind": "lora_audit",
                    "metrics": result.to_summary_dict(include_layers=False),
                }
            append_path.parent.mkdir(parents=True, exist_ok=True)
            with append_path.open("a", encoding="utf-8") as f:
                f.write(jsonlib.dumps(event, default=str) + "\n")
            if not getattr(args, "json", False):
                print(f"Appended lora_audit metrics to {append_path}")
    except Exception as e:
        print(f"Error: Audit failed: {e}")
        sys.exit(1)

    top_wasteful = int(getattr(args, "top_wasteful", 0) or 0)
    # Include layers if --layers flag is set OR if --top-wasteful is specified
    include_layers = getattr(args, "layers", False) or top_wasteful > 0

    if args.json:
        try:
            # When --layers is set, include all layers; otherwise respect top_wasteful
            if getattr(args, "layers", False):
                payload = result.to_summary_dict(include_layers=True, topk_layers=None)
            else:
                payload = result.to_summary_dict(include_layers=include_layers, topk_layers=top_wasteful if include_layers else None)
            
            # Add per-layer rank suggestions if requested
            suggest_per_layer = getattr(args, "suggest_per_layer", False)
            if suggest_per_layer:
                if not include_layers and not getattr(args, "layers", False):
                    print("Error: --suggest-per-layer requires --layers flag", file=sys.stderr)
                    sys.exit(1)
                
                try:
                    from gradience.vnext.rank_suggestion import suggest_per_layer_ranks
                    rank_suggestions = suggest_per_layer_ranks(payload)
                    payload["rank_suggestions"] = rank_suggestions.to_dict()
                except Exception as e:
                    payload["rank_suggestions_error"] = str(e)
                    
        except Exception:
            # Fallback if result isn't the expected dataclass
            payload = {"error": "unexpected_audit_result_type", "type": str(type(result))}
        print(jsonlib.dumps(payload, indent=2))
        return

    _print_audit_summary(result, top_wasteful=top_wasteful)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gradience",
        description="Spectral telemetry and restraint-first diagnostics for neural network training",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # verify
    verify_parser = subparsers.add_parser("verify", help="Verify installation")
    verify_parser.set_defaults(func=cmd_verify)

    # report
    report_parser = subparsers.add_parser("report", help="Generate report from telemetry")
    report_parser.add_argument("file", help="Path to telemetry JSONL file")
    report_parser.set_defaults(func=cmd_report)

    # check
    check_parser = subparsers.add_parser("check", help="Validate a config and emit restraint-first recommendations")
    check_parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to config JSON/YAML (canonical vNext or flat/PEFT-style). Optional if using --peft/--training.",
    )

    # Convenience wrapper inputs
    check_parser.add_argument("--peft", type=str, default=None, help="Path to PEFT adapter_config.json (or YAML)")
    check_parser.add_argument("--training", type=str, default=None, help="Path to training_args.json (or YAML)")

    # Convenience wrapper inputs (directories)
    check_parser.add_argument(
        "--peft-dir",
        type=str,
        default=None,
        help="Path to a PEFT output directory (auto-detects adapter_config.json). Ignored if --peft is set.",
    )
    check_parser.add_argument(
        "--training-dir",
        type=str,
        default=None,
        help="Path to a training output directory (auto-detects training_args.json). Ignored if --training is set.",
    )

    # `--task` is a convenience alias for `--dataset` (matches internal naming)
    check_parser.add_argument("--task", type=str, default=None, help="Convenience alias for --dataset (e.g., gsm8k, sst2)")

    # Optional overrides
    check_parser.add_argument("--model", type=str, default=None, help="Override model name")
    check_parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    check_parser.add_argument(
        "--task-profile",
        type=str,
        default=None,
        choices=["easy_classification", "hard_reasoning", "generation", "unknown"],
        help="Override task profile",
    )

    check_parser.add_argument("--optimizer", type=str, default=None, help="Override optimizer name (e.g., AdamW)")
    check_parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    check_parser.add_argument("--weight-decay", type=float, default=None, help="Override weight decay")

    check_parser.add_argument("--r", type=int, default=None, help="Override LoRA rank")
    check_parser.add_argument("--alpha", type=float, default=None, help="Override LoRA alpha")
    check_parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Override target modules (space-separated and/or comma-separated)",
    )

    check_parser.add_argument("--notes", type=str, default=None, help="Attach notes to ConfigSnapshot")

    check_parser.add_argument("--verbose", action="store_true", help="Print rationale/evidence")
    check_parser.add_argument("--json", action="store_true", help="Output JSON instead of pretty text")

    check_parser.set_defaults(func=cmd_check)

    # audit
    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit a PEFT LoRA adapter directory for rank/utilization waste",
    )
    audit_parser.add_argument(
        "--append",
        default=None,
        help="Append lora_audit metrics event to an existing vNext run JSONL",
    )
    audit_parser.add_argument(
        "--peft-dir",
        type=str,
        required=True,
        help="Path to a PEFT output directory (containing adapter_config.* and adapter weights)",
    )
    audit_parser.add_argument(
        "--adapter-config",
        type=str,
        default=None,
        help="Optional explicit path to adapter_config.json/yaml (overrides auto-detect)",
    )
    audit_parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional explicit path to adapter_model.(safetensors|bin|pt) (overrides auto-detect)",
    )
    audit_parser.add_argument(
        "--top-wasteful",
        type=int,
        default=0,
        help="Print N most wasteful layers (lowest utilization). 0 disables.",
    )
    audit_parser.add_argument(
        "--top-singular-values",
        type=int,
        default=0,
        help="Include top-k singular values per layer in JSON output (cost: small).",
    )
    audit_parser.add_argument("--json", action="store_true", help="Output JSON instead of pretty text")
    audit_parser.add_argument(
        "--layers",
        action="store_true",
        help="Include per-layer audit rows in --json output (can be large).",
    )
    audit_parser.add_argument(
        "--suggest-per-layer",
        action="store_true",
        help="Include per-layer rank suggestions in --json output (requires --layers).",
    )
    # UDR/SDI support
    audit_parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model ID or path for UDR computation (e.g., 'microsoft/DialoGPT-medium')",
    )
    audit_parser.add_argument(
        "--base-norms-cache",
        type=str,
        default=None,
        help="Path to save/load base model norms cache (speeds up repeated audits)",
    )
    audit_parser.add_argument(
        "--no-udr",
        action="store_true",
        help="Skip UDR computation even if base model available",
    )
    audit_parser.set_defaults(func=cmd_audit)

    # monitor
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Analyze a vNext telemetry JSONL run and emit alerts/recommendations",
    )
    monitor_parser.add_argument("file", help="Path to vNext telemetry JSONL file")
    monitor_parser.add_argument(
        "--gap-threshold",
        type=float,
        default=1.5,
        help="Train/test PPL ratio threshold above which we warn about memorization (default: 1.5)",
    )
    monitor_parser.add_argument(
        "--strict-schema",
        action="store_true",
        help="Fail fast on schema/envelope validation issues instead of skipping bad lines",
    )
    monitor_parser.add_argument("--verbose", action="store_true", help="Print rationale/evidence and telemetry issues")
    monitor_parser.add_argument("--json", action="store_true", help="Output JSON instead of pretty text")
    monitor_parser.set_defaults(func=cmd_monitor)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
