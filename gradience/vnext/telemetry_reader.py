"""gradience.vnext.telemetry_reader

This module provides the reader counterpart to
:class:`~gradience.vnext.telemetry.TelemetryWriter`.

Design goals
------------
- Stream JSONL safely (skip partial / malformed lines)
- Validate schema == ``gradience.vnext.telemetry/v1``
- Normalize missing *optional* fields for known event types
- Provide convenience helpers:
  - :meth:`TelemetryReader.iter_events`
  - :meth:`TelemetryReader.latest_config`
  - :meth:`TelemetryReader.latest_eval`
  - :meth:`TelemetryReader.summarize`

The reader intentionally keeps validation lightweight. For strict enforcement
of schema and envelope requirements, enable ``strict_schema=True``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from .types import (
    TELEMETRY_SCHEMA_VERSION,
    ConfigSnapshot,
    EvalMetrics,
    SignalSnapshot,
)


JsonDict = Dict[str, Any]


class TelemetrySchemaError(ValueError):
    """Raised when telemetry schema version does not match expectations."""


class TelemetryFormatError(ValueError):
    """Raised when a telemetry record is missing required envelope fields."""


@dataclass
class ValidationIssue:
    line: int
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return f"line {self.line}: {self.message}"


def _safe_json_loads(line: str) -> Optional[JsonDict]:
    """Parse JSON into a dict, returning None on failure."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _normalize_event(e: JsonDict) -> JsonDict:
    """Normalize missing *optional* fields for known event types.

    Notes:
      - We *do not* invent required envelope fields (schema, ts, run_id, event).
        Those are validated separately.
      - We do set defaults for optional per-event payload fields so downstream
        consumers can rely on keys existing.
    """

    event_type = str(e.get("event") or "")

    # Common optional envelope fields
    e.setdefault("step", None)

    if event_type == "run_start":
        e.setdefault("meta", {})
        e.setdefault("config", {})
    elif event_type == "eval":
        e.setdefault("split", "unknown")
        e.setdefault("metrics", {})
    elif event_type == "metrics":
        e.setdefault("kind", "unknown")
        e.setdefault("metrics", {})
    elif event_type == "alert":
        e.setdefault("context", {})
    elif event_type == "recommendation":
        e.setdefault("recommendations", [])
    elif event_type == "run_end":
        # status is required by convention; defaulting keeps tools from crashing
        e.setdefault("status", "unknown")
    # train_step and unknown events: nothing else to normalize

    return e


def _evalmetrics_from_metrics_dict(
    metrics: Dict[str, Any],
    *,
    step: Optional[int] = None,
    split: Optional[str] = None,
) -> EvalMetrics:
    """Convert an eval metrics dict into :class:`~gradience.vnext.types.EvalMetrics`.

    Unknown keys are preserved in ``extras`` for forward compatibility.
    """
    if metrics is None:
        metrics = {}

    known = {"loss", "ppl", "accuracy", "n"}
    extras = {k: v for k, v in metrics.items() if k not in known}
    if step is not None:
        extras.setdefault("step", step)
    if split is not None:
        extras.setdefault("split", split)

    return EvalMetrics(
        loss=metrics.get("loss"),
        ppl=metrics.get("ppl"),
        accuracy=metrics.get("accuracy"),
        n=metrics.get("n"),
        extras=extras,
    )


class TelemetryReader:
    """Stream and summarize vNext telemetry JSONL."""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        strict_schema: bool = False,
        normalize: bool = True,
        allowed_schemas: Optional[Sequence[str]] = None,
    ):
        self.path = Path(path)
        self.strict_schema = strict_schema
        self.normalize = normalize
        self.allowed_schemas = tuple(allowed_schemas) if allowed_schemas is not None else (TELEMETRY_SCHEMA_VERSION,)
        self.issues: List[ValidationIssue] = []

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def iter_events(self, event_type: Optional[str] = None) -> Iterator[JsonDict]:
        """Iterate telemetry events.

        - Skips blank lines and malformed JSON.
        - Validates the envelope and schema.
        - Optionally normalizes missing optional fields.

        Args:
            event_type: If provided, only yield events whose ``event`` matches.
        """

        if not self.path.exists():
            return

        with self.path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                e = _safe_json_loads(line)
                if e is None:
                    self.issues.append(ValidationIssue(lineno, "malformed JSON; skipping"))
                    continue

                # Schema validation
                schema = e.get("schema")
                if schema not in self.allowed_schemas:
                    msg = f"schema mismatch: {schema!r} (expected one of {list(self.allowed_schemas)!r})"
                    if self.strict_schema:
                        raise TelemetrySchemaError(msg)
                    self.issues.append(ValidationIssue(lineno, msg))
                    continue

                # Envelope validation
                for k in ("ts", "run_id", "event"):
                    if k not in e:
                        msg = f"missing required key '{k}'"
                        if self.strict_schema:
                            raise TelemetryFormatError(msg)
                        self.issues.append(ValidationIssue(lineno, msg))
                        e = None
                        break
                if e is None:
                    continue

                if self.normalize:
                    e = _normalize_event(e)

                if event_type is None or e.get("event") == event_type:
                    yield e

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate(self, *, max_issues: int = 1000) -> List[str]:
        """Scan the file and return human-friendly validation issues."""
        self.issues.clear()
        for _ in self.iter_events(event_type=None):
            if len(self.issues) >= max_issues:
                self.issues.append(ValidationIssue(-1, f"too many issues; stopped after {max_issues}"))
                break
        return [str(i) for i in self.issues]

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def latest_config(self) -> Optional[ConfigSnapshot]:
        """Return the most recent :class:`~gradience.vnext.types.ConfigSnapshot` (from run_start)."""
        last: Optional[JsonDict] = None
        for e in self.iter_events(event_type="run_start"):
            last = e
        if not last:
            return None
        cfg = last.get("config")
        if not isinstance(cfg, dict):
            return None
        return ConfigSnapshot.from_dict(cfg)

    def latest_eval_event(self, *, split: str = "test") -> Optional[JsonDict]:
        """Return the most recent raw eval event for a given split."""
        last: Optional[JsonDict] = None
        for e in self.iter_events(event_type="eval"):
            if str(e.get("split") or "") == split:
                last = e
        return last

    def latest_eval(self, split: str = "test") -> Optional[EvalMetrics]:
        """Return the most recent eval metrics for a given split as EvalMetrics."""
        e = self.latest_eval_event(split=split)
        if not e:
            return None
        metrics = e.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        return _evalmetrics_from_metrics_dict(metrics, step=e.get("step"), split=split)

    def summarize(self) -> SignalSnapshot:
        """Summarize the run into a :class:`~gradience.vnext.types.SignalSnapshot`.

        Uses latest train/test eval metrics and pulls additional scalar signals
        from the most recent ``metrics`` events.
        """
        train_eval = self.latest_eval(split="train") or EvalMetrics()
        test_eval = self.latest_eval(split="test") or EvalMetrics()

        gap: Optional[float] = None
        if train_eval.ppl is not None and test_eval.ppl is not None and train_eval.ppl != 0:
            try:
                gap = float(test_eval.ppl) / float(train_eval.ppl)
            except Exception:
                gap = None

        # Generic (non-kind-specific) metrics
        stable_rank_mean: Optional[float] = None
        utilization_mean: Optional[float] = None
        dominance_act_mean: Optional[float] = None
        kappa_mean: Optional[float] = None

        # If present, we treat metrics(kind="lora_audit") as the authoritative
        # source for stable-rank/utilization summaries (because it is derived
        # from a full adapter audit rather than a partial live probe).
        lora_audit_summary: Optional[Dict[str, Any]] = None
        lora_audit_step: Optional[int] = None
        sr_from_audit: Optional[float] = None
        util_from_audit: Optional[float] = None

        sr_keys = ("stable_rank_mean", "avg_stable_rank", "stable_rank")
        util_keys = ("utilization_mean", "utilization", "rank_utilization")
        dom_keys = ("dominance_act_mean", "activation_dominance_mean", "avg_dominance_act")
        kappa_keys = ("kappa_mean", "kappa")

        last_step: Optional[int] = None
        for e in self.iter_events(event_type="metrics"):
            if isinstance(e.get("step"), int):
                last_step = e.get("step")

            kind = str(e.get("kind") or "unknown")
            m = e.get("metrics")
            if not isinstance(m, dict):
                continue

            # Kind-specific ingestion (vNext canonical)
            if kind == "lora_audit":
                lora_audit_summary = m
                if isinstance(e.get("step"), int):
                    lora_audit_step = e.get("step")

                # Prefer canonical keys
                if "stable_rank_mean" in m:
                    sr_from_audit = m.get("stable_rank_mean")
                if "utilization_mean" in m:
                    util_from_audit = m.get("utilization_mean")

            # Generic ingestion for other kinds (back/forward-compatible)
            if kind != "lora_audit":
                for k in sr_keys:
                    if k in m:
                        stable_rank_mean = m.get(k)
                        break
                for k in util_keys:
                    if k in m:
                        utilization_mean = m.get(k)
                        break

            # These are orthogonal: we ingest regardless of kind.
            for k in dom_keys:
                if k in m:
                    dominance_act_mean = m.get(k)
                    break
            for k in kappa_keys:
                if k in m:
                    kappa_mean = m.get(k)
                    break

        # If an audit summary exists, it wins for stable rank/utilization.
        if sr_from_audit is not None:
            stable_rank_mean = sr_from_audit
        if util_from_audit is not None:
            utilization_mean = util_from_audit

        extras: Dict[str, Any] = {}
        cfg = self.latest_config()
        if cfg is not None:
            extras["model_name"] = cfg.model_name
            extras["dataset_name"] = cfg.dataset_name
            extras["task_profile"] = cfg.task_profile.value
            if cfg.lora.r is not None:
                extras["lora_r"] = cfg.lora.r
        if last_step is not None:
            extras["last_metrics_step"] = last_step

        # Attach LoRA audit payload (if present) so monitor/policy can surface
        # detailed efficiency signals without reparsing the entire file.
        if lora_audit_summary is not None:
            extras["lora_audit"] = lora_audit_summary
        if lora_audit_step is not None:
            extras["lora_audit_step"] = lora_audit_step

        return SignalSnapshot(
            train=train_eval,
            test=test_eval,
            gap=gap,
            stable_rank_mean=stable_rank_mean,
            utilization_mean=utilization_mean,
            dominance_act_mean=dominance_act_mean,
            kappa_mean=kappa_mean,
            extras=extras,
        )

    # Back-compat alias for older tooling
    def summary(self) -> Dict[str, Any]:  # pragma: no cover
        """Return a human-friendly dict summary.

        Prefer :meth:`summarize` for typed output.
        """
        s = self.summarize()
        out: Dict[str, Any] = {
            "train": s.train.to_dict(),
            "test": s.test.to_dict(),
            "gap": s.gap,
            "stable_rank_mean": s.stable_rank_mean,
            "utilization_mean": s.utilization_mean,
            "dominance_act_mean": s.dominance_act_mean,
            "kappa_mean": s.kappa_mean,
        }
        out.update(s.extras)
        return out
