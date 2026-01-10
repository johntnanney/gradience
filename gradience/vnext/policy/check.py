"""Config validator policy.

`check_config()` consumes a :class:`~gradience.vnext.types.ConfigSnapshot` and
emits a list of :class:`~gradience.vnext.types.Recommendation`.

This is the lowest-ambition / highest-value part of the "restraint navigator"
strategy:

* Don't try to predict final accuracy.
* Do try to prevent obviously-bad configurations (too aggressive LR, overly
  wide target modules, bloated rank) for known task families.

The implementation is deliberately conservative:

* It prefers **safe starting points** (restraint) over "optimal".
* It attaches **confidence** and **scope** to each heuristic.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

from ..types import ConfigSnapshot, Recommendation, Severity, SignalSnapshot, TaskProfile


# ---------------------------------------------------------------------------
# Target-module classification
# ---------------------------------------------------------------------------

# Common attention projection names across popular architectures.
_ATTN_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "qkv_proj",
    "Wqkv",
    "to_q",
    "to_k",
    "to_v",
    "to_out",
}

# Common MLP / FFN projection names.
_MLP_NAMES = {
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
    "w1",
    "w2",
    "w3",
}


def _classify_target_modules(mods: Iterable[str]) -> Dict[str, List[str]]:
    """Return a dict with keys: 'attn', 'mlp', 'other'."""

    attn: List[str] = []
    mlp: List[str] = []
    other: List[str] = []

    for m in mods:
        m_str = str(m)
        m_low = m_str.lower()

        # Exact match first
        if m_str in _ATTN_NAMES or m_low in _ATTN_NAMES:
            attn.append(m_str)
            continue
        if m_str in _MLP_NAMES or m_low in _MLP_NAMES:
            mlp.append(m_str)
            continue

        # Heuristic substring match
        if any(tok in m_low for tok in ("q_proj", "k_proj", "v_proj", "o_proj", "to_q", "to_k", "to_v", "to_out")):
            attn.append(m_str)
        elif any(tok in m_low for tok in ("gate", "up_proj", "down_proj", "fc1", "fc2", "mlp", "ffn", "w1", "w2", "w3")):
            mlp.append(m_str)
        else:
            other.append(m_str)

    return {"attn": attn, "mlp": mlp, "other": other}


def _infer_task_profile_from_dataset_name(dataset_name: Optional[str]) -> Optional[TaskProfile]:
    """Best-effort inference when task_profile is UNKNOWN."""

    if not dataset_name:
        return None
    d = dataset_name.lower()

    # Hard reasoning / exact-match generation style
    if any(k in d for k in ("gsm8k", "svamp", "math", "aqua", "asdiv", "multiarith", "proof", "reason")):
        return TaskProfile.HARD_REASONING

    # Easy-ish classification
    if any(k in d for k in ("sst", "sst2", "mnli", "qqp", "qnli", "rte", "cola", "mrpc", "sentiment", "classification")):
        return TaskProfile.EASY_CLASSIFICATION

    return None


def _rec(
    *,
    severity: Severity,
    action: str,
    message: str,
    rationale: Optional[str] = None,
    confidence: Optional[float] = None,
    scope: Optional[str] = None,
    evidence: Optional[Dict[str, object]] = None,
) -> Recommendation:
    return Recommendation(
        severity=severity,
        action=action,
        message=message,
        rationale=rationale,
        confidence=confidence,
        scope=scope,
        evidence=dict(evidence or {}),
    )


# ---------------------------------------------------------------------------
# Policy: check
# ---------------------------------------------------------------------------


def check_config(config: ConfigSnapshot) -> List[Recommendation]:
    """Validate a config and emit actionable recommendations.

    Notes
    -----
    This function is intentionally:

    * side-effect free
    * fast (no model weights required)
    * conservative (recommends safe starting points)
    """

    recs: List[Recommendation] = []

    # Resolve task profile (prefer explicit)
    task_profile = config.task_profile
    inferred = None
    if task_profile == TaskProfile.UNKNOWN:
        inferred = _infer_task_profile_from_dataset_name(config.dataset_name)
        if inferred is not None:
            task_profile = inferred
            recs.append(
                _rec(
                    severity=Severity.INFO,
                    action="set_task_profile",
                    message=f"Task profile not set; inferred '{task_profile.value}' from dataset_name='{config.dataset_name}'.",
                    rationale="Gradience policies use coarse task families to choose safe defaults.",
                    confidence=0.6,
                    scope="config",
                    evidence={"dataset_name": config.dataset_name, "inferred": task_profile.value},
                )
            )

    # Extract core knobs
    lr = config.optimizer.lr
    wd = config.optimizer.weight_decay
    lora = config.lora
    r = lora.r
    a_over_r = lora.alpha_over_r
    targets = list(lora.target_modules or [])
    target_classes = _classify_target_modules(targets)

    # Helper: how "wide" is target set
    has_mlp_targets = len(target_classes["mlp"]) > 0
    has_attn_targets = len(target_classes["attn"]) > 0
    has_other_targets = len(target_classes["other"]) > 0

    # ------------------------------------------------------------------
    # Task family: HARD_REASONING
    # ------------------------------------------------------------------
    if task_profile == TaskProfile.HARD_REASONING:
        # (1) LR: dominant restraint lever in our LoRA studies
        if lr is None:
            recs.append(
                _rec(
                    severity=Severity.WARNING,
                    action="set_lr",
                    message="Learning rate not specified. For reasoning tasks, start with lr=5e-5 (AdamW) as a safe default.",
                    rationale="Reasoning-style fine-tunes were highly sensitive to LR; lower LR reduced memorization and improved accuracy.",
                    confidence=0.8,
                    scope=task_profile.value,
                    evidence={"suggested_lr": 5e-5},
                )
            )
        else:
            # Treat >1e-4 as potentially aggressive for reasoning; 5e-5 was a strong default in experiments.
            if lr > 1e-4:
                recs.append(
                    _rec(
                        severity=Severity.WARNING,
                        action="lower_lr",
                        message=f"LR={lr:.2e} may be too aggressive for reasoning tasks. Try lr=5e-5 as a restraint-first baseline.",
                        rationale="Large adapter updates tend to memorize on hard reasoning tasks; lower LR is the simplest reliable restraint move.",
                        confidence=0.8,
                        scope=task_profile.value,
                        evidence={"current_lr": lr, "suggested_lr": 5e-5, "threshold_lr": 1e-4},
                    )
                )
            elif lr > 5e-5:
                recs.append(
                    _rec(
                        severity=Severity.INFO,
                        action="consider_lower_lr",
                        message=f"LR={lr:.2e} is moderate. If you observe memorization (gap>1.5x), lower to lr=5e-5.",
                        rationale="In experiments, lr=5e-5 often improved generalization compared to higher LR.",
                        confidence=0.6,
                        scope=task_profile.value,
                        evidence={"current_lr": lr, "suggested_lr": 5e-5},
                    )
                )

        # (2) Targets: attention-only tends to be the efficient safe default
        if not targets:
            recs.append(
                _rec(
                    severity=Severity.WARNING,
                    action="set_target_modules",
                    message="LoRA target_modules is empty. For transformer reasoning tasks, start with attention projections: [q_proj, k_proj, v_proj, o_proj].",
                    rationale="Attention-only adaptation is a strong, efficient default for many reasoning-style tasks.",
                    confidence=0.7,
                    scope=task_profile.value,
                    evidence={"suggested_targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
                )
            )
        else:
            if has_mlp_targets:
                recs.append(
                    _rec(
                        severity=Severity.INFO,
                        action="prefer_attention_only",
                        message=(
                            "Config adapts MLP/FFN modules as well as attention. "
                            "For reasoning tasks, consider starting with attention-only to reduce overwrite risk and adapter size."
                        ),
                        rationale="In LoRA experiments, attention-only configurations often matched or beat broader target sets while using far fewer parameters.",
                        confidence=0.7,
                        scope=task_profile.value,
                        evidence={"mlp_targets": target_classes["mlp"], "attn_targets": target_classes["attn"]},
                    )
                )
            if not has_attn_targets and has_mlp_targets:
                recs.append(
                    _rec(
                        severity=Severity.WARNING,
                        action="include_attention_targets",
                        message="This config adapts MLP modules but not attention projections. Consider adding attention targets for reasoning tasks.",
                        rationale="Attention adaptation is often high-leverage for reasoning tasks because it controls token routing and context use.",
                        confidence=0.7,
                        scope=task_profile.value,
                        evidence={"targets": targets},
                    )
                )
            if has_other_targets:
                recs.append(
                    _rec(
                        severity=Severity.INFO,
                        action="review_target_modules",
                        message="Some target_modules are not recognized as attention or MLP projections. Verify they are intended.",
                        rationale="Unexpected target modules can silently change adapter size and behavior.",
                        confidence=0.4,
                        scope="config",
                        evidence={"other_targets": target_classes["other"]},
                    )
                )

        # (3) Rank: start small, scale up only if underfitting
        if r is None:
            recs.append(
                _rec(
                    severity=Severity.INFO,
                    action="set_rank",
                    message="LoRA rank (r) is not set. Start with r=8 as an efficiency-first default.",
                    rationale="Adapters often use only a small effective subspace; starting small reduces overfitting and cost.",
                    confidence=0.6,
                    scope=task_profile.value,
                    evidence={"suggested_r": 8},
                )
            )
        else:
            if r >= 32:
                recs.append(
                    _rec(
                        severity=Severity.INFO,
                        action="consider_reduce_rank",
                        message=f"Rank r={r} is large. If you're optimizing for efficiency, try r=8 or r=16 (especially with attention-only targets).",
                        rationale="In multiple LoRA studies, small ranks achieved similar accuracy with far fewer parameters.",
                        confidence=0.6,
                        scope=task_profile.value,
                        evidence={"current_r": r, "suggested_rs": [8, 16]},
                    )
                )
            elif r > 16:
                recs.append(
                    _rec(
                        severity=Severity.INFO,
                        action="consider_reduce_rank",
                        message=f"Rank r={r} may be more than needed. Consider r=8 or r=16 unless you observe underfitting.",
                        rationale="Starting smaller is often safer; scale up only if accuracy plateaus too low with low gap.",
                        confidence=0.5,
                        scope=task_profile.value,
                        evidence={"current_r": r, "suggested_rs": [8, 16]},
                    )
                )

        # (4) Alpha/r: secondary knob (mainly matters at higher LR)
        if a_over_r is not None and lr is not None:
            if lr > 1e-4 and a_over_r >= 1.0:
                recs.append(
                    _rec(
                        severity=Severity.INFO,
                        action="consider_lower_alpha_over_r",
                        message=f"At LR={lr:.2e}, alpha/r={a_over_r:.2f} may be high. Consider lowering alpha/r (e.g., 0.25) as an additional restraint.",
                        rationale="When LR is relatively high, reducing adapter amplitude can help prevent overwriting/memorization.",
                        confidence=0.5,
                        scope=task_profile.value,
                        evidence={"current_alpha_over_r": a_over_r, "suggested_alpha_over_r": 0.25, "lr": lr},
                    )
                )
            elif lr <= 5e-5:
                # At very low LR, alpha/r tends to be second-order; don't nag.
                pass

        # (5) Weight decay: not a primary LoRA lever in your findings; just sanity check
        if wd is not None and wd < 0:
            recs.append(
                _rec(
                    severity=Severity.ERROR,
                    action="fix_weight_decay",
                    message=f"weight_decay={wd} is negative. This is almost certainly a mistake.",
                    rationale="Negative weight decay in AdamW is uncommon and can destabilize training.",
                    confidence=0.9,
                    scope="config",
                    evidence={"weight_decay": wd},
                )
            )

    # ------------------------------------------------------------------
    # Task family: EASY_CLASSIFICATION
    # ------------------------------------------------------------------
    elif task_profile == TaskProfile.EASY_CLASSIFICATION:
        # For easy tasks, prioritize efficiency; many configs saturate.
        if r is not None and r > 16:
            recs.append(
                _rec(
                    severity=Severity.INFO,
                    action="reduce_rank_for_efficiency",
                    message=f"For easy classification tasks, r={r} is likely overkill. Consider r=4 or r=8.",
                    rationale="On many easy supervised tasks, performance saturates quickly; smaller adapters are usually sufficient.",
                    confidence=0.6,
                    scope=task_profile.value,
                    evidence={"current_r": r, "suggested_rs": [4, 8]},
                )
            )

        if has_mlp_targets:
            recs.append(
                _rec(
                    severity=Severity.INFO,
                    action="prefer_attention_only_for_efficiency",
                    message="Config includes MLP/FFN targets. For easy classification, consider attention-only to reduce params.",
                    rationale="If you can reach the same accuracy, fewer adapted modules is cheaper and often just as good.",
                    confidence=0.5,
                    scope=task_profile.value,
                    evidence={"mlp_targets": target_classes["mlp"]},
                )
            )

        # LR usually less critical here, but warn on extreme values.
        if lr is not None and lr > 5e-4:
            recs.append(
                _rec(
                    severity=Severity.WARNING,
                    action="lower_lr",
                    message=f"LR={lr:.2e} is unusually high. Consider lowering to avoid instability.",
                    rationale="Very high LR can destabilize fine-tuning even on easy tasks.",
                    confidence=0.7,
                    scope="config",
                    evidence={"current_lr": lr, "suggested_lr": 1e-4},
                )
            )

    # ------------------------------------------------------------------
    # UNKNOWN or GENERATION (fallback)
    # ------------------------------------------------------------------
    else:
        # If we can't classify, suggest a safe probe.
        recs.append(
            _rec(
                severity=Severity.INFO,
                action="calibration_probe",
                message="Task profile is unknown. Consider a short calibration run with a restraint-first config: attention-only, r=8, lr=5e-5.",
                rationale="Short probes help identify whether the run is sensitive (reasoning-like) or saturating (easy classification-like).",
                confidence=0.5,
                scope="unknown",
                evidence={"suggested": {"targets": ["q_proj", "k_proj", "v_proj", "o_proj"], "r": 8, "lr": 5e-5}},
            )
        )

    # If nothing triggered, emit a tiny positive signal.
    if not recs:
        recs.append(
            _rec(
                severity=Severity.INFO,
                action="config_ok",
                message="No obvious issues detected in this configuration.",
                rationale="Gradience check rules are conservative; absence of warnings does not guarantee success.",
                confidence=0.5,
                scope="config",
                evidence={},
            )
        )

    return recs


# ---------------------------------------------------------------------------
# Policy: run-time (monitoring)
# ---------------------------------------------------------------------------


def _dedup_recommendations(recs: List[Recommendation]) -> List[Recommendation]:
    """De-duplicate recommendations by action, keeping the highest severity.

    We keep this simple and stable: if two recs share the same ``action``,
    keep the one with higher severity. If severities tie, keep the first.
    """

    order = {
        Severity.CRITICAL: 0,
        Severity.ERROR: 1,
        Severity.WARNING: 2,
        Severity.INFO: 3,
    }

    best: Dict[str, Recommendation] = {}
    for r in recs:
        key = str(r.action)
        if key not in best:
            best[key] = r
            continue
        cur = best[key]
        if order.get(r.severity, 99) < order.get(cur.severity, 99):
            best[key] = r

    # Preserve a stable-ish order: severity then action
    out = list(best.values())
    out.sort(key=lambda x: (order.get(x.severity, 99), str(x.action)))
    return out


def check_run(
    config: Optional[ConfigSnapshot],
    signals: SignalSnapshot,
    *,
    gap_threshold: float = 1.5,
) -> List[Recommendation]:
    """Emit recommendations using both config and observed signals.

    This is the policy function intended for ``gradience monitor``.

    It purposely follows the research synthesis posture:
      - **Gap is king** (most actionable memorization signal)
      - spectral/structural metrics are *diagnostic* and should not pretend
        to be a predictor

    Parameters
    ----------
    config:
        The :class:`~gradience.vnext.types.ConfigSnapshot` from telemetry.
        May be ``None`` if run_start wasn't logged.
    signals:
        A :class:`~gradience.vnext.types.SignalSnapshot` from
        :meth:`~gradience.vnext.telemetry_reader.TelemetryReader.summarize`.
    gap_threshold:
        Threshold for ``test_ppl/train_ppl`` above which we warn about
        memorization.
    """

    recs: List[Recommendation] = []

    # Config-only heuristics still apply.
    if config is not None:
        recs.extend(check_config(config))

    # ------------------------------------------------------------------
    # Signal-based heuristics
    # ------------------------------------------------------------------

    gap = signals.gap
    train_ppl = signals.train.ppl
    test_ppl = signals.test.ppl
    test_acc = signals.test.accuracy
    sr = signals.stable_rank_mean

    # If we can't compute gap, tell the user what to fix in logging.
    if gap is None:
        if train_ppl is None or test_ppl is None:
            recs.append(
                _rec(
                    severity=Severity.INFO,
                    action="log_train_test_ppl",
                    message="Could not compute train/test gap (missing train/test PPL). Ensure telemetry logs eval metrics for both 'train' and 'test' splits.",
                    rationale="The train/test PPL ratio is the most actionable memorization signal in Gradience's restraint-first strategy.",
                    confidence=0.6,
                    scope="monitor",
                    evidence={"train_ppl": train_ppl, "test_ppl": test_ppl},
                )
            )
    else:
        if gap >= gap_threshold:
            # Memorization regime
            lr = config.optimizer.lr if config is not None else None
            # Suggest a specific LR only when we know it.
            suggested_lr = 5e-5

            msg = f"Train/test gap is {gap:.2f}x (threshold {gap_threshold:.2f}x): memorization risk."
            if lr is not None:
                msg += f" Current LR={lr:.2e}."

            recs.append(
                _rec(
                    severity=Severity.WARNING,
                    action="memorization_detected",
                    message=msg,
                    rationale="A large PPL gap indicates the adapter is fitting training data much better than test data. The first-line fix is increasing restraint (lower LR, narrower targets).",
                    confidence=0.8,
                    scope="monitor",
                    evidence={"gap": gap, "threshold": gap_threshold, "lr": lr},
                )
            )

            # Specific “next move” recommendation: LR first.
            if lr is None:
                recs.append(
                    _rec(
                        severity=Severity.WARNING,
                        action="lower_lr",
                        message=f"Lower learning rate as a restraint move (suggested lr≈{suggested_lr:.0e} for many reasoning-style fine-tunes).",
                        rationale="Across LoRA studies, LR was the dominant knob controlling memorization vs generalization.",
                        confidence=0.7,
                        scope="monitor",
                        evidence={"suggested_lr": suggested_lr},
                    )
                )
            else:
                # Only suggest lowering if we're above the known safe baseline.
                if lr > suggested_lr:
                    recs.append(
                        _rec(
                            severity=Severity.WARNING,
                            action="lower_lr",
                            message=f"Lower LR from {lr:.2e} → {suggested_lr:.0e} (restraint-first baseline) to reduce memorization.",
                            rationale="Lower LR reduces adapter update magnitude; this reliably reduced memorization in reasoning tasks.",
                            confidence=0.8,
                            scope="monitor",
                            evidence={"current_lr": lr, "suggested_lr": suggested_lr},
                        )
                    )

        # Possible underfitting heuristic (very conservative)
        # Low gap + low accuracy + very concentrated updates can indicate you
        # simply don't have enough adapter capacity or training budget.
        if gap is not None and gap < 1.2 and test_acc is not None:
            if test_acc < 0.2 and (sr is not None and sr < 2.0):
                recs.append(
                    _rec(
                        severity=Severity.INFO,
                        action="possible_underfitting",
                        message=(
                            f"Gap is low ({gap:.2f}x) but test accuracy is also low ({test_acc:.1%}). "
                            "This can indicate underfitting; consider increasing rank (e.g., r=16) or training longer."
                        ),
                        rationale="When restraint is very strong (low LR / low stable rank), the adapter may not have enough capacity to express the needed change.",
                        confidence=0.4,
                        scope="monitor",
                        evidence={"gap": gap, "test_accuracy": test_acc, "stable_rank_mean": sr},
                    )
                )

    # Efficiency nudge: low utilization on big ranks.
    if config is not None and config.lora.r is not None and sr is not None and config.lora.r > 0:
        util = float(sr) / float(config.lora.r)
        if util < 0.25 and config.lora.r >= 16:
            recs.append(
                _rec(
                    severity=Severity.INFO,
                    action="low_utilization",
                    message=f"Adapter utilization looks low (stable_rank≈{sr:.2f} of r={config.lora.r} → {util:.0%}). Consider reducing rank for efficiency.",
                    rationale="Low utilization suggests you are paying for rank you aren't using. Audit can refine a per-layer recommendation.",
                    confidence=0.5,
                    scope="monitor",
                    evidence={"stable_rank_mean": sr, "r": config.lora.r, "utilization": util},
                )
            )

    # ------------------------------------------------------------------
    # Audit-driven compression recommendations
    # ------------------------------------------------------------------

    def _nice_rank(x: int, *, max_rank: int) -> int:
        """Round `x` up to a "nice" LoRA rank (power-of-two-ish), capped at max_rank."""
        if x <= 0:
            return 0
        candidates = [2, 4, 8, 16, 32, 64, 128, 256]
        for c in candidates:
            if c >= x:
                return min(c, max_rank)
        return max_rank

    # If telemetry includes a LoRA audit payload, use it to make more concrete
    # compression suggestions (rather than generic "consider reducing rank").
    audit: Optional[Dict[str, Any]] = None
    try:
        if isinstance(signals.extras, dict):
            maybe = signals.extras.get("lora_audit")
            if isinstance(maybe, dict):
                audit = maybe
    except Exception:
        audit = None

    if config is not None and config.lora.r is not None and audit is not None:
        cur_r = int(config.lora.r)
        if cur_r > 0:
            # Pull audit summaries (best-effort)
            audit_util = audit.get("utilization_mean", None)
            audit_sr = audit.get("stable_rank_mean", None)
            energy_p90 = audit.get("energy_rank_90_p90", None)
            total_params = audit.get("total_lora_params", None)

            try:
                util_f = float(audit_util) if audit_util is not None else None
            except Exception:
                util_f = None
            try:
                sr_f = float(audit_sr) if audit_sr is not None else None
            except Exception:
                sr_f = None
            try:
                e90_f = float(energy_p90) if energy_p90 is not None else None
            except Exception:
                e90_f = None

            # Decide whether rank looks "oversized".
            oversized = False
            if util_f is not None and cur_r >= 16 and util_f < 0.25:
                oversized = True
            if e90_f is not None and cur_r >= 16 and e90_f > 0 and e90_f <= (cur_r / 2):
                oversized = True

            # Suggest a concrete rank using audit energy ranks when available.
            suggested_r: Optional[int] = None
            if oversized:
                if e90_f is not None and e90_f > 0:
                    base = int(math.ceil(e90_f))
                    suggested_r = _nice_rank(max(2, base), max_rank=cur_r)
                else:
                    # Fall back to the research default: r=8 is often enough.
                    suggested_r = 8 if cur_r > 8 else None

            if suggested_r is not None and suggested_r > 0 and suggested_r < cur_r:
                est_new_params: Optional[float] = None
                est_savings: Optional[float] = None
                try:
                    if isinstance(total_params, (int, float)) and cur_r > 0:
                        est_new_params = float(total_params) * (float(suggested_r) / float(cur_r))
                        est_savings = 1.0 - (float(suggested_r) / float(cur_r))
                except Exception:
                    est_new_params = None
                    est_savings = None

                msg = (
                    f"LoRA audit suggests rank is oversized (util≈{util_f:.0%} of r={cur_r})" if util_f is not None else f"LoRA audit suggests rank is oversized (r={cur_r})"
                )
                msg += f". Consider compressing r→{suggested_r} for efficiency."

                if est_savings is not None:
                    msg += f" (~{est_savings:.0%} fewer LoRA params)"

                recs.append(
                    _rec(
                        severity=Severity.INFO,
                        action="compress_rank",
                        message=msg,
                        rationale=(
                            "A LoRA audit measures how many effective directions the adapter actually uses. "
                            "If most layers reach 90% energy in a small k, you can often reduce r with minimal impact."
                        ),
                        confidence=0.6,
                        scope="monitor",
                        evidence={
                            "current_r": cur_r,
                            "suggested_r": suggested_r,
                            "utilization_mean": util_f,
                            "stable_rank_mean": sr_f,
                            "energy_rank_90_p90": e90_f,
                            "total_lora_params": total_params,
                            "estimated_new_lora_params": est_new_params,
                        },
                    )
                )

            # Optional: module-wise compression hint (MLP often wastes rank on some tasks).
            by_type = audit.get("by_type")
            if isinstance(by_type, dict):
                try:
                    mlp = by_type.get("mlp") if isinstance(by_type.get("mlp"), dict) else None
                    attn = by_type.get("attn") if isinstance(by_type.get("attn"), dict) else None
                    if mlp is not None and attn is not None:
                        mlp_util = float(mlp.get("utilization_mean", 0.0))
                        attn_util = float(attn.get("utilization_mean", 0.0))
                        target_classes = _classify_target_modules(config.lora.target_modules or [])
                        has_mlp_targets = len(target_classes.get("mlp") or []) > 0
                        if has_mlp_targets and mlp_util < 0.10 and attn_util >= (mlp_util * 2.0):
                            recs.append(
                                _rec(
                                    severity=Severity.INFO,
                                    action="compress_mlp_targets",
                                    message=(
                                        f"LoRA audit: MLP/FFN utilization is very low ({mlp_util:.0%}) compared to attention ({attn_util:.0%}). "
                                        "Consider attention-only, or lower rank specifically for MLP modules."
                                    ),
                                    rationale="When MLP deltas carry little energy, adapting them may be wasted capacity (and risk overwrite) without benefit.",
                                    confidence=0.5,
                                    scope="monitor",
                                    evidence={
                                        "mlp_utilization_mean": mlp_util,
                                        "attn_utilization_mean": attn_util,
                                        "mlp_targets": target_classes.get("mlp"),
                                    },
                                )
                            )
                except Exception:
                    # Don't let type parsing break monitor.
                    pass

    # --- audit-driven compression recommendation ---
    # If audit suggests a smaller global rank, recommend compression conservatively.
    try:
        extras = getattr(signals, 'extras', None) or {}
        lora_audit = extras.get('lora_audit') if isinstance(extras, dict) else None
    except Exception:
        lora_audit = None

    def _get_current_r(cfg):
        if cfg is None:
            return None
        # ConfigSnapshot dataclass
        try:
            l = getattr(cfg, 'lora', None)
            if l is not None and hasattr(l, 'r'):
                return int(l.r)
        except Exception:
            pass
        # Dict-like config
        try:
            l = cfg.get('lora') if hasattr(cfg, 'get') else None
            if isinstance(l, dict) and 'r' in l:
                return int(l['r'])
        except Exception:
            pass
        return None

    current_r = _get_current_r(config)
    if isinstance(lora_audit, dict) and current_r is not None:
        s_med = lora_audit.get('suggested_r_global_median')
        s_p90 = lora_audit.get('suggested_r_global_90')
        util = lora_audit.get('utilization_mean')
        try:
            s_med = int(s_med) if s_med is not None else None
        except Exception:
            s_med = None
        try:
            s_p90 = int(s_p90) if s_p90 is not None else None
        except Exception:
            s_p90 = None
        try:
            util = float(util) if util is not None else None
        except Exception:
            util = None

        # Conservative trigger
        if util is not None and util < 0.25 and current_r >= 8 and s_med is not None and s_med < current_r:
            msg = (
                f"Audit suggests r={s_med} for most layers (median); "
                f"r={s_p90} covers worst-case layers at 90% energy. "
                f"Current r={current_r} → consider trying r={s_med}."
            )
            recs.append(Recommendation(
                severity=Severity.INFO,
                action='compress_rank',
                message=msg,
                rationale='Low utilization suggests adapter capacity is overprovisioned; compressing rank can reduce params with minimal impact.',
                confidence=0.7,
                scope=str(getattr(config, 'task_profile', 'unknown')) if config is not None else 'unknown',
                evidence={
                    'utilization_mean': util,
                    'suggested_r_global_median': s_med,
                    'suggested_r_global_90': s_p90,
                    'current_r': current_r,
                },
            ))

    # --- suppress config_ok when actionable recs exist ---
    # If we have any actionable recommendation, drop the noise-only config_ok.
    try:
        has_actionable = any(getattr(r, 'action', None) != 'config_ok' for r in recs)
    except Exception:
        has_actionable = False
    if has_actionable:
        recs = [r for r in recs if getattr(r, 'action', None) != 'config_ok']

    return _dedup_recommendations(recs)
