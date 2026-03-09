#!/usr/bin/env python3
"""
Study 16 Re-Audit: Re-run merge audits with updated verdict engine.

Re-audits all 6 Study 16 adapter pairs using the updated 6-branch verdict
decision tree (Frobenius imbalance detection added as Branch 0).  Generates
merge recommendations for each pair and compares old → new verdicts.

This is the CPU-only prerequisite before re-running merge ablations:
  - Downloads adapters from HuggingFace (if not cached)
  - Runs merge_audit() with the corrected verdict engine
  - Generates per-pair recommendations via recommend_merge()
  - Produces a diff showing which verdicts changed
  - Writes JSON + markdown summary

Output structure:
    results/study16_merge_ablation/reaudit/
        {pair_id}/
            merge_audit.json
            merge_audit.md
            recommendation.json
        _summary/
            reaudit_summary.json
            reaudit_summary.md
            verdict_migration.json

Usage:
    python scripts/study16_reaudit.py --verbose
    python scripts/study16_reaudit.py --verbose --pairs pair_03 pair_04
    python scripts/study16_reaudit.py --verbose --no-download  # skip HF, use cache only
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gradience.vnext.merge import merge_audit, recommend_merge
from gradience.vnext.merge.verdicts import VerdictThresholds

# ---------------------------------------------------------------------------
# Pair definitions (from study16_merge_ablation.py)
# ---------------------------------------------------------------------------


@dataclass
class AdapterPairDef:
    """Definition of an adapter pair for the ablation."""

    pair_id: str
    repo_a: str
    repo_b: str
    label_a: str
    label_b: str
    task_a: str
    task_b: str
    expected_verdict: str


PAIRS: list[AdapterPairDef] = [
    AdapterPairDef(
        pair_id="pair_01_metamath_x_openwebmath16",
        repo_a="LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32",
        repo_b="LoRA-TMLR-2024/openwebmath-lora-rank-16-20B-tokens",
        label_a="metamath-r16",
        label_b="openwebmath-r16",
        task_a="math",
        task_b="math",
        expected_verdict="redundant",
    ),
    AdapterPairDef(
        pair_id="pair_02_metamath_x_magicoder",
        repo_a="LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32",
        repo_b="LoRA-TMLR-2024/magicoder-lora-rank-16-alpha-32",
        label_a="metamath-r16",
        label_b="magicoder-r16",
        task_a="math",
        task_b="code",
        expected_verdict="moderate",
    ),
    AdapterPairDef(
        pair_id="pair_03_magicoder_x_btgenbot",
        repo_a="LoRA-TMLR-2024/magicoder-lora-rank-16-alpha-32",
        repo_b="AIRLab-POLIMI/llama-2-7b-chat-hf-btgenbot-adapter",
        label_a="magicoder-r16",
        label_b="btgenbot-r8",
        task_a="code",
        task_b="chat",
        expected_verdict="conflicting",
    ),
    AdapterPairDef(
        pair_id="pair_04_openwebmath64_x_btgenbot",
        repo_a="LoRA-TMLR-2024/openwebmath-lora-rank-64-20B-tokens",
        repo_b="AIRLab-POLIMI/llama-2-7b-chat-hf-btgenbot-adapter",
        label_a="openwebmath-r64",
        label_b="btgenbot-r8",
        task_a="math",
        task_b="chat",
        expected_verdict="imbalanced",
    ),
    AdapterPairDef(
        pair_id="pair_05_metamath_x_llm2vec",
        repo_a="LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32",
        repo_b="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised",
        label_a="metamath-r16",
        label_b="llm2vec-r16",
        task_a="math",
        task_b="classification",
        expected_verdict="conflicting",
    ),
    AdapterPairDef(
        pair_id="pair_06_catsubcat_x_btgenbot",
        repo_a="shivanikerai/Llama-2-7b-chat-hf-adapter-cat-subcat-mapping-v2.0",
        repo_b="AIRLab-POLIMI/llama-2-7b-chat-hf-btgenbot-adapter",
        label_a="catsubcat-r16",
        label_b="btgenbot-r8",
        task_a="chat",
        task_b="chat",
        expected_verdict="mixed",
    ),
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_adapter(repo_id: str, cache_dir: Path) -> Path:
    """Download a PEFT adapter from HuggingFace Hub.  Returns local path."""
    from huggingface_hub import snapshot_download

    local_dir = cache_dir / repo_id.replace("/", "--")

    config_candidates = list(local_dir.glob("adapter_config*"))
    weight_candidates = list(local_dir.glob("adapter_model*"))
    if config_candidates and weight_candidates:
        return local_dir

    print(f"    Downloading {repo_id} ...", end="", flush=True)
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
    print(" done")
    return local_dir


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PairAuditResult:
    """Compact record of one re-audit for summary reporting."""

    pair_id: str
    label_a: str
    label_b: str
    task_a: str
    task_b: str
    expected_verdict: str
    overall_verdict: str
    compatibility_score: float
    n_layers: int
    verdict_counts: dict[str, int]
    recommendation_strategy: str | None = None
    recommendation_risk: str | None = None
    n_compress: int = 0
    audit_time_s: float = 0.0
    # Diff fields
    old_verdict: str | None = None
    old_layer_verdict_counts: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "pair_id": self.pair_id,
            "label": f"{self.label_a} x {self.label_b}",
            "task_a": self.task_a,
            "task_b": self.task_b,
            "expected_verdict": self.expected_verdict,
            "overall_verdict": self.overall_verdict,
            "compatibility_score": round(self.compatibility_score, 4),
            "n_layers": self.n_layers,
            "verdict_counts": self.verdict_counts,
            "audit_time_s": round(self.audit_time_s, 2),
        }
        if self.recommendation_strategy is not None:
            d["recommendation"] = {
                "overall_strategy": self.recommendation_strategy,
                "overall_risk": self.recommendation_risk,
                "n_layers_needing_compression": self.n_compress,
            }
        if self.old_verdict is not None:
            d["old_verdict"] = self.old_verdict
            d["verdict_changed"] = self.old_verdict != self.overall_verdict
            if self.old_layer_verdict_counts:
                d["old_layer_verdict_counts"] = self.old_layer_verdict_counts
        return d


# ---------------------------------------------------------------------------
# Core audit + recommend for one pair
# ---------------------------------------------------------------------------


def run_pair_audit(
    pair_def: AdapterPairDef,
    cache_dir: Path,
    output_dir: Path,
    old_results_dir: Path,
    counter: str,
    do_recommend: bool = True,
    skip_download: bool = False,
    verbose: bool = False,
) -> PairAuditResult | None:
    """Audit one pair, generate recommendation, compare to old verdict."""

    pair_dir = output_dir / pair_def.pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{counter}] {pair_def.label_a} x {pair_def.label_b}", end="", flush=True)

    # --- Resolve adapter paths ---
    if skip_download:
        dir_a = cache_dir / pair_def.repo_a.replace("/", "--")
        dir_b = cache_dir / pair_def.repo_b.replace("/", "--")
        if not dir_a.exists() or not dir_b.exists():
            print("  [MISSING — not in cache]")
            return None
    else:
        try:
            dir_a = download_adapter(pair_def.repo_a, cache_dir)
            dir_b = download_adapter(pair_def.repo_b, cache_dir)
        except Exception as e:
            print(f"  [DOWNLOAD FAILED: {e}]")
            return None

    # --- Run audit ---
    print(" auditing...", end="", flush=True)
    start = time.monotonic()

    try:
        report = merge_audit(
            adapter_a_dir=str(dir_a),
            adapter_b_dir=str(dir_b),
            output_dir=str(pair_dir),
            verbose=False,
        )
    except Exception as e:
        print(f" [AUDIT FAILED: {e}]")
        return None

    elapsed = time.monotonic() - start
    verdict = report.aggregate.get("overall_verdict", "unknown")
    score = report.aggregate.get("compatibility_score", 0.0)

    # Per-layer verdict counts
    verdict_counts: dict[str, int] = Counter()
    for lv in report.layer_verdicts:
        verdict_counts[lv["verdict"]] += 1

    print(f" {verdict.upper()} (score={score:.3f}) [{elapsed:.1f}s]", end="")

    # --- Recommendation ---
    rec_strategy = None
    rec_risk = None
    n_compress = 0

    if do_recommend:
        try:
            rec = recommend_merge(report)
            rec_strategy = rec.overall_strategy
            rec_risk = rec.overall_risk
            n_compress = rec.n_layers_needing_compression

            rec_path = pair_dir / "recommendation.json"
            with open(rec_path, "w") as f:
                json.dump(rec.to_dict(), f, indent=2)

            # Count per-layer strategy distribution for the report
            layer_strats = Counter(lr.strategy for lr in rec.layer_recommendations)
            strat_summary = ", ".join(f"{s}:{n}" for s, n in layer_strats.most_common())
            print(f" → {rec_strategy} (risk={rec_risk}, {strat_summary})", end="")
        except Exception as e:
            print(f" → rec FAILED: {e}", end="")

    # --- Diff against old audit ---
    old_verdict = None
    old_layer_verdict_counts = None
    old_audit_json = old_results_dir / pair_def.pair_id / "audit" / "merge_audit.json"

    if old_audit_json.exists():
        try:
            with open(old_audit_json) as f:
                old_data = json.load(f)
            old_verdict = old_data.get("aggregate", {}).get("overall_verdict")
            old_layer_counts: dict[str, int] = Counter()
            for lv in old_data.get("per_layer", []):
                old_layer_counts[lv.get("verdict", "unknown")] += 1
            old_layer_verdict_counts = dict(old_layer_counts)

            if old_verdict and old_verdict != verdict:
                # Show what changed at the layer level
                old_imb = old_layer_counts.get("imbalanced", 0)
                new_imb = verdict_counts.get("imbalanced", 0)
                print(f"  [CHANGED: {old_verdict}→{verdict}, imbalanced layers: {old_imb}→{new_imb}]", end="")
            elif old_verdict:
                old_imb = old_layer_counts.get("imbalanced", 0)
                new_imb = verdict_counts.get("imbalanced", 0)
                if old_imb != new_imb:
                    print(f"  [layers changed: imbalanced {old_imb}→{new_imb}]", end="")
        except (json.JSONDecodeError, KeyError):
            pass

    print()  # newline

    return PairAuditResult(
        pair_id=pair_def.pair_id,
        label_a=pair_def.label_a,
        label_b=pair_def.label_b,
        task_a=pair_def.task_a,
        task_b=pair_def.task_b,
        expected_verdict=pair_def.expected_verdict,
        overall_verdict=verdict,
        compatibility_score=score,
        n_layers=len(report.layer_verdicts),
        verdict_counts=dict(verdict_counts),
        recommendation_strategy=rec_strategy,
        recommendation_risk=rec_risk,
        n_compress=n_compress,
        audit_time_s=elapsed,
        old_verdict=old_verdict,
        old_layer_verdict_counts=old_layer_verdict_counts,
    )


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def write_summary(
    results: list[PairAuditResult],
    summary_dir: Path,
) -> None:
    """Write JSON, markdown summary, and verdict migration report."""
    summary_dir.mkdir(parents=True, exist_ok=True)

    overall_verdicts = Counter(r.overall_verdict for r in results)
    layer_verdicts_total: Counter = Counter()
    for r in results:
        layer_verdicts_total.update(r.verdict_counts)

    rec_strategies = Counter(r.recommendation_strategy for r in results if r.recommendation_strategy is not None)
    rec_risks = Counter(r.recommendation_risk for r in results if r.recommendation_risk is not None)

    has_diff = any(r.old_verdict is not None for r in results)

    # --- Summary JSON ---
    summary: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "Study 16 re-audit with 6-branch verdict engine (Frobenius imbalance)",
        "n_pairs": len(results),
        "overall_verdict_distribution": dict(overall_verdicts),
        "layer_verdict_distribution": dict(layer_verdicts_total),
        "total_layers": sum(layer_verdicts_total.values()),
        "recommendation_strategy_distribution": dict(rec_strategies),
        "recommendation_risk_distribution": dict(rec_risks),
        "n_needing_compression": sum(1 for r in results if r.n_compress > 0),
        "mean_audit_time_s": round(sum(r.audit_time_s for r in results) / max(len(results), 1), 2),
        "per_pair": [r.to_dict() for r in results],
    }

    with open(summary_dir / "reaudit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Verdict migration JSON ---
    if has_diff:
        migrations = []
        for r in results:
            if r.old_verdict is not None:
                entry: dict[str, Any] = {
                    "pair_id": r.pair_id,
                    "label": f"{r.label_a} x {r.label_b}",
                    "old_verdict": r.old_verdict,
                    "new_verdict": r.overall_verdict,
                    "changed": r.old_verdict != r.overall_verdict,
                }
                if r.old_layer_verdict_counts:
                    entry["old_layer_verdicts"] = r.old_layer_verdict_counts
                entry["new_layer_verdicts"] = r.verdict_counts
                migrations.append(entry)

        migration_summary = {
            "n_compared": len(migrations),
            "n_changed": sum(1 for m in migrations if m["changed"]),
            "n_unchanged": sum(1 for m in migrations if not m["changed"]),
            "changes": [m for m in migrations if m["changed"]],
            "all": migrations,
        }

        with open(summary_dir / "verdict_migration.json", "w") as f:
            json.dump(migration_summary, f, indent=2)

    # --- Markdown summary ---
    lines = [
        "# Study 16 Re-Audit: Updated Verdict Engine",
        "",
        f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "Re-audit of all Study 16 adapter pairs with the corrected 6-branch",
        "verdict decision tree.  Branch 0 now detects Frobenius imbalance",
        "(norm ratio > 5x) before checking for orthogonal subspaces, catching",
        "magnitude mismatches that were previously invisible.",
        "",
        f"**Pairs audited: {len(results)}**",
        "",
        "## Overall Verdict Distribution",
        "",
        "| Verdict | Count |",
        "|---------|------:|",
    ]

    for v in ["safe", "redundant", "imbalanced", "conflicting"]:
        count = overall_verdicts.get(v, 0)
        lines.append(f"| {v.upper()} | {count} |")

    lines.extend(
        [
            "",
            "## Per-Layer Verdict Distribution",
            "",
            "| Verdict | Layers | % |",
            "|---------|-------:|--:|",
        ]
    )

    total_layers = sum(layer_verdicts_total.values())
    for v in ["safe", "redundant", "imbalanced", "conflicting"]:
        count = layer_verdicts_total.get(v, 0)
        pct = 100 * count / max(total_layers, 1)
        lines.append(f"| {v.upper()} | {count} | {pct:.0f}% |")

    # Diff section
    if has_diff:
        changed = [r for r in results if r.old_verdict is not None and r.old_verdict != r.overall_verdict]
        layer_changed = [
            r
            for r in results
            if r.old_layer_verdict_counts is not None and r.old_layer_verdict_counts != r.verdict_counts
        ]
        compared = [r for r in results if r.old_verdict is not None]

        lines.extend(
            [
                "",
                "## Changes vs. Original Study 16 Audits",
                "",
                f"Compared: {len(compared)} pairs",
                f"Overall verdict changed: **{len(changed)}**",
                f"Layer-level verdicts changed: **{len(layer_changed)}**",
            ]
        )

        if changed or layer_changed:
            lines.extend(
                [
                    "",
                    "| Pair | Old Overall | New Overall | Old Layers | New Layers |",
                    "|------|-------------|-------------|------------|------------|",
                ]
            )
            for r in results:
                if r.old_verdict is None:
                    continue
                # Only show rows where something changed
                overall_changed = r.old_verdict != r.overall_verdict
                layers_changed = (
                    r.old_layer_verdict_counts is not None and r.old_layer_verdict_counts != r.verdict_counts
                )
                if not overall_changed and not layers_changed:
                    continue

                def _fmt_counts(d: dict[str, int]) -> str:
                    parts = []
                    for v in ["safe", "imbalanced", "redundant", "conflicting"]:
                        if d.get(v, 0) > 0:
                            parts.append(f"{v[0].upper()}:{d[v]}")
                    return " ".join(parts)

                old_vc = _fmt_counts(r.old_layer_verdict_counts or {})
                new_vc = _fmt_counts(r.verdict_counts)
                old_v = r.old_verdict
                new_v = f"**{r.overall_verdict}**" if overall_changed else r.overall_verdict

                lines.append(f"| {r.label_a} x {r.label_b} | {old_v} | {new_v} | {old_vc} | {new_vc} |")

    # Recommendation section
    if rec_strategies:
        lines.extend(
            [
                "",
                "## Merge Recommendations",
                "",
                "| Pair | Overall Strategy | Risk | Compress? | Layer Strategies |",
                "|------|-----------------|------|-----------|-----------------|",
            ]
        )
        for r in results:
            if r.recommendation_strategy is None:
                continue
            compress = "yes" if r.n_compress > 0 else "no"
            # Reconstruct layer strategy distribution from verdict_counts
            # (the detailed per-layer data is in the recommendation.json)
            lines.append(
                f"| {r.label_a} x {r.label_b} | {r.recommendation_strategy} | "
                f"{r.recommendation_risk} | {compress} | — |"
            )

    # Per-pair detail
    lines.extend(
        [
            "",
            "## Per-Pair Detail",
            "",
            "| Pair | Tasks | Expected | Actual | Score | Layers | Rec Risk |",
            "|------|-------|----------|--------|------:|-------:|----------|",
        ]
    )
    for r in results:
        risk = r.recommendation_risk or "—"
        lines.append(
            f"| {r.label_a} x {r.label_b} | {r.task_a}/{r.task_b} | "
            f"{r.expected_verdict} | {r.overall_verdict} | "
            f"{r.compatibility_score:.3f} | {r.n_layers} | {risk} |"
        )

    lines.append("")

    with open(summary_dir / "reaudit_summary.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\n  Summary written to {summary_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 16 Re-Audit: Re-run merge audits with updated verdict engine")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/study16_merge_ablation/reaudit"),
        help="Directory for re-audit output (default: results/study16_merge_ablation/reaudit)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "gradience" / "adapters",
        help="Cache directory for downloaded adapters",
    )
    parser.add_argument(
        "--old-results-dir",
        type=Path,
        default=Path("results/study16_merge_ablation"),
        help="Directory containing original Study 16 audit results for diff",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        nargs="*",
        default=None,
        help="Specific pair IDs to run (substring match, e.g. 'pair_03 pair_04')",
    )
    parser.add_argument(
        "--no-recommend",
        action="store_true",
        help="Skip recommendation generation (audit only)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip HuggingFace downloads — use cached adapters only",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_download:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Filter pairs
    pairs_to_run = PAIRS
    if args.pairs:
        pairs_to_run = [p for p in PAIRS if any(filt in p.pair_id for filt in args.pairs)]
        if not pairs_to_run:
            available = [p.pair_id for p in PAIRS]
            print(f"No matching pairs. Available: {available}")
            return

    # Print thresholds so the user can verify the Frobenius branch is active
    thresholds = VerdictThresholds()
    print(f"Study 16 Re-Audit — {len(pairs_to_run)} pairs")
    print("  Verdict engine: 6-branch (Frobenius imbalance enabled)")
    print(
        f"  Thresholds: imbalanced_frob={thresholds.imbalanced_frob}, "
        f"imbalanced={thresholds.imbalanced}, "
        f"low_overlap={thresholds.low_overlap}, "
        f"high_overlap={thresholds.high_overlap}"
    )
    print(f"  Output:     {args.output_dir}")
    print(f"  Cache:      {args.cache_dir}")
    print(f"  Old results: {args.old_results_dir}")
    print(f"  Download:   {'yes' if not args.no_download else 'no (cache only)'}")
    print(f"  Recommend:  {'yes' if not args.no_recommend else 'no'}")
    print()

    total_start = time.monotonic()
    results: list[PairAuditResult] = []
    n_failed = 0

    for i, pair_def in enumerate(pairs_to_run):
        result = run_pair_audit(
            pair_def,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            old_results_dir=args.old_results_dir,
            counter=f"{i + 1}/{len(pairs_to_run)}",
            do_recommend=not args.no_recommend,
            skip_download=args.no_download,
            verbose=args.verbose,
        )
        if result:
            results.append(result)
        else:
            n_failed += 1

    elapsed = time.monotonic() - total_start

    # --- Console summary ---
    print(f"\n{'=' * 70}")
    print(f"Study 16 Re-Audit complete: {len(results)} pairs in {elapsed:.1f}s")
    if n_failed:
        print(f"  Failed/missing: {n_failed}")

    if results:
        # Overall verdict distribution
        verdict_counts = Counter(r.overall_verdict for r in results)
        print("\n  Overall verdict distribution:")
        for v in ["safe", "redundant", "imbalanced", "conflicting"]:
            count = verdict_counts.get(v, 0)
            print(f"    {v.upper():>12s}: {count}")

        # Layer-level distribution
        layer_totals: Counter = Counter()
        for r in results:
            layer_totals.update(r.verdict_counts)
        total_layers = sum(layer_totals.values())
        print(f"\n  Per-layer verdict distribution ({total_layers} layers total):")
        for v in ["safe", "redundant", "imbalanced", "conflicting"]:
            count = layer_totals.get(v, 0)
            pct = 100 * count / max(total_layers, 1)
            print(f"    {v.upper():>12s}: {count:5d} ({pct:.0f}%)")

        # Diff summary
        changed = [r for r in results if r.old_verdict is not None and r.old_verdict != r.overall_verdict]
        compared = [r for r in results if r.old_verdict is not None]
        if compared:
            print("\n  Verdict changes vs original Study 16:")
            print(f"    Compared: {len(compared)} | Changed: {len(changed)}")
            for r in changed:
                old_imb = (r.old_layer_verdict_counts or {}).get("imbalanced", 0)
                new_imb = r.verdict_counts.get("imbalanced", 0)
                print(
                    f"    {r.label_a} x {r.label_b}: "
                    f"{r.old_verdict} → {r.overall_verdict} "
                    f"(imbalanced layers: {old_imb} → {new_imb})"
                )

            # Also show pairs where layers changed but overall didn't
            layer_changed_only = [
                r
                for r in results
                if r.old_verdict is not None
                and r.old_verdict == r.overall_verdict
                and r.old_layer_verdict_counts is not None
                and r.old_layer_verdict_counts != r.verdict_counts
            ]
            if layer_changed_only:
                print("\n    Layer-level changes (overall verdict unchanged):")
                for r in layer_changed_only:
                    old_imb = (r.old_layer_verdict_counts or {}).get("imbalanced", 0)
                    new_imb = r.verdict_counts.get("imbalanced", 0)
                    print(f"    {r.label_a} x {r.label_b}: imbalanced layers: {old_imb} → {new_imb}")

        # Write full summary
        summary_dir = args.output_dir / "_summary"
        write_summary(results, summary_dir)

    print(f"\nDone. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
