#!/usr/bin/env python3
"""
Task-Balanced Extension Pass for Public Ecosystem Census
========================================================

CPU-only follow-up that reuses the existing ecosystem census pipeline and
adds targeted cohort rebalancing toward underrepresented task categories.

This runner implements a bounded extension pass:
1) Freeze pilot-plus baseline
2) Define task-balanced expansion targets
3) Run targeted discovery/download/audit
4) Re-run core census analysis stack on augmented cohort
5) Compare pre/post architecture-vs-task interpretation
6) Emit a bounded extension memo + strategy summary
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure repo root import path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import ecosystem_census as census


TARGET_TASKS = ("code", "math_reasoning", "domain_specialist", "classification")
DEFAULT_TARGET_COUNTS = {
    "code": 8,
    "math_reasoning": 8,
    "domain_specialist": 6,
    "classification": 5,
}

TASK_SEARCH_TERMS = {
    "code": ["code", "coder", "python", "sql"],
    "math_reasoning": ["math", "gsm", "reason", "proof"],
    "domain_specialist": ["medical", "biomedical", "legal", "law", "finance", "clinical", "data"],
    "classification": ["classification", "sentiment", "topic", "ner", "classify"],
}

CONFIDENCE_ORDER = {"high": 3, "medium": 2, "low": 1}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  wrote {path}")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"  wrote {path}")


def _to_fingerprint(row: dict[str, Any]) -> census.SpectralFingerprint:
    return census.SpectralFingerprint(
        repo_id=row["repo_id"],
        architecture_family=row["architecture_family"],
        task_category=row["task_category"],
        nominal_rank=int(row.get("nominal_rank", 0) or 0),
        stable_rank_mean=float(row.get("stable_rank_mean", 0.0) or 0.0),
        stable_rank_std=float(row.get("stable_rank_std", 0.0) or 0.0),
        utilization_mean=float(row.get("utilization_mean", 0.0) or 0.0),
        energy_rank_90_p50=float(row.get("energy_rank_90_p50", 0.0) or 0.0),
        entropy_erank_mean=float(row.get("entropy_erank_mean", 0.0) or 0.0),
        attn_stable_rank_mean=float(row.get("attn_stable_rank_mean", 0.0) or 0.0),
        mlp_stable_rank_mean=float(row.get("mlp_stable_rank_mean", 0.0) or 0.0),
        attn_utilization_mean=float(row.get("attn_utilization_mean", 0.0) or 0.0),
        mlp_utilization_mean=float(row.get("mlp_utilization_mean", 0.0) or 0.0),
        edge_gap_mean=float(row.get("edge_gap_mean", 0.0) or 0.0),
    )


def _distribution_str(counter: Counter[str]) -> str:
    if not counter:
        return "{}"
    ordered = dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))
    return json.dumps(ordered, sort_keys=False)


def load_baseline_state(output_dir: Path) -> dict[str, Any]:
    records = _read_json(output_dir / "adapter_records.json", [])
    fp_rows = _read_json(output_dir / "fingerprint_table.json", [])
    fingerprints = [_to_fingerprint(r) for r in fp_rows]
    p3 = _read_json(output_dir / "architecture_task_decomposition.json", {})
    p4 = _read_json(output_dir / "clustering_results.json", {})
    p5 = _read_json(output_dir / "module_type_replication.json", {})
    p2 = _read_json(output_dir / "confound_assessment.json", {})
    p2b_rank = _read_json(output_dir / "rank_matched_analysis.json", {})
    p2b_task = _read_json(output_dir / "task_label_confidence.json", {})
    p2b_variant = _read_json(output_dir / "variant_robustness.json", {})
    manifest = _read_json(output_dir / "manifest.json", {})
    audit_cache = _read_json(output_dir / "audit_cache.json", {})

    arch_counts = Counter(fp.architecture_family for fp in fingerprints)
    task_counts = Counter(fp.task_category for fp in fingerprints)

    return {
        "records": records,
        "fingerprints": fingerprints,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "p2": p2,
        "p2b_rank": p2b_rank,
        "p2b_task": p2b_task,
        "p2b_variant": p2b_variant,
        "manifest": manifest,
        "audit_cache": audit_cache,
        "arch_counts": arch_counts,
        "task_counts": task_counts,
    }


def stage_a_baseline_freeze(output_dir: Path, baseline: dict[str, Any]) -> None:
    p3 = baseline["p3"]
    p4 = baseline["p4"]
    p5 = baseline["p5"]
    p2 = baseline["p2"]
    arch_counts = baseline["arch_counts"]
    task_counts = baseline["task_counts"]
    fps = baseline["fingerprints"]
    summary = p3.get("summary", {})

    limitation = (
        "Task balance remains the main limitation if target task categories are sparse "
        "or if one category dominates kNN neighborhoods."
    )
    if task_counts:
        top_task, top_n = task_counts.most_common(1)[0]
        if len(fps) > 0 and (top_n / len(fps)) >= 0.55:
            limitation = (
                f"Task imbalance remains severe: {top_task} dominates ({top_n}/{len(fps)}), "
                "which can suppress interpretable task-conditioned effects."
            )

    baseline_json = {
        "snapshot_date": _now_iso(),
        "baseline_manifest": baseline["manifest"],
        "counts": {
            "n_records": len(baseline["records"]),
            "n_fingerprints": len(fps),
            "architecture_counts": dict(arch_counts),
            "task_counts": dict(task_counts),
        },
        "architecture_task_summary": summary,
        "clustering": {
            "architecture_knn_purity_mean": p4.get("architecture_knn_purity", {}).get("mean"),
            "task_knn_purity_mean": p4.get("task_knn_purity", {}).get("mean"),
            "architecture_random_purity": p4.get("baselines", {}).get("architecture_random_purity"),
            "task_random_purity": p4.get("baselines", {}).get("task_random_purity"),
        },
        "module_type_replication": {
            "sign_test_interpretation": p5.get("sign_test_interpretation"),
            "replicates_encoder_pattern": p5.get("replicates_encoder_pattern"),
        },
        "confound_overall": p2.get("overall", {}),
        "task_balance_limitation": limitation,
    }

    md_lines = [
        "# Task-Balanced Extension Baseline",
        "",
        f"- Snapshot date (UTC): {baseline_json['snapshot_date']}",
        f"- Baseline fingerprints: {len(fps)}",
        f"- Architecture counts: `{_distribution_str(arch_counts)}`",
        f"- Task counts: `{_distribution_str(task_counts)}`",
        "",
        "## Frozen Pilot-Plus Signals",
        "",
        f"- Mean architecture eta-squared: {summary.get('mean_architecture_eta_sq', 'N/A')}",
        f"- Mean task eta-squared: {summary.get('mean_task_eta_sq', 'N/A')}",
        f"- Dominant factor: {summary.get('dominant_factor', 'N/A')}",
        f"- Architecture kNN purity (mean): {p4.get('architecture_knn_purity', {}).get('mean', 'N/A')}",
        f"- Task kNN purity (mean): {p4.get('task_knn_purity', {}).get('mean', 'N/A')}",
        f"- Module-type sign result: {p5.get('sign_test_interpretation', 'N/A')}",
        "",
        "## Limitation",
        "",
        baseline_json["task_balance_limitation"],
        "",
    ]

    _write_json(output_dir / "task_balanced_extension_baseline.json", baseline_json)
    _write_text(output_dir / "task_balanced_extension_baseline.md", "\n".join(md_lines))


def _infer_task_label_confidence(
    tags: list[str] | None,
    inferred_task: str,
    task_category: str,
    repo_id: str,
) -> tuple[str, str]:
    """
    Assign task-label confidence and source.

    High:
      - explicit task tags or strongly specific domain tags + non-unknown category
    Medium:
      - category inferred from repo-name keyword heuristics
    Low:
      - general_unknown or weak/no signal
    """
    tags_set = {t.lower() for t in (tags or [])}
    repo_lower = repo_id.lower()

    if task_category == "general_unknown":
        return "low", "metadata"

    explicit_tag_map = {
        "code": {"code", "text-generation", "text2text-generation"},
        "math_reasoning": {"math", "reasoning"},
        "domain_specialist": {"medical", "clinical", "legal", "law", "finance", "biomedical"},
        "classification": {"text-classification", "classification", "sentiment"},
    }
    explicit = explicit_tag_map.get(task_category, set())
    if any(tag in tags_set for tag in explicit):
        return "high", "tags"

    keyword_map = {
        "code": ("code", "coder", "python", "sql"),
        "math_reasoning": ("math", "gsm", "reason"),
        "domain_specialist": ("medical", "legal", "clinical", "bio", "law", "finance", "data"),
        "classification": ("classify", "classification", "sentiment", "topic", "ner"),
    }
    kws = keyword_map.get(task_category, tuple())
    if any(k in repo_lower for k in kws):
        return "medium", "metadata"

    if inferred_task in {"code", "math", "medical", "legal", "data", "classification"}:
        return "medium", "metadata"
    return "low", "metadata"


def stage_b_extension_design(
    output_dir: Path,
    baseline: dict[str, Any],
    families: list[str],
    target_counts: dict[str, int],
    family_min_target: int,
) -> dict[str, Any]:
    task_counts: Counter[str] = baseline["task_counts"]
    arch_counts: Counter[str] = baseline["arch_counts"]

    task_targets: dict[str, Any] = {}
    for t in TARGET_TASKS:
        current = int(task_counts.get(t, 0))
        target = int(target_counts.get(t, 0))
        task_targets[t] = {
            "current_count": current,
            "target_count": target,
            "needed": max(target - current, 0),
            "search_strategy_notes": TASK_SEARCH_TERMS.get(t, []),
            "high_confidence_labeling_criteria": (
                "high=explicit task/domain tag; medium=repo-name heuristic with non-unknown task; "
                "low=general/ambiguous"
            ),
        }

    fam_targets: dict[str, Any] = {}
    for fam in families:
        current = int(arch_counts.get(fam, 0))
        fam_targets[fam] = {
            "current_count": current,
            "target_min_count": family_min_target,
            "needed_for_parity": max(family_min_target - current, 0),
        }

    design_json = {
        "snapshot_date": _now_iso(),
        "cohort_mode": "task-balanced extension",
        "target_count_rules": {
            "rule_a": ">=3 task categories with >=8 adapters each",
            "rule_b": "or >=4 task categories with >=5 adapters each",
        },
        "task_targets": task_targets,
        "architecture_coverage_goals": fam_targets,
        "families": families,
        "deprioritize_category": "chat_instruct",
        "notes": (
            "New additions are chosen for task-balance value first, with architecture parity as a secondary constraint."
        ),
    }

    md_lines = [
        "# Task-Balanced Extension Design",
        "",
        f"- Snapshot date (UTC): {design_json['snapshot_date']}",
        f"- Families preserved: {', '.join(families)}",
        "- Prioritize: code, math_reasoning, domain_specialist, classification",
        "- Deprioritize: chat_instruct (unless needed for architecture parity)",
        "",
        "## Task Targets",
        "",
        "| Category | Current | Target | Needed |",
        "|----------|---------|--------|--------|",
    ]
    for t in TARGET_TASKS:
        r = task_targets[t]
        md_lines.append(f"| {t} | {r['current_count']} | {r['target_count']} | {r['needed']} |")
    md_lines += [
        "",
        "## Architecture Coverage Goals",
        "",
        "| Family | Current | Target Min | Needed |",
        "|--------|---------|------------|--------|",
    ]
    for fam in families:
        r = fam_targets[fam]
        md_lines.append(f"| {fam} | {r['current_count']} | {r['target_min_count']} | {r['needed_for_parity']} |")
    md_lines.append("")

    _write_json(output_dir / "task_balanced_extension_design.json", design_json)
    _write_text(output_dir / "task_balanced_extension_design.md", "\n".join(md_lines))
    return design_json


def _build_queries_for_base(short_name: str, deficits: dict[str, int]) -> list[str]:
    queries = [short_name]
    for task, needed in deficits.items():
        if needed <= 0:
            continue
        for kw in TASK_SEARCH_TERMS.get(task, []):
            queries.append(f"{short_name} {kw}")
    # Dedup while preserving order
    deduped: list[str] = []
    seen = set()
    for q in queries:
        if q not in seen:
            deduped.append(q)
            seen.add(q)
    return deduped


def _has_required_adapter_files(model: Any) -> bool:
    siblings = getattr(model, "siblings", None) or []
    filenames = {s.rfilename for s in siblings}
    has_config = any(f.startswith("adapter_config") for f in filenames)
    has_weights = any(f.startswith("adapter_model") or f == "pytorch_model.bin" for f in filenames)
    return has_config and has_weights


def _estimate_weight_size_mb(model: Any) -> float | None:
    try:
        total = 0
        found = False
        for s in getattr(model, "siblings", None) or []:
            if s.rfilename.startswith("adapter_model") or s.rfilename == "pytorch_model.bin":
                sz = getattr(s, "size", None)
                if sz is not None and sz > 0:
                    total += int(sz)
                    found = True
        if not found:
            return None
        return total / (1024.0 * 1024.0)
    except Exception:
        return None


def _candidate_priority(
    task_category: str,
    architecture_family: str,
    deficits: dict[str, int],
    family_deficits: dict[str, int],
    downloads: int,
    confidence: str,
) -> float:
    score = 0.0
    score += 3.0 * float(max(deficits.get(task_category, 0), 0))
    score += 1.5 * float(max(family_deficits.get(architecture_family, 0), 0))
    if task_category == "chat_instruct":
        score -= 4.0
    if task_category == "general_unknown":
        score -= 2.0
    score += {3: 2.0, 2: 1.0, 1: 0.0}.get(CONFIDENCE_ORDER.get(confidence, 1), 0.0)
    score += min(math.log10(max(downloads, 1)), 3.0) * 0.25
    return score


def discover_targeted_candidates(
    families: list[str],
    deficits: dict[str, int],
    family_deficits: dict[str, int],
    seen_repo_ids: set[str],
    min_downloads: int,
    search_limit: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    api = HfApi()
    by_repo: dict[str, dict[str, Any]] = {}
    family_model_map = {fam: census.ARCHITECTURE_FAMILIES.get(fam, []) for fam in families}

    for family in families:
        for base_model in family_model_map.get(family, []):
            short = base_model.split("/")[-1]
            queries = _build_queries_for_base(short, deficits)
            for query in queries:
                try:
                    models = api.list_models(
                        filter="peft",
                        search=query,
                        sort="downloads",
                        direction=-1,
                        limit=search_limit,
                        full=True,
                    )
                except Exception as e:
                    print(f"  discover warning [{family}:{query}]: {e}")
                    continue

                for model in models:
                    repo_id = model.id
                    if repo_id in seen_repo_ids:
                        continue
                    downloads = int(getattr(model, "downloads", 0) or 0)
                    if downloads < min_downloads:
                        continue
                    if not _has_required_adapter_files(model):
                        continue
                    est_size = _estimate_weight_size_mb(model)
                    if est_size is not None and est_size > 500.0:
                        continue

                    tags = sorted(set(getattr(model, "tags", []) or []))
                    inferred_task = census._infer_task(set(tags), repo_id)
                    task_category = census.infer_task_category(inferred_task)
                    confidence, source = _infer_task_label_confidence(tags, inferred_task, task_category, repo_id)

                    # Keep candidates that help task deficits or family parity.
                    helps_task = deficits.get(task_category, 0) > 0
                    helps_family = family_deficits.get(family, 0) > 0
                    if not helps_task and not helps_family:
                        continue

                    prio = _candidate_priority(
                        task_category=task_category,
                        architecture_family=family,
                        deficits=deficits,
                        family_deficits=family_deficits,
                        downloads=downloads,
                        confidence=confidence,
                    )
                    candidate = {
                        "repo_id": repo_id,
                        "base_model": base_model,
                        "architecture_family": family,
                        "downloads": downloads,
                        "tags": tags,
                        "inferred_task": inferred_task,
                        "task_category": task_category,
                        "task_label_confidence": confidence,
                        "task_label_source": source,
                        "priority": round(prio, 4),
                    }
                    prev = by_repo.get(repo_id)
                    if prev is None or candidate["priority"] > prev["priority"]:
                        by_repo[repo_id] = candidate

    ranked = sorted(by_repo.values(), key=lambda r: (-r["priority"], -r["downloads"], r["repo_id"]))
    return ranked[:max_candidates]


def _record_to_dict(record: census.CensusAdapterRecord) -> dict[str, Any]:
    base = asdict(record)
    if hasattr(record, "task_label_confidence"):
        base["task_label_confidence"] = getattr(record, "task_label_confidence")
    if hasattr(record, "task_label_source"):
        base["task_label_source"] = getattr(record, "task_label_source")
    return base


def _progress_targets_met(
    task_counts: Counter[str],
    family_counts: Counter[str],
    target_counts: dict[str, int],
    family_min_target: int,
    families: list[str],
) -> bool:
    tasks_met = [
        task_counts.get("code", 0) >= target_counts["code"],
        task_counts.get("math_reasoning", 0) >= target_counts["math_reasoning"],
        task_counts.get("domain_specialist", 0) >= target_counts["domain_specialist"],
        task_counts.get("classification", 0) >= target_counts["classification"],
    ]
    # Primary stop rule: three task categories at >=8 OR four categories at >=5
    n_ge_8 = sum(
        1 for t in TARGET_TASKS if task_counts.get(t, 0) >= 8
    )
    n_ge_5 = sum(
        1 for t in TARGET_TASKS if task_counts.get(t, 0) >= 5
    )
    fam_ok = all(family_counts.get(f, 0) >= min(family_min_target, 5) for f in families)
    return (n_ge_8 >= 3 or n_ge_5 >= 4) and fam_ok and all(tasks_met[:3])


def stage_c_targeted_extension_pass(
    output_dir: Path,
    cache_dir: Path,
    baseline: dict[str, Any],
    design: dict[str, Any],
    families: list[str],
    max_new_adapters: int,
    min_downloads: int,
    search_limit: int,
    skip_discovery: bool,
) -> dict[str, Any]:
    baseline_records = baseline["records"]
    baseline_task_counts = Counter(fp.task_category for fp in baseline["fingerprints"])
    baseline_family_counts = Counter(fp.architecture_family for fp in baseline["fingerprints"])

    target_counts = {t: int(design["task_targets"][t]["target_count"]) for t in TARGET_TASKS}
    family_min_target = 5

    deficits = {t: max(target_counts[t] - baseline_task_counts.get(t, 0), 0) for t in TARGET_TASKS}
    family_deficits = {
        fam: max(family_min_target - baseline_family_counts.get(fam, 0), 0)
        for fam in families
    }

    seen_repo_ids = {r.get("repo_id") for r in baseline_records if r.get("repo_id")}
    existing_extension = _read_json(output_dir / "adapter_records_extension.json", [])
    for r in existing_extension:
        if r.get("repo_id"):
            seen_repo_ids.add(r["repo_id"])

    candidates: list[dict[str, Any]] = []
    if skip_discovery and existing_extension:
        print("\n[Stage C] Discovery skipped; reusing existing extension records.")
    elif not skip_discovery:
        print("\n[Stage C] Discovering targeted candidates...")
        candidates = discover_targeted_candidates(
            families=families,
            deficits=deficits,
            family_deficits=family_deficits,
            seen_repo_ids=seen_repo_ids,
            min_downloads=min_downloads,
            search_limit=search_limit,
            max_candidates=max_new_adapters * 4,
        )
    elif skip_discovery:
        print("\n[Stage C] Discovery skipped by flag; using no new candidates.")

    if candidates:
        print(f"  discovered targeted candidates: {len(candidates)}")
    else:
        print("  no targeted candidates discovered in this pass")

    # Select with quota-aware greedy pass.
    selected: list[dict[str, Any]] = []
    if not (skip_discovery and existing_extension):
        running_task = Counter(baseline_task_counts)
        running_family = Counter(baseline_family_counts)

        for cand in candidates:
            if len(selected) >= max_new_adapters:
                break

            tcat = cand["task_category"]
            fam = cand["architecture_family"]
            helps_task = running_task.get(tcat, 0) < target_counts.get(tcat, 0)
            helps_family = running_family.get(fam, 0) < family_min_target

            if not (helps_task or helps_family):
                continue

            selected.append(cand)
            running_task[tcat] += 1
            running_family[fam] += 1

            if _progress_targets_met(running_task, running_family, target_counts, family_min_target, families):
                break
    print(f"  selected for audit: {len(selected)}")

    # Load audit cache (shared with baseline census run)
    audit_cache_path = output_dir / "audit_cache.json"
    audit_cache = _read_json(audit_cache_path, baseline["audit_cache"] or {})

    extension_records: list[census.CensusAdapterRecord] = []
    if skip_discovery and existing_extension:
        for row in existing_extension:
            rec = census.CensusAdapterRecord(
                repo_id=row.get("repo_id", ""),
                base_model=row.get("base_model", ""),
                architecture_family=row.get("architecture_family", "unknown"),
                inferred_task=row.get("inferred_task", "general"),
                task_category=row.get("task_category", "general_unknown"),
                rank=row.get("rank"),
                alpha=row.get("alpha"),
                lora_dropout=row.get("lora_dropout"),
                target_modules=row.get("target_modules"),
                peft_type=row.get("peft_type"),
                downloads=row.get("downloads"),
                tags=row.get("tags"),
                n_layers=row.get("n_layers"),
                total_params=row.get("total_params"),
                download_size_mb=row.get("download_size_mb"),
                audit_time_s=row.get("audit_time_s"),
                status=row.get("status", "audited"),
                issues=row.get("issues", []) or [],
                exclusion_reason=row.get("exclusion_reason"),
            )
            setattr(rec, "task_label_confidence", row.get("task_label_confidence", "medium"))
            setattr(rec, "task_label_source", row.get("task_label_source", "metadata"))
            extension_records.append(rec)
    else:
        for idx, cand in enumerate(selected, start=1):
            rec = census.CensusAdapterRecord(
                repo_id=cand["repo_id"],
                base_model=cand["base_model"],
                architecture_family=cand["architecture_family"],
                inferred_task=cand["inferred_task"],
                task_category=cand["task_category"],
                downloads=cand["downloads"],
                tags=cand["tags"],
            )
            setattr(rec, "task_label_confidence", cand["task_label_confidence"])
            setattr(rec, "task_label_source", cand["task_label_source"])

            print(f"  [{idx}/{len(selected)}] {rec.repo_id} ({rec.task_category}, {rec.architecture_family})")

            if rec.repo_id in audit_cache:
                rec.status = "audited"
                ad = audit_cache[rec.repo_id]
                rec.n_layers = ad.get("n_layers")
                rec.total_params = ad.get("total_lora_params")
                layer_rows = ad.get("layer_data", {}).get("layer_rows", [])
                if layer_rows:
                    rec.rank = layer_rows[0].get("r")
                    rec.alpha = layer_rows[0].get("alpha")
                extension_records.append(rec)
                print("    using cached audit")
                continue

            try:
                adapter_path, size_mb = census.download_adapter(rec.repo_id, cache_dir)
                rec.download_size_mb = size_mb
                rec.status = "downloaded"
            except Exception as e:
                rec.status = "failed"
                rec.issues.append(f"Download error: {e}")
                extension_records.append(rec)
                print(f"    download failed: {e}")
                continue

            result_dict, rec = census.audit_adapter(rec.repo_id, adapter_path, rec)
            if result_dict is not None:
                audit_cache[rec.repo_id] = result_dict
                print(f"    audited: layers={rec.n_layers}, rank={rec.rank}, task={rec.task_category}")
            else:
                print(f"    {rec.status}: {rec.exclusion_reason or '; '.join(rec.issues)}")
            extension_records.append(rec)

    _write_json(audit_cache_path, audit_cache)

    ext_dict_rows = [_record_to_dict(r) for r in extension_records]
    _write_json(output_dir / "adapter_records_extension.json", ext_dict_rows)

    excluded_ext = [
        {
            "repo_id": r.repo_id,
            "architecture_family": r.architecture_family,
            "task_category": r.task_category,
            "status": r.status,
            "exclusion_reason": r.exclusion_reason,
            "issues": r.issues,
            "task_label_confidence": getattr(r, "task_label_confidence", "low"),
            "task_label_source": getattr(r, "task_label_source", "metadata"),
        }
        for r in extension_records
        if r.status in ("excluded", "failed")
    ]
    _write_json(output_dir / "excluded_adapters_extension.json", excluded_ext)

    # Update manifest with extension metadata (without replacing baseline keys)
    manifest = dict(baseline["manifest"])
    manifest["task_balanced_extension"] = {
        "date": _now_iso(),
        "families": families,
        "target_counts": target_counts,
        "baseline_task_counts": dict(baseline_task_counts),
        "baseline_family_counts": dict(baseline_family_counts),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "audited_count": sum(1 for r in extension_records if r.status == "audited"),
        "excluded_count": sum(1 for r in extension_records if r.status == "excluded"),
        "failed_count": sum(1 for r in extension_records if r.status == "failed"),
    }
    _write_json(output_dir / "manifest.json", manifest)

    return {
        "extension_records": extension_records,
        "audit_cache": audit_cache,
        "manifest": manifest,
        "selected": selected,
        "candidates": candidates,
    }


def _build_augmented_records(
    baseline_records: list[dict[str, Any]],
    extension_records: list[census.CensusAdapterRecord],
) -> list[dict[str, Any]]:
    by_repo: dict[str, dict[str, Any]] = {}
    for r in baseline_records:
        rid = r.get("repo_id")
        if rid:
            by_repo[rid] = dict(r)
    for r in extension_records:
        rid = r.repo_id
        if rid:
            by_repo[rid] = _record_to_dict(r)
    return list(by_repo.values())


def _extract_augmented_fingerprints(
    augmented_records: list[dict[str, Any]],
    audit_cache: dict[str, dict[str, Any]],
) -> list[census.SpectralFingerprint]:
    fps: list[census.SpectralFingerprint] = []
    for rec in augmented_records:
        rid = rec.get("repo_id")
        if not rid or rid not in audit_cache:
            continue
        # Only include records that are likely valid audited adapters.
        if rec.get("status") not in (None, "audited", "downloaded"):
            continue
        row_record = census.CensusAdapterRecord(
            repo_id=rid,
            base_model=rec.get("base_model", ""),
            architecture_family=rec.get("architecture_family", "unknown"),
            inferred_task=rec.get("inferred_task", "general"),
            task_category=rec.get("task_category", "general_unknown"),
            rank=rec.get("rank"),
            alpha=rec.get("alpha"),
            lora_dropout=rec.get("lora_dropout"),
            target_modules=rec.get("target_modules"),
            peft_type=rec.get("peft_type"),
            downloads=rec.get("downloads"),
        )
        fp = census.extract_fingerprint(row_record, audit_cache[rid])
        if fp is not None:
            fps.append(fp)
    # dedupe by repo
    dedup = {}
    for fp in fps:
        dedup[fp.repo_id] = fp
    return list(dedup.values())


def _generate_task_balanced_decomposition_md(payload: dict[str, Any]) -> str:
    p2 = payload["confound_assessment"]
    p2b = payload["analytic_controls"]
    p3 = payload["decomposition"]
    summary = p3.get("summary", {})
    lines = [
        "# Task-Balanced Decomposition",
        "",
        "## Confound-First Readout",
        "",
        f"- Max confound R-squared: {p2.get('overall', {}).get('max_r_squared', 'N/A')}",
        f"- Residualization recommended: {p2.get('overall', {}).get('residualization_recommended', 'N/A')}",
        "",
        "## Architecture vs Task Effect Sizes",
        "",
        f"- Mean architecture eta-squared: {summary.get('mean_architecture_eta_sq', 'N/A')}",
        f"- Mean task eta-squared: {summary.get('mean_task_eta_sq', 'N/A')}",
        f"- Dominant factor: {summary.get('dominant_factor', 'N/A')}",
        "",
        "| Metric | Arch eta-sq | Task eta-sq |",
        "|--------|-------------|-------------|",
    ]
    arch = p3.get("architecture_effect", {})
    task = p3.get("task_effect", {})
    for metric in ["stable_rank_mean", "utilization_mean", "energy_rank_90_p50", "entropy_erank_mean", "edge_gap_mean"]:
        a = arch.get(metric, {}).get("eta_sq", "N/A")
        t = task.get(metric, {}).get("eta_sq", "N/A")
        lines.append(f"| {metric} | {a} | {t} |")

    lines += [
        "",
        "## Analytic Controls",
        "",
        f"- Rank-matched subset n: {p2b.get('rank_matched_subset', {}).get('n_matched', 'N/A')}",
        f"- Task-confidence subset n: {p2b.get('task_label_confidence_subset', {}).get('n_confident', 'N/A')}",
        (
            "- Residualized architecture eta-sq mean: "
            f"{p2b.get('variant_robustness', {}).get('mean_residualized_arch_eta_sq', 'N/A')}"
        ),
        "",
    ]
    return "\n".join(lines)


def _generate_task_balanced_clustering_md(p4: dict[str, Any], p5: dict[str, Any]) -> str:
    lines = [
        "# Task-Balanced Clustering and Module-Type Readout",
        "",
    ]
    if not p4.get("computable", False):
        lines += [
            "Clustering was not computable on this cohort.",
            "",
        ]
    else:
        lines += [
            "## kNN Purity (k=5)",
            "",
            f"- Architecture purity mean: {p4.get('architecture_knn_purity', {}).get('mean', 'N/A')}",
            f"- Task purity mean: {p4.get('task_knn_purity', {}).get('mean', 'N/A')}",
            f"- Architecture random baseline: {p4.get('baselines', {}).get('architecture_random_purity', 'N/A')}",
            f"- Task random baseline: {p4.get('baselines', {}).get('task_random_purity', 'N/A')}",
            "",
        ]

    lines += [
        "## Module-Type Summary",
        "",
        f"- {p5.get('sign_test_interpretation', 'N/A')}",
        f"- Replicates encoder-era pattern: {p5.get('replicates_encoder_pattern', 'N/A')}",
        "",
    ]
    return "\n".join(lines)


def _task_conditioned_subtype_summary(
    fingerprints: list[census.SpectralFingerprint],
    audit_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Lightweight subtype-by-task readout for Stage D.

    Produces per-task mean stable rank by common decoder subtype labels where available.
    """
    subtype_suffix_map = {
        "q_proj": "q",
        "k_proj": "k",
        "v_proj": "v",
        "o_proj": "o",
        "gate_proj": "gate",
        "up_proj": "up",
        "down_proj": "down",
    }
    task_subtype_vals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    task_by_repo = {fp.repo_id: fp.task_category for fp in fingerprints}
    for repo_id, task_category in task_by_repo.items():
        ad = audit_cache.get(repo_id, {})
        layer_rows = ad.get("layer_data", {}).get("layer_rows", [])
        for layer in layer_rows:
            sr = layer.get("stable_rank")
            if sr is None:
                continue
            layer_name = layer.get("name", "") or layer.get("layer_name", "")
            subtype = None
            for suffix, label in subtype_suffix_map.items():
                if suffix in layer_name:
                    subtype = label
                    break
            if subtype is None:
                continue
            task_subtype_vals[task_category][subtype].append(float(sr))

    summary: dict[str, Any] = {}
    for task_category, subtype_map in sorted(task_subtype_vals.items()):
        task_out: dict[str, Any] = {}
        for subtype, values in sorted(subtype_map.items()):
            if not values:
                continue
            task_out[subtype] = {
                "n_layers": len(values),
                "stable_rank_mean": round(census._mean(values), 4),
                "stable_rank_std": round(census._std(values), 4),
            }
        summary[task_category] = task_out
    return summary


def stage_d_augmented_analysis(
    output_dir: Path,
    baseline: dict[str, Any],
    stage_c: dict[str, Any],
) -> dict[str, Any]:
    augmented_records = _build_augmented_records(baseline["records"], stage_c["extension_records"])
    audit_cache = stage_c["audit_cache"]
    fingerprints = _extract_augmented_fingerprints(augmented_records, audit_cache)

    print(f"\n[Stage D] Augmented fingerprints: {len(fingerprints)}")
    if len(fingerprints) < 5:
        raise RuntimeError("Insufficient augmented fingerprints (<5) to run decomposition analysis.")

    p1 = census.phase1_distributional(fingerprints)
    p2 = census.phase2_confound_assessment(fingerprints)
    p2b = census.phase2b_analytic_controls(fingerprints)
    p3 = census.phase3_architecture_task_decomposition(fingerprints)
    p4 = census.phase4_clustering(fingerprints, k=5)
    p5 = census.phase5_module_type_asymmetry(fingerprints, audit_cache)
    task_subtype = _task_conditioned_subtype_summary(fingerprints, audit_cache)

    decomp_payload = {
        "phase": "task_balanced_decomposition",
        "date": _now_iso(),
        "n_fingerprints": len(fingerprints),
        "distributional": p1,
        "confound_assessment": p2,
        "analytic_controls": p2b,
        "decomposition": p3,
        "minimum_per_cell_rule": {
            "recommended_exploratory_min_per_cell": 4,
            "recommended_serious_min_per_cell": 6,
            "note": "Do not over-interpret architecture×task interactions when cells are sparse.",
        },
    }
    cluster_payload = {
        "phase": "task_balanced_clustering",
        "date": _now_iso(),
        "n_fingerprints": len(fingerprints),
        "clustering": p4,
        "module_type": p5,
        "task_conditioned_subtype_summary": task_subtype,
    }

    _write_json(output_dir / "task_balanced_decomposition.json", decomp_payload)
    _write_text(output_dir / "task_balanced_decomposition.md", _generate_task_balanced_decomposition_md(decomp_payload))
    _write_json(output_dir / "task_balanced_clustering.json", cluster_payload)
    _write_text(output_dir / "task_balanced_clustering.md", _generate_task_balanced_clustering_md(p4, p5))

    # Extension-specific confound/module outputs (retain baseline files untouched)
    _write_json(output_dir / "confound_assessment_task_balanced.json", p2)
    _write_json(output_dir / "module_type_replication_task_balanced.json", p5)

    # Extension-specific fingerprint table (baseline remains available in fingerprint_table.json)
    fp_rows = [
        {
            "repo_id": fp.repo_id,
            "architecture_family": fp.architecture_family,
            "task_category": fp.task_category,
            "nominal_rank": fp.nominal_rank,
            **{name: getattr(fp, name) for name in census.SpectralFingerprint.dimension_names()},
        }
        for fp in sorted(fingerprints, key=lambda x: (x.architecture_family, x.task_category, x.repo_id))
    ]
    _write_json(output_dir / "fingerprint_table_task_balanced.json", fp_rows)

    return {
        "augmented_records": augmented_records,
        "fingerprints": fingerprints,
        "p1": p1,
        "p2": p2,
        "p2b": p2b,
        "p3": p3,
        "p4": p4,
        "p5": p5,
    }


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(b - a, 4)


def _high_conf_subset_effects(
    fingerprints: list[census.SpectralFingerprint],
    confidence_by_repo: dict[str, str],
) -> dict[str, Any]:
    subset = [fp for fp in fingerprints if confidence_by_repo.get(fp.repo_id, "medium") in ("high", "medium")]
    if len(subset) < 5:
        return {"n": len(subset), "note": "insufficient high/medium-confidence adapters"}
    p3 = census.phase3_architecture_task_decomposition(subset)
    p4 = census.phase4_clustering(subset, k=5 if len(subset) >= 6 else 3)
    return {
        "n": len(subset),
        "decomposition_summary": p3.get("summary", {}),
        "clustering": {
            "architecture_knn_purity_mean": p4.get("architecture_knn_purity", {}).get("mean"),
            "task_knn_purity_mean": p4.get("task_knn_purity", {}).get("mean"),
        },
    }


def stage_e_comparison(
    output_dir: Path,
    baseline: dict[str, Any],
    stage_c: dict[str, Any],
    stage_d: dict[str, Any],
) -> dict[str, Any]:
    b_p3 = baseline["p3"]
    b_p4 = baseline["p4"]
    b_p5 = baseline["p5"]

    a_p3 = stage_d["p3"]
    a_p4 = stage_d["p4"]
    a_p5 = stage_d["p5"]

    b_summary = b_p3.get("summary", {})
    a_summary = a_p3.get("summary", {})

    metric_deltas: dict[str, Any] = {}
    for metric in ["stable_rank_mean", "utilization_mean", "energy_rank_90_p50", "entropy_erank_mean", "edge_gap_mean"]:
        b_arch = b_p3.get("architecture_effect", {}).get(metric, {}).get("eta_sq")
        b_task = b_p3.get("task_effect", {}).get(metric, {}).get("eta_sq")
        a_arch = a_p3.get("architecture_effect", {}).get(metric, {}).get("eta_sq")
        a_task = a_p3.get("task_effect", {}).get(metric, {}).get("eta_sq")
        metric_deltas[metric] = {
            "baseline_arch_eta_sq": b_arch,
            "baseline_task_eta_sq": b_task,
            "task_balanced_arch_eta_sq": a_arch,
            "task_balanced_task_eta_sq": a_task,
            "arch_eta_sq_delta": _delta(b_arch, a_arch),
            "task_eta_sq_delta": _delta(b_task, a_task),
        }

    comparison = {
        "phase": "task_balance_comparison",
        "date": _now_iso(),
        "cohort_counts": {
            "baseline_n_fingerprints": len(baseline["fingerprints"]),
            "augmented_n_fingerprints": len(stage_d["fingerprints"]),
            "baseline_task_counts": dict(Counter(fp.task_category for fp in baseline["fingerprints"])),
            "augmented_task_counts": dict(Counter(fp.task_category for fp in stage_d["fingerprints"])),
            "baseline_architecture_counts": dict(Counter(fp.architecture_family for fp in baseline["fingerprints"])),
            "augmented_architecture_counts": dict(Counter(fp.architecture_family for fp in stage_d["fingerprints"])),
        },
        "effect_size_shift": {
            "baseline_mean_architecture_eta_sq": b_summary.get("mean_architecture_eta_sq"),
            "baseline_mean_task_eta_sq": b_summary.get("mean_task_eta_sq"),
            "augmented_mean_architecture_eta_sq": a_summary.get("mean_architecture_eta_sq"),
            "augmented_mean_task_eta_sq": a_summary.get("mean_task_eta_sq"),
            "architecture_eta_sq_delta": _delta(
                b_summary.get("mean_architecture_eta_sq"), a_summary.get("mean_architecture_eta_sq")
            ),
            "task_eta_sq_delta": _delta(b_summary.get("mean_task_eta_sq"), a_summary.get("mean_task_eta_sq")),
            "baseline_dominant_factor": b_summary.get("dominant_factor"),
            "augmented_dominant_factor": a_summary.get("dominant_factor"),
        },
        "metric_level_effect_deltas": metric_deltas,
        "purity_shift": {
            "baseline_architecture_knn_purity": b_p4.get("architecture_knn_purity", {}).get("mean"),
            "baseline_task_knn_purity": b_p4.get("task_knn_purity", {}).get("mean"),
            "augmented_architecture_knn_purity": a_p4.get("architecture_knn_purity", {}).get("mean"),
            "augmented_task_knn_purity": a_p4.get("task_knn_purity", {}).get("mean"),
            "architecture_knn_delta": _delta(
                b_p4.get("architecture_knn_purity", {}).get("mean"),
                a_p4.get("architecture_knn_purity", {}).get("mean"),
            ),
            "task_knn_delta": _delta(
                b_p4.get("task_knn_purity", {}).get("mean"), a_p4.get("task_knn_purity", {}).get("mean")
            ),
        },
        "module_type_shift": {
            "baseline_sign_test": b_p5.get("sign_test_interpretation"),
            "augmented_sign_test": a_p5.get("sign_test_interpretation"),
            "baseline_replicates_encoder_pattern": b_p5.get("replicates_encoder_pattern"),
            "augmented_replicates_encoder_pattern": a_p5.get("replicates_encoder_pattern"),
        },
    }

    # Confidence-subset behavior on augmented cohort
    confidence_map: dict[str, str] = {}
    for rec in baseline["records"]:
        rid = rec.get("repo_id")
        if not rid:
            continue
        cat = rec.get("task_category", "general_unknown")
        confidence_map[rid] = "low" if cat == "general_unknown" else "medium"
    for rec in stage_c["extension_records"]:
        confidence_map[rec.repo_id] = getattr(rec, "task_label_confidence", "medium")
    comparison["high_confidence_subset_behavior"] = _high_conf_subset_effects(stage_d["fingerprints"], confidence_map)

    # Outcome typing for Stage F handoff
    task_delta = comparison["effect_size_shift"]["task_eta_sq_delta"] or 0.0
    arch_delta = comparison["effect_size_shift"]["architecture_eta_sq_delta"] or 0.0
    task_purity_delta = comparison["purity_shift"]["task_knn_delta"] or 0.0
    arch_mean = comparison["effect_size_shift"]["augmented_mean_architecture_eta_sq"] or 0.0
    task_mean = comparison["effect_size_shift"]["augmented_mean_task_eta_sq"] or 0.0

    if task_delta >= 0.05 and task_purity_delta >= 0.05:
        decision = "strengthens_task_story"
    elif arch_mean >= task_mean and task_delta < 0.03 and arch_delta >= -0.03:
        decision = "architecture_remains_primary"
    else:
        decision = "mixed_but_bounded"
    comparison["decision_candidate"] = decision

    md_lines = [
        "# Task-Balance Comparison (Pre vs Post)",
        "",
        "## Effect Size Shift",
        "",
        f"- Baseline mean architecture eta-sq: {comparison['effect_size_shift']['baseline_mean_architecture_eta_sq']}",
        f"- Baseline mean task eta-sq: {comparison['effect_size_shift']['baseline_mean_task_eta_sq']}",
        f"- Task-balanced mean architecture eta-sq: {comparison['effect_size_shift']['augmented_mean_architecture_eta_sq']}",
        f"- Task-balanced mean task eta-sq: {comparison['effect_size_shift']['augmented_mean_task_eta_sq']}",
        f"- Architecture eta-sq delta: {comparison['effect_size_shift']['architecture_eta_sq_delta']}",
        f"- Task eta-sq delta: {comparison['effect_size_shift']['task_eta_sq_delta']}",
        "",
        "## Purity Shift",
        "",
        f"- Architecture kNN purity delta: {comparison['purity_shift']['architecture_knn_delta']}",
        f"- Task kNN purity delta: {comparison['purity_shift']['task_knn_delta']}",
        "",
        "## Candidate Decision",
        "",
        f"- {comparison['decision_candidate']}",
        "",
    ]

    _write_json(output_dir / "task_balance_comparison.json", comparison)
    _write_text(output_dir / "task_balance_comparison.md", "\n".join(md_lines))
    return comparison


def stage_f_extension_memo(
    output_dir: Path,
    baseline: dict[str, Any],
    stage_c: dict[str, Any],
    stage_d: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    decision = comparison.get("decision_candidate", "mixed_but_bounded")
    effect = comparison.get("effect_size_shift", {})
    purity = comparison.get("purity_shift", {})

    memo = [
        "# Task-Balanced Extension Memo",
        "",
        "## 1) What Changed",
        "",
        f"- Baseline fingerprints: {len(baseline['fingerprints'])}",
        f"- Augmented fingerprints: {len(stage_d['fingerprints'])}",
        f"- New candidates discovered: {len(stage_c.get('candidates', []))}",
        f"- New adapters selected for audit: {len(stage_c.get('selected', []))}",
        "",
        "## 2) What Changed in Results",
        "",
        f"- Mean architecture eta-sq: {effect.get('baseline_mean_architecture_eta_sq')} -> {effect.get('augmented_mean_architecture_eta_sq')}",
        f"- Mean task eta-sq: {effect.get('baseline_mean_task_eta_sq')} -> {effect.get('augmented_mean_task_eta_sq')}",
        f"- Architecture kNN purity: {purity.get('baseline_architecture_knn_purity')} -> {purity.get('augmented_architecture_knn_purity')}",
        f"- Task kNN purity: {purity.get('baseline_task_knn_purity')} -> {purity.get('augmented_task_knn_purity')}",
        f"- Module-type sign test (augmented): {stage_d['p5'].get('sign_test_interpretation', 'N/A')}",
        "",
        "## 3) What Remains Bounded",
        "",
        "- Found-artifact observational setting only (no causal claim).",
        "- Task labels remain metadata-driven and partially noisy.",
        "- Architecture×task interactions remain exploratory unless per-cell counts are adequate.",
        "- No product-policy implication from this extension pass.",
        "",
        "## 4) Decoder Question Impact",
        "",
        "This extension sharpens the architecture-vs-task interpretation for the public ecosystem slice and "
        "informs the later controlled decoder GPU study as either causal disambiguation or confirmation.",
        "",
        "## 5) Decision",
        "",
        f"- **{decision}**",
        "",
    ]

    memo_path = output_dir / "task_balanced_extension_memo.md"
    _write_text(memo_path, "\n".join(memo))

    strategy_lines = [
        "# Public Ecosystem Census: Task-Balanced Extension Summary",
        "",
        f"- Decision: `{decision}`",
        f"- Baseline n: {len(baseline['fingerprints'])}",
        f"- Augmented n: {len(stage_d['fingerprints'])}",
        f"- Mean architecture eta-sq (augmented): {effect.get('augmented_mean_architecture_eta_sq')}",
        f"- Mean task eta-sq (augmented): {effect.get('augmented_mean_task_eta_sq')}",
        f"- Task kNN purity delta: {purity.get('task_knn_delta')}",
        "",
        "Guardrails: observational census only; confound-sensitive; no causal architecture/task claim.",
        "",
    ]
    strategy_path = _REPO_ROOT / "docs" / "strategy" / "public_ecosystem_census_task_balanced_summary.md"
    _write_text(strategy_path, "\n".join(strategy_lines))

    decision_json = {
        "date": _now_iso(),
        "decision": decision,
        "n_baseline": len(baseline["fingerprints"]),
        "n_augmented": len(stage_d["fingerprints"]),
        "architecture_eta_sq_augmented": effect.get("augmented_mean_architecture_eta_sq"),
        "task_eta_sq_augmented": effect.get("augmented_mean_task_eta_sq"),
        "task_knn_delta": purity.get("task_knn_delta"),
        "note": "Bounded observational extension result; no causal or policy claim.",
    }
    _write_json(output_dir / "task_balanced_extension_decision_summary.json", decision_json)
    return decision_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task-balanced extension pass for public ecosystem census")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("field_trials/public_ecosystem_census"),
        help="Census output directory (default: field_trials/public_ecosystem_census)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/census_adapters"),
        help="Adapter cache directory (default: .cache/census_adapters)",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["llama", "mistral", "qwen"],
        help="Architecture families to preserve",
    )
    parser.add_argument("--target-code", type=int, default=DEFAULT_TARGET_COUNTS["code"])
    parser.add_argument("--target-math-reasoning", type=int, default=DEFAULT_TARGET_COUNTS["math_reasoning"])
    parser.add_argument("--target-domain-specialist", type=int, default=DEFAULT_TARGET_COUNTS["domain_specialist"])
    parser.add_argument("--target-classification", type=int, default=DEFAULT_TARGET_COUNTS["classification"])
    parser.add_argument("--max-new-adapters", type=int, default=40, help="Upper bound on new adapters to audit")
    parser.add_argument("--min-downloads", type=int, default=10)
    parser.add_argument("--search-limit", type=int, default=120, help="HF results per query")
    parser.add_argument(
        "--skip-discovery",
        action="store_true",
        help="Skip targeted discovery/download and only regenerate baseline+analysis artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    cache_dir = args.cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    target_counts = {
        "code": args.target_code,
        "math_reasoning": args.target_math_reasoning,
        "domain_specialist": args.target_domain_specialist,
        "classification": args.target_classification,
    }
    family_min_target = 5

    print("=" * 72)
    print("PUBLIC ECOSYSTEM CENSUS -- TASK-BALANCED EXTENSION PASS")
    print("=" * 72)
    print(f"output_dir: {output_dir}")
    print(f"cache_dir: {cache_dir}")
    print(f"families: {args.families}")
    print(f"target_counts: {target_counts}")
    print(f"max_new_adapters: {args.max_new_adapters}")
    if args.skip_discovery:
        print("mode: SKIP discovery (analysis-only)")
    print("=" * 72)

    baseline = load_baseline_state(output_dir)
    if not baseline["fingerprints"]:
        raise RuntimeError(
            "Baseline census artifacts not found or empty. Run scripts/ecosystem_census.py first."
        )

    print("\n[Stage A] Baseline freeze...")
    stage_a_baseline_freeze(output_dir, baseline)

    print("\n[Stage B] Extension design...")
    design = stage_b_extension_design(
        output_dir=output_dir,
        baseline=baseline,
        families=args.families,
        target_counts=target_counts,
        family_min_target=family_min_target,
    )

    print("\n[Stage C] Targeted extension pass...")
    stage_c = stage_c_targeted_extension_pass(
        output_dir=output_dir,
        cache_dir=cache_dir,
        baseline=baseline,
        design=design,
        families=args.families,
        max_new_adapters=args.max_new_adapters,
        min_downloads=args.min_downloads,
        search_limit=args.search_limit,
        skip_discovery=args.skip_discovery,
    )

    print("\n[Stage D] Augmented analysis re-run...")
    stage_d = stage_d_augmented_analysis(output_dir, baseline, stage_c)

    print("\n[Stage E] Task-balance comparison...")
    comparison = stage_e_comparison(output_dir, baseline, stage_c, stage_d)

    print("\n[Stage F] Extension memo...")
    decision = stage_f_extension_memo(output_dir, baseline, stage_c, stage_d, comparison)

    print("\nDone.")
    print(f"Decision: {decision['decision']}")


if __name__ == "__main__":
    main()
