"""Task-relationship advisory validation round.

Runs 5 inventories across different regimes, capturing advisory behavior.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROUND_ROOT = Path("results/task_advisory_round")

# Adapter locations
VERIFIED = Path("results/adjudication_study/verified_adapters")
AMBIG = Path("results/adjudication_study/ambiguous_relationship_regime/adapters")

# ---------------------------------------------------------------------------
# QA artifact generation
# ---------------------------------------------------------------------------


def _generate_qa(adapter_dir: Path, qa_path: Path, task: str, score: float, base_score: float) -> None:
    """Generate a QA artifact for a verified adapter."""
    if qa_path.exists():
        return

    qa_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "gradience.cli", "audit-adapter",
        "--peft-dir", str(adapter_dir),
        "--metric-name", "accuracy",
        "--higher-is-better",
        "--eval-dataset", f"{task}_dev",
        "--adapter-score", str(score),
        "--base-score", str(base_score),
        "--out", str(qa_path),
        "--no-udr",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  QA generation failed for {adapter_dir.name}: {result.stderr[:200]}")
    else:
        print(f"  Generated QA: {qa_path.name}")


def ensure_qa_artifacts():
    """Generate QA artifacts for all adapters we'll use."""
    qa_root = ROUND_ROOT / "qa_pool"
    qa_root.mkdir(parents=True, exist_ok=True)

    # Verified adapters (Study 1)
    verified_manifests = {
        "sst2_r16_s42": ("sst2", 0.822, 0.466),
        "sst2_r16_s123": ("sst2", 0.820, 0.596),
        "sst2_r8_s42": ("sst2", 0.828, 0.466),
        "qnli_r16_s42_v2": ("qnli", 0.768, 0.466),
        "qnli_r16_s123_v2": ("qnli", 0.772, 0.466),
        "qnli_r8_s42_v2": ("qnli", 0.750, 0.466),
    }

    for name, (task, score, base) in verified_manifests.items():
        _generate_qa(VERIFIED / name, qa_root / f"{name}_qa.json", task, score, base)

    # Ambiguous relationship adapters (Study 2)
    ambig_manifests = {
        "qnli_s42": ("qnli", 0.696, 0.470),
        "qnli_s7": ("qnli", 0.710, 0.542),
        "rte_s42": ("rte", 0.567, 0.516),
        "rte_s7": ("rte", 0.556, 0.473),
        "mnli_s42": ("mnli", 0.530, 0.352),
        "mnli_s7": ("mnli", 0.532, 0.302),
    }

    for name, (task, score, base) in ambig_manifests.items():
        _generate_qa(AMBIG / name, qa_root / f"{name}_qa.json", task, score, base)

    return qa_root


# ---------------------------------------------------------------------------
# Inventory definitions
# ---------------------------------------------------------------------------

INVENTORIES = [
    {
        "id": "same_task_sst2_control",
        "category": "A",
        "description": "Same-task control: 3 SST-2 adapters. Advisory should stay silent.",
        "adapters": [
            ("sst2_r16_s42", VERIFIED / "sst2_r16_s42"),
            ("sst2_r16_s123", VERIFIED / "sst2_r16_s123"),
            ("sst2_r8_s42", VERIFIED / "sst2_r8_s42"),
        ],
    },
    {
        "id": "nli_family_adjacent",
        "category": "B",
        "description": "Adjacent-task: 4 NLI-family adapters (2 QNLI + 2 RTE). Main target regime.",
        "adapters": [
            ("qnli_s42", AMBIG / "qnli_s42"),
            ("qnli_s7", AMBIG / "qnli_s7"),
            ("rte_s42", AMBIG / "rte_s42"),
            ("rte_s7", AMBIG / "rte_s7"),
        ],
    },
    {
        "id": "cross_task_sst2_qnli",
        "category": "C",
        "description": "Distant cross-task: 2 SST-2 + 2 QNLI. Advisory restating the obvious?",
        "adapters": [
            ("sst2_r16_s42", VERIFIED / "sst2_r16_s42"),
            ("sst2_r8_s42", VERIFIED / "sst2_r8_s42"),
            ("qnli_r16_s42_v2", VERIFIED / "qnli_r16_s42_v2"),
            ("qnli_r8_s42_v2", VERIFIED / "qnli_r8_s42_v2"),
        ],
    },
    {
        "id": "messy_mixed_quality",
        "category": "D",
        "description": "Messy pool: 5 adapters across 3 NLI tasks, mixed QA quality.",
        "adapters": [
            ("qnli_s42", AMBIG / "qnli_s42"),
            ("rte_s42", AMBIG / "rte_s42"),
            ("mnli_s42", AMBIG / "mnli_s42"),
            ("mnli_s7", AMBIG / "mnli_s7"),
            ("rte_s7", AMBIG / "rte_s7"),
        ],
    },
    {
        "id": "large_diverse_pool",
        "category": "E",
        "description": "Large pool: 7 adapters across SST-2 + QNLI + RTE. Advisory at scale.",
        "adapters": [
            ("sst2_r16_s42", VERIFIED / "sst2_r16_s42"),
            ("sst2_r16_s123", VERIFIED / "sst2_r16_s123"),
            ("qnli_r16_s42_v2", VERIFIED / "qnli_r16_s42_v2"),
            ("qnli_r16_s123_v2", VERIFIED / "qnli_r16_s123_v2"),
            ("rte_s42", AMBIG / "rte_s42"),
            ("rte_s7", AMBIG / "rte_s7"),
            ("qnli_s42", AMBIG / "qnli_s42"),
        ],
    },
]


# ---------------------------------------------------------------------------
# Run one inventory
# ---------------------------------------------------------------------------


def run_inventory(inv: dict, qa_pool: Path) -> dict:
    """Run merge-audit on all pairs, capture advisory data."""
    inv_id = inv["id"]
    inv_dir = ROUND_ROOT / inv_id
    pair_dir = inv_dir / "pair_reports"
    qa_dir = inv_dir / "qa"
    pair_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Inventory: {inv_id} (Category {inv['category']})")
    print(f"  {inv['description']}")
    print(f"  Adapters: {len(inv['adapters'])}")

    # Copy QA artifacts
    adapter_names = []
    for name, adapter_path in inv["adapters"]:
        qa_src = qa_pool / f"{name}_qa.json"
        qa_dst = qa_dir / f"{name}_qa.json"
        if qa_src.exists():
            import shutil
            shutil.copy(qa_src, qa_dst)
        adapter_names.append(name)

    # Run all pairs
    n = len(inv["adapters"])
    total_pairs = n * (n - 1) // 2
    print(f"  Running {total_pairs} pairs...")

    pair_results = []
    for i in range(n):
        for j in range(i + 1, n):
            name_a, path_a = inv["adapters"][i]
            name_b, path_b = inv["adapters"][j]
            pair_id = f"{name_a}_vs_{name_b}"

            qa_a = qa_dir / f"{name_a}_qa.json"
            qa_b = qa_dir / f"{name_b}_qa.json"

            out_json = pair_dir / f"{pair_id}.json"

            cmd = [
                sys.executable, "-m", "gradience.cli", "merge-audit",
                "--adapter-a", str(path_a),
                "--adapter-b", str(path_b),
            ]
            if qa_a.exists():
                cmd += ["--source-a-qa", str(qa_a)]
            if qa_b.exists():
                cmd += ["--source-b-qa", str(qa_b)]
            cmd += ["--emit-report", str(out_json)]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"    {pair_id}: FAILED - {result.stderr[:100]}")
                pair_results.append({
                    "pair_id": pair_id,
                    "error": result.stderr[:200],
                })
                continue

            # Load report and extract advisory
            try:
                report_data = json.loads(out_json.read_text())
                advisory = report_data.get("task_relationship_advisory")
                pair_risk = report_data.get("pair_risk", "?")
                dominant_issue = report_data.get("dominant_issue", "?")

                pair_results.append({
                    "pair_id": pair_id,
                    "adapter_a": name_a,
                    "adapter_b": name_b,
                    "pair_risk": pair_risk,
                    "dominant_issue": dominant_issue,
                    "has_advisory": advisory is not None,
                    "advisory_text": advisory,
                })

                adv_mark = " [ADVISORY]" if advisory else ""
                print(f"    {pair_id}: risk={pair_risk}, issue={dominant_issue}{adv_mark}")
            except Exception as e:
                pair_results.append({"pair_id": pair_id, "error": str(e)})

    # Run inventory summary
    summary_dir = inv_dir / "inventory"
    summary_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "gradience.cli", "summarize-inventory",
        "--qa-dir", str(qa_dir),
        "--report-dir", str(pair_dir),
        "--emit-report", str(summary_dir / "summary.json"),
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    # Run neighborhoods
    nbr_dir = inv_dir / "neighborhoods"
    nbr_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "gradience.cli", "suggest-neighborhoods",
        "--qa-dir", str(qa_dir),
        "--report-dir", str(pair_dir),
        "--emit-report", str(nbr_dir / "neighborhoods.json"),
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    # Summarize
    advisory_pairs = [p for p in pair_results if p.get("has_advisory")]
    non_advisory_pairs = [p for p in pair_results if not p.get("has_advisory") and "error" not in p]

    result = {
        "inventory_id": inv_id,
        "category": inv["category"],
        "adapter_count": n,
        "total_pairs": total_pairs,
        "advisory_count": len(advisory_pairs),
        "non_advisory_count": len(non_advisory_pairs),
        "pairs": pair_results,
    }

    # Save inventory result
    (inv_dir / "advisory_summary.json").write_text(json.dumps(result, indent=2))

    print(f"\n  Summary: {total_pairs} pairs, {len(advisory_pairs)} with advisory, {len(non_advisory_pairs)} without")
    if advisory_pairs:
        print(f"  Advisory-bearing pairs:")
        for p in advisory_pairs:
            print(f"    - {p['pair_id']} (risk={p['pair_risk']})")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("TASK-RELATIONSHIP ADVISORY — VALIDATION ROUND 01")
    print("=" * 70)

    # Step 1: Ensure QA artifacts
    print("\nStep 1: Generating QA artifacts...")
    qa_pool = ensure_qa_artifacts()

    # Step 2: Run inventories
    all_results = []
    for inv in INVENTORIES:
        result = run_inventory(inv, qa_pool)
        all_results.append(result)

    # Step 3: Cross-inventory summary
    print(f"\n{'='*70}")
    print("CROSS-INVENTORY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Inventory':<30s} {'Cat':<4s} {'Pairs':<6s} {'Adv':<5s} {'Rate':<8s}")
    print(f"{'-'*30} {'-'*4} {'-'*6} {'-'*5} {'-'*8}")

    total_pairs_all = 0
    total_advisory_all = 0
    for r in all_results:
        rate = r["advisory_count"] / r["total_pairs"] if r["total_pairs"] > 0 else 0
        total_pairs_all += r["total_pairs"]
        total_advisory_all += r["advisory_count"]
        print(f"{r['inventory_id']:<30s} {r['category']:<4s} {r['total_pairs']:<6d} {r['advisory_count']:<5d} {rate:<8.0%}")

    overall_rate = total_advisory_all / total_pairs_all if total_pairs_all > 0 else 0
    print(f"{'TOTAL':<30s} {'—':<4s} {total_pairs_all:<6d} {total_advisory_all:<5d} {overall_rate:<8.0%}")

    # Save cross-inventory summary
    summary = {
        "inventories": all_results,
        "totals": {
            "total_pairs": total_pairs_all,
            "total_advisory": total_advisory_all,
            "advisory_rate": overall_rate,
        },
    }
    (ROUND_ROOT / "round_01_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFull summary written to {ROUND_ROOT / 'round_01_summary.json'}")


if __name__ == "__main__":
    main()
