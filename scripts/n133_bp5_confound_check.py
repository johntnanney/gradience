"""N133 B-P5 confound check.

Two tests, both CPU-only on already-pulled artifacts:

(1) Re-check inv_min_erank on a task subset that drops summarization (erank 9.0)
    and sst2 (erank 5.2). Remaining tasks {mnli, code, gsm8k, squad} have erank
    in a compressed band 6.04–7.65. Does erank still rank within-group?

(2) Run corrected-sign alignment triage (retain LOWEST alignment) on the full
    60 cross-task pairs and inspect which pairs it schedules as "safest to
    merge". If the top-safe list is dominated by summarization/code pairs,
    the binary-partition confound is confirmed at scale.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POD = PROJECT_ROOT / "sidecar" / "data" / "n133" / "pod_pull"
OUT = PROJECT_ROOT / "sidecar" / "data" / "n133"


def load():
    profiles = json.loads((POD / "audits" / "adapter_profiles.json").read_text())
    pair_sum = json.loads((POD / "audits" / "pair_alignment_summary.json").read_text())
    pair_full = json.loads((POD / "audits" / "pair_alignment_full.json").read_text())
    merges = json.loads((POD / "merges" / "merge_eval_summary.json").read_text())
    return profiles, pair_sum, pair_full, merges


def adapter_erank(profiles):
    return {
        name: float(np.mean([l["erank"] for l in p["per_layer"]]))
        for name, p in profiles.items()
    }


def max_deg(r):
    ta, tb = r["task_a"], r["task_b"]
    return max(r.get(f"degradation_{ta}", 0), r.get(f"degradation_{tb}", 0))


# ---------------------------------------------------------------------------
# Test 1: inv_min_erank on compressed-erank subset
# ---------------------------------------------------------------------------

def test1_compressed_subset(profiles, merges):
    print("=" * 72)
    print("  TEST 1: inv_min_erank on compressed-erank subset")
    print("  Dropping summarization (erank 9.0) and sst2 (erank 5.2)")
    print("  Remaining tasks: {mnli, code, gsm8k, squad}")
    print("=" * 72 + "\n")

    drop_tasks = {"summarization", "sst2"}
    erank = adapter_erank(profiles)

    kept_eranks = {n: e for n, e in erank.items()
                   if n.split("_")[0] not in drop_tasks}
    print("Kept adapter eranks:")
    for n in sorted(kept_eranks):
        print(f"  {n:<30s} {kept_eranks[n]:.3f}")
    print(f"  range: [{min(kept_eranks.values()):.3f}, "
          f"{max(kept_eranks.values()):.3f}]  "
          f"(was [5.14, 9.09] in full set)")

    # Filter evaluated cross-task pairs
    kept_pairs = []
    for r in merges["results"]:
        if r["category"] == "same_task":
            continue
        if r["task_a"] in drop_tasks or r["task_b"] in drop_tasks:
            continue
        md = max_deg(r)
        mine = min(erank[r["adapter_a"]], erank[r["adapter_b"]])
        kept_pairs.append({
            "pair": r["pair"],
            "task_a": r["task_a"],
            "task_b": r["task_b"],
            "min_erank": mine,
            "max_deg": md,
            "is_good": md < 0.10,
        })

    print(f"\nKept evaluated pairs: {len(kept_pairs)}/12")
    print(f"  good (max_deg < 0.10): {sum(p['is_good'] for p in kept_pairs)}")
    print(f"  bad:                   {sum(not p['is_good'] for p in kept_pairs)}")

    print("\nPer-pair:")
    print(f"  {'pair':<45s} {'min_erank':>10s} {'max_deg':>8s} {'good?':>6s}")
    for p in sorted(kept_pairs, key=lambda x: x["min_erank"]):
        tag = "YES" if p["is_good"] else "no"
        print(f"  {p['pair']:<45s} {p['min_erank']:>10.3f} "
              f"{p['max_deg']:>+8.3f} {tag:>6s}")

    # Spearman
    if len(kept_pairs) >= 4:
        scores = np.array([1.0 / p["min_erank"] for p in kept_pairs])
        degs = np.array([p["max_deg"] for p in kept_pairs])
        rho, p_rho = stats.spearmanr(scores, degs)
        r_p, p_r = stats.pearsonr(scores, degs)
        print(f"\nSpearman ρ(inv_min_erank, max_deg) = {rho:+.4f} (p={p_rho:.4f})")
        print(f"Pearson  r(inv_min_erank, max_deg) = {r_p:+.4f} (p={p_r:.4f})")

        # Triage recall: retain bottom k_safe by inv_min_erank (= highest min_erank)
        order_safe_first = np.argsort(scores)  # ascending inv_erank = high erank first
        n = len(kept_pairs)
        k_safe = max(2, n // 2)  # retain half
        retained = {kept_pairs[i]["pair"] for i in order_safe_first[:k_safe]}
        n_good_total = sum(1 for p in kept_pairs if p["is_good"])
        retained_good = sum(
            1 for p in kept_pairs if p["pair"] in retained and p["is_good"]
        )
        print(f"\nTriage recall (retain {k_safe} lowest-risk by inv_min_erank):")
        print(f"  retained_good = {retained_good} / {n_good_total}")

        # Also test raw mean_alignment (corrected-sign)
        # Need pair_sum lookup — handle at the top-level
    return kept_pairs


def test1_alignment_on_same_subset(kept_pairs, pair_sum):
    """Corrected-sign alignment triage on the same compressed subset."""
    for kp in kept_pairs:
        kp["mean_alignment"] = pair_sum[kp["pair"]]["mean_alignment"]

    scores = np.array([kp["mean_alignment"] for kp in kept_pairs])
    degs = np.array([kp["max_deg"] for kp in kept_pairs])
    rho, p_rho = stats.spearmanr(scores, degs)
    r_p, p_r = stats.pearsonr(scores, degs)
    print(f"\nSpearman ρ(mean_alignment, max_deg) = {rho:+.4f} (p={p_rho:.4f})")
    print(f"Pearson  r(mean_alignment, max_deg) = {r_p:+.4f} (p={p_r:.4f})")

    # Triage: retain LOWEST alignment as safest
    order = np.argsort(scores)
    n = len(kept_pairs)
    k_safe = max(2, n // 2)
    retained = {kept_pairs[i]["pair"] for i in order[:k_safe]}
    n_good = sum(1 for p in kept_pairs if p["is_good"])
    retained_good = sum(
        1 for p in kept_pairs if p["pair"] in retained and p["is_good"]
    )
    print(f"\nTriage recall (retain {k_safe} lowest-alignment):")
    print(f"  retained_good = {retained_good} / {n_good}")

    print("\nPer-pair alignment table:")
    for p in sorted(kept_pairs, key=lambda x: x["mean_alignment"]):
        tag = "YES" if p["is_good"] else "no"
        print(f"  {p['pair']:<45s} align={p['mean_alignment']:.4f}  "
              f"max_deg={p['max_deg']:+.3f}  good={tag}")


# ---------------------------------------------------------------------------
# Test 2: corrected-sign alignment triage on all 60 cross-task pairs
# ---------------------------------------------------------------------------

def test2_full_cross_task_ranking(pair_sum):
    print("\n" + "=" * 72)
    print("  TEST 2: corrected-sign alignment triage on all 60 cross-task pairs")
    print("  'Safest to merge' = LOWEST alignment")
    print("=" * 72 + "\n")

    cross = [
        (k, v["mean_alignment"], v["task_a"], v["task_b"])
        for k, v in pair_sum.items()
        if not v["is_same_task"]
    ]
    cross.sort(key=lambda x: x[1])  # ascending = safest first

    n = len(cross)
    print(f"Total cross-task pairs: {n}")
    print(f"Alignment range: [{cross[0][1]:.4f}, {cross[-1][1]:.4f}]  "
          f"(CV = {np.std([c[1] for c in cross])/np.mean([c[1] for c in cross]):.3f})")

    for label, cut in [("top 10 safest", 10), ("top 18 safest", 18),
                       ("top 30 safest", 30)]:
        top = cross[:cut]
        task_counter = Counter()
        for _k, _a, ta, tb in top:
            task_counter[ta] += 1
            task_counter[tb] += 1
        print(f"\n{label} (cumulative):")
        print(f"  Task appearances: {dict(task_counter.most_common())}")

    print("\nTop 18 safest cross-task pairs (what triage would schedule next):")
    print(f"  {'rank':>4s} {'pair':<50s} {'align':>8s}")
    for i, (k, a, _ta, _tb) in enumerate(cross[:18]):
        print(f"  {i+1:>4d} {k:<50s} {a:>8.4f}")

    print("\nBottom 18 (riskiest cross-task pairs):")
    print(f"  {'rank':>4s} {'pair':<50s} {'align':>8s}")
    for i, (k, a, _ta, _tb) in enumerate(cross[-18:]):
        rank = n - 18 + i + 1
        print(f"  {rank:>4d} {k:<50s} {a:>8.4f}")

    # Task composition analysis: what fraction of each task's pairs land in top-k safest?
    print("\nPer-task representation in bottom-half by alignment (safest 30):")
    tasks = set()
    for _k, _a, ta, tb in cross:
        tasks.add(ta)
        tasks.add(tb)
    safest_30 = set(k for k, _, _, _ in cross[:30])
    for t in sorted(tasks):
        total_cross = sum(1 for _k, _a, ta, tb in cross if ta == t or tb == t)
        in_safe = sum(1 for k, _, ta, tb in cross if (ta == t or tb == t)
                      and k in safest_30)
        print(f"  {t:<14s}: {in_safe:>2d} / {total_cross:>2d} pairs in safest 30  "
              f"({in_safe/total_cross*100:.0f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    profiles, pair_sum, pair_full, merges = load()

    kept = test1_compressed_subset(profiles, merges)
    test1_alignment_on_same_subset(kept, pair_sum)
    test2_full_cross_task_ranking(pair_sum)

    # Save a structured report
    report = {
        "experiment": "N133 B-P5 confound check",
        "test1_compressed_subset": {
            "dropped_tasks": ["summarization", "sst2"],
            "n_pairs": len(kept),
            "pairs": kept,
        },
    }
    (OUT / "bp5_confound_check.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Saved {OUT / 'bp5_confound_check.json'}")


if __name__ == "__main__":
    main()
