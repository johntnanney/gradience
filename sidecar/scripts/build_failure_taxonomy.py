#!/usr/bin/env python3
"""Build failure taxonomy and example flip catalog from analyzed predictions.

Consolidates per-example categories into the final 5-category taxonomy and
extracts representative examples for each.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "example_semantics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Load analyzed results
CASES = ["SR-01", "SR-02", "FR-01", "FR-02", "CT-01", "NM-01", "NM-02", "AN-01"]


def load_analyzed(case_id: str) -> dict:
    with open(RESULTS_DIR / f"analyzed_{case_id}.json") as f:
        return json.load(f)


def load_predictions(case_id: str) -> dict:
    with open(PREDICTIONS_DIR / f"{case_id}.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Taxonomy mapping: raw categories → final taxonomy
# ---------------------------------------------------------------------------
#
# Raw categories:
#   preserved_consensus, better_source_preserved, consensus_breakage,
#   better_source_loss, shared_failure, merge_recovery, source_a_loss, other
#
# Final taxonomy (5 categories):
#   A. Preserved consensus — both sources correct, merge correct
#   B. Consensus breakage — both sources correct, merge wrong
#   C. Better-source loss — one source correct, merge follows worse or fails
#   D. Ambiguity/neither-source — merge doesn't follow either source
#   E. Benign disagreement absorption — merge finds correct despite source disagreement
#
# Shared failure is excluded from the taxonomy (it's pre-existing, not merge-caused)

TAXONOMY_MAP = {
    "preserved_consensus": "A_preserved_consensus",
    "consensus_breakage": "B_consensus_breakage",
    "better_source_loss": "C_better_source_loss",
    "better_source_preserved": "E_benign_absorption",
    "merge_recovery": "E_benign_absorption",
    "shared_failure": "X_shared_failure",  # excluded from taxonomy (pre-existing)
    "source_a_loss": "C_better_source_loss",  # cross-task variant
    "other": "D_neither_source",  # captures multi-class neither-source cases
}


def classify_example_taxonomy(ex: dict, preds_data: dict) -> str:
    """Map a per-example analysis to the final taxonomy category."""
    raw = ex["category"]
    base_cat = TAXONOMY_MAP.get(raw, "D_neither_source")

    # Refine: check for neither-source behavior to upgrade to D
    if ex["neither_source"] and base_cat not in ("A_preserved_consensus", "X_shared_failure"):
        base_cat = "D_neither_source"

    return base_cat


def build_taxonomy():
    taxonomy_summary = {}
    flip_catalog = {}

    for case_id in CASES:
        analyzed = load_analyzed(case_id)
        preds_data = load_predictions(case_id)
        texts = None  # Load lazily if needed

        case_cats = {}
        case_examples = {}

        for ex in analyzed["per_example"]:
            tax_cat = classify_example_taxonomy(ex, preds_data)
            case_cats[tax_cat] = case_cats.get(tax_cat, 0) + 1

            # Collect flip examples (up to 8 per category)
            if tax_cat not in case_examples:
                case_examples[tax_cat] = []

            if len(case_examples[tax_cat]) < 8:
                case_examples[tax_cat].append({
                    "idx": ex["idx"],
                    "label": ex["label"],
                    "pred_a": ex["pred_a"],
                    "pred_b": ex["pred_b"],
                    "pred_m": ex["pred_m"],
                    "confidence_merged": ex["confidence_merged"],
                    "confidence_source_a": ex["confidence_source_a"],
                    "margin_gap": ex["margin_gap"],
                    "raw_category": ex["category"],
                })

        taxonomy_summary[case_id] = {
            "case_class": analyzed["case_class"],
            "num_examples": analyzed["num_examples"],
            "taxonomy_counts": case_cats,
            "taxonomy_rates": {
                k: round(v / analyzed["num_examples"], 4)
                for k, v in case_cats.items()
            },
        }

        flip_catalog[case_id] = {
            "case_class": analyzed["case_class"],
            "examples_by_category": case_examples,
        }

    return taxonomy_summary, flip_catalog


def build_taxonomy_definition():
    """The frozen taxonomy definition."""
    return {
        "version": "1.0",
        "date": "2026-03-28",
        "categories": [
            {
                "id": "A_preserved_consensus",
                "label": "Preserved consensus",
                "definition": "Both sources correct, merge also correct.",
                "behavioral_meaning": "The merge preserves agreed-upon knowledge. This is the safe baseline.",
                "associated_classes": ["safe_retained", "near_miss", "control"],
                "confidence_signature": "Typically high (>0.7). The merge is confident and correct.",
            },
            {
                "id": "B_consensus_breakage",
                "label": "Consensus breakage",
                "definition": "Both sources correct, merge wrong.",
                "behavioral_meaning": "The merge introduces a new error on an example where both sources agreed. This is the most important failure category — it indicates that the merge process itself is damaging shared knowledge.",
                "associated_classes": ["fragile"],
                "confidence_signature": "Variable. In fragile merges, often low-confidence (ambiguity collapse). In control merges, may be high-confidence wrong.",
            },
            {
                "id": "C_better_source_loss",
                "label": "Better-source loss",
                "definition": "One source was correct, the merge follows the worse source or produces a wrong answer.",
                "behavioral_meaning": "The merge fails to preserve the better source's discriminative advantage. In same-task pairs this reflects the merge averaging the two learned rules and landing on the wrong side. In cross-task pairs it reflects the cross-task adapter disrupting the task-relevant source's learned rule.",
                "associated_classes": ["near_miss", "fragile", "control"],
                "confidence_signature": "Typically moderate. The merge may be somewhat uncertain because the two sources disagree.",
            },
            {
                "id": "D_neither_source",
                "label": "Neither-source behavior",
                "definition": "The merged model's prediction matches neither source's prediction.",
                "behavioral_meaning": "The merge has produced a novel decision that neither source learned. This is the behavioral signature of representation-space averaging that destroys both sources' discriminative rules. In the mechanism ladder, this corresponds to the case where the readout gate is open and upstream pathology is transmitted.",
                "associated_classes": ["fragile", "control"],
                "confidence_signature": "Low confidence (ambiguity collapse). The merge is uncertain because it has no stable rule to apply.",
            },
            {
                "id": "E_benign_absorption",
                "label": "Benign disagreement absorption",
                "definition": "Sources disagree (or merge recovers from shared failure), but the merge lands on the correct answer.",
                "behavioral_meaning": "The merge successfully absorbs disagreement or corrects shared errors. This is a positive category — the merge is producing value by averaging complementary learned rules.",
                "associated_classes": ["safe_retained", "near_miss", "fragile"],
                "confidence_signature": "Variable but not collapsed. The merge found a correct rule even under disagreement.",
            },
        ],
        "excluded": {
            "id": "X_shared_failure",
            "label": "Shared failure (excluded)",
            "definition": "Neither source correct, merge also wrong.",
            "behavioral_meaning": "Pre-existing failure. The merge cannot be blamed for errors that both sources already made. Excluded from the taxonomy because it does not reflect merge behavior.",
        },
    }


def main():
    taxonomy_summary, flip_catalog = build_taxonomy()
    taxonomy_def = build_taxonomy_definition()

    # Save taxonomy
    output = {
        "taxonomy_definition": taxonomy_def,
        "per_case_summary": taxonomy_summary,
    }
    with open(RESULTS_DIR / "failure_taxonomy.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved: failure_taxonomy.json")

    # Save flip catalog
    with open(RESULTS_DIR / "example_flip_catalog.json", "w") as f:
        json.dump(flip_catalog, f, indent=2)
    print("Saved: example_flip_catalog.json")

    # Print summary
    print("\nTaxonomy rates by case:")
    for case_id in CASES:
        s = taxonomy_summary[case_id]
        print(f"\n{case_id} ({s['case_class']})")
        for cat, rate in sorted(s["taxonomy_rates"].items()):
            count = s["taxonomy_counts"][cat]
            print(f"  {cat:30s}: {count:4d}  ({rate*100:5.1f}%)")


if __name__ == "__main__":
    main()
