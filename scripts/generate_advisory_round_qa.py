"""Generate QA artifacts for advisory round inventories."""
from __future__ import annotations

import json
from pathlib import Path

from gradience.api import audit_adapter

# All adapters needed for the 5 inventories with their verified eval scores
ADAPTERS = [
    # Verified adjudication (SST-2)
    {"name": "sst2_r16_s42", "dir": "results/adjudication_study/verified_adapters/sst2_r16_s42",
     "task": "sst2_dev", "score": 0.822, "base": 0.466, "metric": "accuracy"},
    {"name": "sst2_r16_s123", "dir": "results/adjudication_study/verified_adapters/sst2_r16_s123",
     "task": "sst2_dev", "score": 0.820, "base": 0.596, "metric": "accuracy"},
    {"name": "sst2_r8_s42", "dir": "results/adjudication_study/verified_adapters/sst2_r8_s42",
     "task": "sst2_dev", "score": 0.828, "base": 0.466, "metric": "accuracy"},
    # Verified adjudication (QNLI)
    {"name": "qnli_r16_s42", "dir": "results/adjudication_study/verified_adapters/qnli_r16_s42",
     "task": "qnli_dev", "score": 0.492, "base": 0.466, "metric": "accuracy"},
    {"name": "qnli_r16_s123", "dir": "results/adjudication_study/verified_adapters/qnli_r16_s123",
     "task": "qnli_dev", "score": 0.542, "base": 0.430, "metric": "accuracy"},
    {"name": "qnli_r8_s42", "dir": "results/adjudication_study/verified_adapters/qnli_r8_s42",
     "task": "qnli_dev", "score": 0.492, "base": 0.466, "metric": "accuracy"},
    # NLI family
    {"name": "mnli_s42", "dir": "results/adjudication_study/ambiguous_relationship_regime/adapters/mnli_s42",
     "task": "mnli_dev", "score": 0.530, "base": 0.352, "metric": "accuracy"},
    {"name": "mnli_s7", "dir": "results/adjudication_study/ambiguous_relationship_regime/adapters/mnli_s7",
     "task": "mnli_dev", "score": 0.532, "base": 0.302, "metric": "accuracy"},
    {"name": "qnli_s42", "dir": "results/adjudication_study/ambiguous_relationship_regime/adapters/qnli_s42",
     "task": "qnli_dev", "score": 0.696, "base": 0.470, "metric": "accuracy"},
    {"name": "qnli_s7", "dir": "results/adjudication_study/ambiguous_relationship_regime/adapters/qnli_s7",
     "task": "qnli_dev", "score": 0.710, "base": 0.542, "metric": "accuracy"},
    {"name": "rte_s42", "dir": "results/adjudication_study/ambiguous_relationship_regime/adapters/rte_s42",
     "task": "rte_dev", "score": 0.567, "base": 0.516, "metric": "accuracy"},
    {"name": "rte_s7", "dir": "results/adjudication_study/ambiguous_relationship_regime/adapters/rte_s7",
     "task": "rte_dev", "score": 0.556, "base": 0.473, "metric": "accuracy"},
]

QA_OUT = Path("results/task_advisory_round/shared_qa")


def main():
    QA_OUT.mkdir(parents=True, exist_ok=True)

    for entry in ADAPTERS:
        out_path = QA_OUT / f"{entry['name']}_qa.json"
        if out_path.exists():
            print(f"  {entry['name']}: already exists, skipping")
            continue

        print(f"  Generating QA: {entry['name']}...")
        artifact = audit_adapter(
            peft_dir=entry["dir"],
            base_model="distilbert-base-uncased",
            adapter_score=entry["score"],
            base_score=entry["base"],
            metric_name=entry["metric"],
            lower_is_better=False,
            eval_dataset=entry["task"],
        )

        d = artifact.to_dict()
        with open(out_path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"    -> {d.get('eligibility', {}).get('status', '?')}")

    print(f"\nAll QA artifacts in {QA_OUT}/")


if __name__ == "__main__":
    main()
