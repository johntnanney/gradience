"""Phase 3 driver for 02_draw_fewshot_examples.py.

Constructs a real Hugging Face dataset loader (pinned to dataset_version_hash
per benchmark) and invokes the script's main() with it. This is the
production execution path; the script's default loader raises
NotImplementedError to prevent accidental dataset access during testing.

Usage:
    python scripts/run_phase3_fewshot.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_ROOT))


def real_hf_loader(hf_dataset, hf_config, split, dataset_version_hash):
    """Load a benchmark split from Hugging Face, pinned to a revision.

    Two pragmatic adjustments at the loader layer (documented in
    IMPLEMENTATION_DEVIATIONS.md):

    1. cais/mmlu requires a config name (one config per subject). Our
       benchmarks.yaml uses hf_config=null; here we substitute "all" so
       the script's subject-column filter narrows to our panel.

    2. Some benchmarks use split-local row indices as their item ID
       (notably HellaSwag's `ind` field, which collides across splits
       for unrelated items). To make the leakage check correct in such
       cases, we prepend "{split}:" to the item ID. This is loader-side
       namespacing; the locked benchmarks.yaml is unchanged.
    """
    from datasets import load_dataset
    if hf_dataset == "cais/mmlu" and hf_config is None:
        hf_config = "all"
    ds = load_dataset(
        hf_dataset,
        hf_config,
        split=split,
        revision=dataset_version_hash,
    )
    rows = [dict(item) for item in ds]
    # Namespace split-local IDs. Applied uniformly across benchmarks: for
    # globally-unique IDs (ARC, MMLU question, etc.) this prefix is just
    # cosmetic; for split-local indices (HellaSwag) it makes the leakage
    # check correct.
    NAMESPACE_FIELDS = {
        "Rowan/hellaswag": "ind",
        "winogrande": "sentence",
        "truthful_qa": "question",
        "gsm8k": "question",
        "cais/mmlu": "question",
        "allenai/ai2_arc": "id",
    }
    field = NAMESPACE_FIELDS.get(hf_dataset)
    if field:
        for row in rows:
            if field in row and row[field] is not None:
                row[field] = f"{split}:{row[field]}"
    return rows


def main():
    spec = importlib.util.spec_from_file_location(
        "fewshot_script",
        PAPER_ROOT / "scripts" / "02_draw_fewshot_examples.py",
    )
    fewshot_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fewshot_module)

    argv = [
        "--config", str(PAPER_ROOT / "configs" / "study_config.yaml"),
        "--seeds", "42", "123", "2024",
        "--out", str(PAPER_ROOT / "manifests" / "fewshot_manifest.csv"),
        "--lock-out", str(
            PAPER_ROOT / "preregistration" / "appendices"
            / "fewshot_draws_LOCKED.json"
        ),
    ]
    return fewshot_module.main(argv=argv, loader=real_hf_loader)


if __name__ == "__main__":
    sys.exit(main())
