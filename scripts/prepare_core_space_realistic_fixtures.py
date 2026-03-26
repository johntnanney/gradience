"""Prepare local adapter copies for realistic core-space benchmark fixtures.

This script copies minimal PEFT adapter files into a deterministic local cache
used by `examples/core_space_benchmark/*realistic*.json` fixtures.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gradience.vnext.audit import find_peft_files
from gradience.vnext.merge.io import load_adapter

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _copy_minimal_adapter(
    *,
    source_dir: Path,
    target_dir: Path,
) -> dict[str, Any]:
    cfg, weights, _issues = find_peft_files(source_dir)
    if cfg is None:
        raise FileNotFoundError(f"No adapter_config.* found under {source_dir}")
    if weights is None:
        raise FileNotFoundError(f"No adapter_model.* found under {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    cfg_dst = target_dir / cfg.name
    weights_dst = target_dir / weights.name
    shutil.copy2(cfg, cfg_dst)
    shutil.copy2(weights, weights_dst)

    info = load_adapter(target_dir)
    return {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "config_file": cfg_dst.name,
        "weights_file": weights_dst.name,
        "rank": info.rank,
        "alpha": info.alpha,
        "lora_pair_count": len(info.lora_pairs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local realistic fixture adapters for run_core_space_benchmark.py"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/core_space_benchmark/real_adapters"),
        help="Output root for prepared adapter copies (default: results/core_space_benchmark/real_adapters)",
    )
    parser.add_argument(
        "--source-ambiguous-a",
        type=str,
        default="bench_runs/final_test/uniform_median_r16/checkpoint-50",
        help="Source adapter dir for realistic_ambiguous_pair adapter A",
    )
    parser.add_argument(
        "--source-ambiguous-b",
        type=str,
        default="bench_runs/qnli_test/per_layer/checkpoint-50",
        help="Source adapter dir for realistic_ambiguous_pair adapter B",
    )
    parser.add_argument(
        "--source-mismatch-a",
        type=str,
        default="test_priority_output/seed_123/probe_r16",
        help="Source adapter dir for realistic_semantic_mismatch adapter A",
    )
    parser.add_argument(
        "--source-mismatch-b",
        type=str,
        default="bench_runs/final_test/uniform_median_r16/checkpoint-50",
        help="Source adapter dir for realistic_semantic_mismatch adapter B",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root

    targets = [
        (
            "realistic_ambiguous_pair.adapter_a",
            _resolve_repo_path(args.source_ambiguous_a),
            output_root / "final_uniform_median_r16",
        ),
        (
            "realistic_ambiguous_pair.adapter_b",
            _resolve_repo_path(args.source_ambiguous_b),
            output_root / "qnli_per_layer_r8",
        ),
        (
            "realistic_semantic_mismatch.adapter_a",
            _resolve_repo_path(args.source_mismatch_a),
            output_root / "priority_probe_r16",
        ),
        (
            "realistic_semantic_mismatch.adapter_b",
            _resolve_repo_path(args.source_mismatch_b),
            output_root / "final_uniform_median_r16",
        ),
    ]

    records: list[dict[str, Any]] = []
    for label, source_dir, target_dir in targets:
        if not source_dir.is_dir():
            raise FileNotFoundError(f"{label}: source adapter directory not found: {source_dir}")
        copied = _copy_minimal_adapter(source_dir=source_dir, target_dir=target_dir)
        copied["label"] = label
        records.append(copied)

    manifest = {
        "prepared_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_root": str(output_root),
        "entries": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Prepared realistic fixture adapters under: {output_root}")
    for rec in records:
        print(
            f"- {rec['label']}: rank={rec['rank']}, alpha={rec['alpha']}, "
            f"pairs={rec['lora_pair_count']} -> {rec['target_dir']}"
        )


if __name__ == "__main__":
    main()

