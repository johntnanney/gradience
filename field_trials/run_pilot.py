#!/usr/bin/env python3
"""Field-trial pilot runner — downloads adapters and runs Gradience preflight.

Usage:
    python field_trials/run_pilot.py inventory_01_same_task_control

Requires: peft, transformers, torch, gradience (editable install).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from itertools import combinations
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("pilot")

FIELD_TRIALS_DIR = Path(__file__).resolve().parent


def download_adapters(manifest: dict, adapter_cache: Path) -> list[Path]:
    """Download all adapters from manifest to local cache. Returns list of local paths."""
    from huggingface_hub import snapshot_download

    paths = []
    for entry in manifest["adapters"]:
        adapter_id = entry["adapter_id"]
        local_name = adapter_id.replace("/", "__")
        local_path = adapter_cache / local_name

        if local_path.exists() and (local_path / "adapter_config.json").exists():
            log.info("cached  %s", adapter_id)
        else:
            log.info("downloading  %s", adapter_id)
            snapshot_download(
                repo_id=adapter_id,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
            )
        paths.append(local_path)
    return paths


def run_adapter_audits(
    manifest: dict,
    adapter_paths: list[Path],
    qa_dir: Path,
) -> list[Path]:
    """Run gradience audit on each adapter. Returns list of QA artifact paths."""
    from gradience.api import audit_adapter

    qa_paths = []
    for entry, adapter_path in zip(manifest["adapters"], adapter_paths):
        adapter_id = entry["adapter_id"]
        local_name = adapter_id.replace("/", "__")
        qa_path = qa_dir / f"{local_name}.json"

        if qa_path.exists():
            log.info("qa cached  %s", adapter_id)
            qa_paths.append(qa_path)
            continue

        log.info("auditing  %s", adapter_id)
        t0 = time.time()

        # Pass behavioral evidence if available in manifest
        kwargs: dict = {
            "peft_dir": adapter_path,
            "eval_dataset": entry.get("dataset"),
            "compute_udr": False,  # skip UDR — no base model weights cached
        }
        base_model = entry.get("base_model")
        if base_model:
            kwargs["base_model"] = base_model

        try:
            artifact = audit_adapter(**kwargs)
            qa_path.write_text(json.dumps(artifact.to_dict(), indent=2))
            log.info("  done in %.1fs  status=%s", time.time() - t0, artifact.status)
            qa_paths.append(qa_path)
        except Exception as e:
            log.error("  FAILED: %s", e)
            # Write a minimal error record so we can continue
            qa_path.write_text(json.dumps({
                "error": str(e),
                "adapter_id": adapter_id,
            }))
            qa_paths.append(qa_path)

    return qa_paths


def run_pair_reports(
    adapter_paths: list[Path],
    qa_paths: list[Path],
    reports_dir: Path,
) -> list[Path]:
    """Run merge-audit for all pairs. Returns list of report paths."""
    report_paths = []

    for (i, a_path), (j, b_path) in combinations(enumerate(adapter_paths), 2):
        pair_name = f"{a_path.name}__x__{b_path.name}"
        report_path = reports_dir / f"{pair_name}.json"

        if report_path.exists():
            log.info("report cached  %s", pair_name[:60])
            report_paths.append(report_path)
            continue

        log.info("merge-audit  %s", pair_name[:60])
        t0 = time.time()

        # Build CLI command — merge_risk_report delegates to CLI subprocess
        try:
            from gradience.api import merge_risk_report

            report = merge_risk_report(
                adapter_a=a_path,
                adapter_b=b_path,
                source_a_qa=qa_paths[i],
                source_b_qa=qa_paths[j],
            )
            report_path.write_text(json.dumps(report.to_dict(), indent=2))
            log.info("  done in %.1fs  risk=%s", time.time() - t0, report.pair_risk)
            report_paths.append(report_path)
        except Exception as e:
            log.error("  FAILED: %s", e)

    return report_paths


def run_inventory_summary(
    qa_dir: Path,
    reports_dir: Path,
    output_dir: Path,
) -> None:
    """Run summarize-inventory and emit-bundle."""
    import subprocess

    log.info("summarize-inventory → %s", output_dir)

    argv = [
        sys.executable, "-m", "gradience",
        "summarize-inventory",
        "--qa-dir", str(qa_dir),
        "--report-dir", str(reports_dir),
        "--emit-bundle", str(output_dir),
    ]

    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("summarize-inventory failed:\n%s\n%s", result.stdout, result.stderr)
    else:
        log.info("bundle written to %s", output_dir)
        if result.stdout.strip():
            log.info("stdout:\n%s", result.stdout[:500])


def generate_html_report(bundle_dir: Path, output_path: Path) -> None:
    """Generate HTML preflight report from the bundle."""
    from gradience.vnext.inventory.html_report import render_html_report

    html = render_html_report(bundle_dir)
    output_path.write_text(html)
    log.info("HTML report → %s", output_path)


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <inventory_dir_name>")
        sys.exit(1)

    inv_name = sys.argv[1]
    inv_dir = FIELD_TRIALS_DIR / inv_name

    manifest_path = inv_dir / "manifest.json"
    if not manifest_path.exists():
        log.error("No manifest.json in %s", inv_dir)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    log.info("=== Field Trial: %s ===", manifest["inventory_id"])
    log.info("Adapters: %d", len(manifest["adapters"]))

    # Set up directories
    adapter_cache = inv_dir / "adapter_cache"
    adapter_cache.mkdir(exist_ok=True)
    preflight_dir = inv_dir / "preflight"
    preflight_dir.mkdir(exist_ok=True)
    qa_dir = preflight_dir / "qa"
    qa_dir.mkdir(exist_ok=True)
    reports_dir = preflight_dir / "pair_reports"
    reports_dir.mkdir(exist_ok=True)
    bundle_dir = preflight_dir / "bundle"
    bundle_dir.mkdir(exist_ok=True)

    # Step 1: Download adapters
    log.info("--- Step 1: Download adapters ---")
    adapter_paths = download_adapters(manifest, adapter_cache)

    # Step 2: Run adapter audits
    log.info("--- Step 2: Adapter audits ---")
    qa_paths = run_adapter_audits(manifest, adapter_paths, qa_dir)

    # Step 3: Run pair merge-audits
    log.info("--- Step 3: Pair merge-audits ---")
    report_paths = run_pair_reports(adapter_paths, qa_paths, reports_dir)

    # Step 4: Summarize inventory
    log.info("--- Step 4: Summarize inventory ---")
    run_inventory_summary(qa_dir, reports_dir, bundle_dir)

    # Step 5: Generate HTML report
    log.info("--- Step 5: HTML report ---")
    generate_html_report(bundle_dir, preflight_dir / "report.html")

    log.info("=== Pilot complete: %s ===", manifest["inventory_id"])


if __name__ == "__main__":
    main()
