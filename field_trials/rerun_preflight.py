#!/usr/bin/env python3
"""Re-run Gradience preflight using evidence from the bootstrap step.

Reads evidence/*.json files and feeds behavioral scores into audit_adapter(),
then runs the full merge-audit + summarize-inventory + HTML report pipeline.

Usage:
    python field_trials/rerun_preflight.py inventory_01_same_task_control
"""
from __future__ import annotations

import json
import logging
import sys
import time
from itertools import combinations
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("rerun")

FIELD_TRIALS_DIR = Path(__file__).resolve().parent


def run_audits_with_evidence(
    manifest: dict,
    adapter_cache: Path,
    evidence_dir: Path,
    qa_dir: Path,
) -> list[tuple[Path, Path]]:
    """Run adapter audits with behavioral evidence. Returns (adapter_path, qa_path) pairs."""
    from gradience.api import audit_adapter

    pairs = []
    for entry in manifest["adapters"]:
        adapter_id = entry["adapter_id"]
        local_name = adapter_id.replace("/", "__")
        adapter_path = adapter_cache / local_name
        qa_path = qa_dir / f"{local_name}.json"
        evidence_path = evidence_dir / f"{local_name}.json"

        # Load evidence
        evidence = {}
        if evidence_path.exists():
            evidence = json.loads(evidence_path.read_text())

        if evidence.get("error"):
            log.warning("skipping  %s  (evidence error: %s)", adapter_id, evidence["error"][:60])
            continue

        log.info("auditing  %s  with evidence", adapter_id)
        t0 = time.time()

        kwargs: dict = {
            "peft_dir": adapter_path,
            "eval_dataset": evidence.get("eval_dataset", entry.get("dataset")),
            "compute_udr": False,
        }

        base_model = entry.get("base_model")
        if base_model:
            kwargs["base_model"] = base_model

        # Feed behavioral evidence
        if evidence.get("adapter_score") is not None:
            kwargs["adapter_score"] = evidence["adapter_score"]
            kwargs["base_score"] = evidence.get("base_score", 0.0)
            kwargs["metric_name"] = evidence.get("metric_name", "accuracy")
            kwargs["lower_is_better"] = evidence.get("lower_is_better", False)

        try:
            artifact = audit_adapter(**kwargs)
            qa_path.write_text(json.dumps(artifact.to_dict(), indent=2))
            d = artifact.to_dict()
            status = d.get("eligibility", {}).get("status", "?")
            confidence = d.get("eligibility", {}).get("confidence", "?")
            log.info("  done in %.1fs  status=%s  confidence=%s",
                     time.time() - t0, status, confidence)
            pairs.append((adapter_path, qa_path))
        except Exception as e:
            log.error("  FAILED: %s", e)

    return pairs


def run_pair_reports(
    pairs: list[tuple[Path, Path]],
    reports_dir: Path,
) -> list[Path]:
    """Run merge-audit for all pairs."""
    from gradience.api import merge_risk_report

    report_paths = []
    for (i, (a_path, a_qa)), (j, (b_path, b_qa)) in combinations(enumerate(pairs), 2):
        pair_name = f"{a_path.name}__x__{b_path.name}"
        report_path = reports_dir / f"{pair_name}.json"

        log.info("merge-audit  %s", pair_name[:70])
        t0 = time.time()

        try:
            report = merge_risk_report(
                adapter_a=a_path,
                adapter_b=b_path,
                source_a_qa=a_qa,
                source_b_qa=b_qa,
            )
            report_path.write_text(json.dumps(report.to_dict(), indent=2))
            log.info("  done in %.1fs  risk=%s", time.time() - t0, report.pair_risk)
            report_paths.append(report_path)
        except Exception as e:
            log.error("  FAILED: %s", e)

    return report_paths


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <inventory_dir_name>")
        sys.exit(1)

    inv_name = sys.argv[1]
    inv_dir = FIELD_TRIALS_DIR / inv_name
    manifest = json.loads((inv_dir / "manifest.json").read_text())

    log.info("=== Preflight Rerun (with evidence): %s ===", manifest["inventory_id"])

    adapter_cache = inv_dir / "adapter_cache"
    evidence_dir = inv_dir / "evidence"
    # Write to a new preflight_v2 directory to preserve the original
    # Use a new directory name each run to avoid permission issues
    import datetime
    ts = datetime.datetime.now().strftime("%H%M%S")
    preflight_dir = inv_dir / f"preflight_ev_{ts}"
    preflight_dir.mkdir(exist_ok=True)
    qa_dir = preflight_dir / "qa"
    qa_dir.mkdir(exist_ok=True)
    reports_dir = preflight_dir / "pair_reports"
    reports_dir.mkdir(exist_ok=True)
    bundle_dir = preflight_dir / "bundle"
    bundle_dir.mkdir(exist_ok=True)

    # Step 1: Audits with evidence
    log.info("--- Step 1: Adapter audits (with evidence) ---")
    pairs = run_audits_with_evidence(manifest, adapter_cache, evidence_dir, qa_dir)
    log.info("  %d adapters audited successfully", len(pairs))

    # Step 2: Pair reports
    log.info("--- Step 2: Pair merge-audits ---")
    report_paths = run_pair_reports(pairs, reports_dir)

    # Step 3: Summarize inventory
    log.info("--- Step 3: Summarize inventory ---")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "gradience", "summarize-inventory",
         "--qa-dir", str(qa_dir), "--report-dir", str(reports_dir),
         "--emit-bundle", str(bundle_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error("summarize-inventory failed:\n%s\n%s", result.stdout, result.stderr)
    else:
        log.info("bundle → %s", bundle_dir)
        if result.stdout.strip():
            log.info("stdout:\n%s", result.stdout[:800])

    # Step 4: HTML report
    log.info("--- Step 4: HTML report ---")
    from gradience.vnext.inventory.html_report import render_html_report
    html = render_html_report(bundle_dir)
    html_path = preflight_dir / "report.html"
    html_path.write_text(html)
    log.info("HTML report → %s", html_path)

    log.info("=== Rerun complete: %s ===", manifest["inventory_id"])


if __name__ == "__main__":
    main()
