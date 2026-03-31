#!/usr/bin/env python3
"""Build a polished alpha bundle for checkpoint triage T02.

This script converts existing structured artifacts into one concise HTML report
plus a compact JSON summary/manifest for demo and onboarding.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRIAL_DIR = Path(__file__).resolve().parent
PRE = TRIAL_DIR / "preflight"
ALPHA_DIR = PRE / "alpha_bundle"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _esc(s: Any) -> str:
    return html.escape(str(s))


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body_rows: list[str] = []
    for row in rows:
        cells = "".join(f"<td>{_esc(c)}</td>" for c in row)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build() -> None:
    preflight = _read_json(PRE / "preflight_results.json")
    evidence = _read_json(TRIAL_DIR / "evidence" / "bootstrap_results.json")
    qa = _read_json(TRIAL_DIR / "qa_artifacts" / "qa_summary.json")
    pairwise = _read_json(TRIAL_DIR / "pairwise" / "pairwise_results.json")
    eval_results = _read_json(TRIAL_DIR / "eval_results.json")

    ALPHA_DIR.mkdir(parents=True, exist_ok=True)

    policy = preflight["policy_summary"]
    inv = preflight["inventory_summary"]
    plan = preflight["action_plan"]
    evidence_summary = evidence["summary"]
    qa_counts = qa["status_counts"]
    workflow_usefulness = eval_results["measurement_schema"]["workflow_usefulness"]

    evidence_rows = []
    for chk in evidence["checkpoints"]:
        evidence_rows.append(
            [
                chk["checkpoint_id"],
                chk["task"],
                f"{chk['checkpoint_score']:.4f}",
                f"{chk['base_score']:.4f}",
                f"{chk['delta_vs_base']:+.4f}",
                chk["evidence_status_candidate"],
            ]
        )

    pair_rows = []
    for p in pairwise["pairs"]:
        pair_rows.append(
            [
                p["pair_id"],
                p["task_relationship"],
                f"{p['compatibility_score']:.4f}",
                p["pair_risk"],
            ]
        )

    follow_rows = []
    for r in eval_results["follow_through"]["records"]:
        follow_rows.append([r["category"], ", ".join(r["checkpoints"]), r["task"], r["interpretation"]])

    summary = {
        "schema": "checkpoint_triage_alpha_bundle/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_trial": "checkpoint_inventory_t02",
        "policy_summary": policy,
        "inventory_snapshot": {
            "checkpoint_count": evidence_summary["checkpoint_count"],
            "pair_count": len(pairwise["pairs"]),
            "retained_count": plan["retained_count"],
            "near_miss_count": len(plan.get("near_miss_candidates", [])),
            "status_counts": qa_counts,
        },
        "scope_statement": {
            "shared_base_only": True,
            "small_encoder_only": True,
            "classification_only": True,
            "evidence_bootstrap_required": True,
        },
        "artifact_links": {
            "preflight_results": "../preflight_results.json",
            "inventory_summary": "../inventory/inventory_summary.json",
            "inventory_action_plan": "../inventory/inventory_action_plan.json",
            "qa_summary": "../../qa_artifacts/qa_summary.json",
            "pairwise_results": "../../pairwise/pairwise_results.json",
            "evidence_bootstrap": "../../evidence/bootstrap_results.json",
            "eval_results": "../../eval_results.json",
        },
    }

    _write_json(ALPHA_DIR / "alpha_summary.json", summary)

    manifest = {
        "schema": "checkpoint_triage_alpha_bundle_manifest/v1",
        "generated_utc": summary["generated_utc"],
        "bundle_dir": str(ALPHA_DIR),
        "files": [
            "alpha_summary.json",
            "bundle_manifest.json",
            "report.html",
        ],
        "source_artifacts": summary["artifact_links"],
    }
    _write_json(ALPHA_DIR / "bundle_manifest.json", manifest)

    html_doc = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Checkpoint Triage Alpha Workflow Report (T02)</title>
  <style>
    :root {{
      --bg: #f7f8f4;
      --ink: #1f2a37;
      --muted: #5f6b78;
      --card: #ffffff;
      --accent: #0f766e;
      --warn: #b45309;
      --bad: #b91c1c;
      --line: #dce2e8;
    }}
    body {{ margin: 0; font-family: Georgia, \"Times New Roman\", serif; background: var(--bg); color: var(--ink); }}
    .wrap {{ max-width: 1080px; margin: 24px auto; padding: 0 16px 40px; }}
    .hero {{ background: linear-gradient(135deg, #0f766e, #155e75); color: #fff; border-radius: 14px; padding: 20px; }}
    .hero h1 {{ margin: 0 0 6px; font-size: 1.6rem; }}
    .hero p {{ margin: 0; opacity: 0.95; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 14px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }}
    .k {{ font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    .v {{ font-size: 1.2rem; font-weight: 700; margin-top: 4px; }}
    h2 {{ margin-top: 26px; margin-bottom: 10px; }}
    p {{ line-height: 1.45; }}
    .scope {{ border-left: 4px solid var(--accent); padding-left: 10px; }}
    ul {{ margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; font-size: 0.92rem; }}
    th {{ background: #edf2f7; font-weight: 700; }}
    tr:last-child td {{ border-bottom: none; }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.8rem; }}
    .pill.good {{ background: #dcfce7; color: #166534; }}
    .pill.warn {{ background: #fef3c7; color: #92400e; }}
    .pill.bad {{ background: #fee2e2; color: #991b1b; }}
    .links code {{ font-size: 0.85rem; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <section class=\"hero\">
      <h1>Checkpoint Triage Alpha Workflow Report</h1>
      <p>Canonical example: <code>field_trials/checkpoint_inventory_t02</code> (generated { _esc(summary['generated_utc']) })</p>
      <div class=\"grid\">
        <div class=\"card\"><div class=\"k\">Inventory Type</div><div class=\"v\">{_esc(policy['inventory_type'])}</div></div>
        <div class=\"card\"><div class=\"k\">Dominant Driver</div><div class=\"v\">{_esc(policy['dominant_driver'])}</div></div>
        <div class=\"card\"><div class=\"k\">Checkpoints</div><div class=\"v\">{_esc(evidence_summary['checkpoint_count'])}</div></div>
        <div class=\"card\"><div class=\"k\">Pairs</div><div class=\"v\">{_esc(len(pairwise['pairs']))}</div></div>
      </div>
    </section>

    <section>
      <h2>Scope (Alpha Contract)</h2>
      <p class=\"scope\">This workflow is intentionally bounded: shared base only, small encoders only, classification only, and evidence bootstrap is required before triage decisions.</p>
      <ul>
        <li><span class=\"pill good\">shared_base_only</span></li>
        <li><span class=\"pill good\">small_encoder_only</span></li>
        <li><span class=\"pill good\">classification_only</span></li>
        <li><span class=\"pill warn\">evidence_bootstrap_required</span></li>
      </ul>
    </section>

    <section>
      <h2>Step 1 — Evidence Bootstrap (First-Class Gate)</h2>
      <p>Evidence status counts: eligible={_esc(evidence_summary['status_counts']['eligible'])}, uncertain={_esc(evidence_summary['status_counts']['uncertain'])}, flagged_weak={_esc(evidence_summary['status_counts']['flagged_weak'])}.</p>
      {_table(["checkpoint", "task", "checkpoint_score", "base_score", "delta_vs_base", "status"], evidence_rows)}
    </section>

    <section>
      <h2>Step 2 — QA Snapshot</h2>
      <p>QA status counts: eligible={_esc(qa_counts['eligible'])}, uncertain={_esc(qa_counts['uncertain'])}, flagged_weak={_esc(qa_counts['flagged_weak'])}.</p>
      <p>{_esc(policy['constraint'])}</p>
    </section>

    <section>
      <h2>Step 3 — Pairwise Compatibility</h2>
      <p>Pair risk counts: medium={_esc(inv['pair_risk_counts'].get('medium', 0))}, high={_esc(inv['pair_risk_counts'].get('high', 0))}.</p>
      {_table(["pair", "relationship", "compatibility", "risk"], pair_rows)}
    </section>

    <section>
      <h2>Step 4 — Action Plan</h2>
      <p><strong>Summary:</strong> {_esc(plan['summary_line'])}</p>
      <ul>
        <li>Retained candidates: {_esc(plan['retained_count'])}</li>
        <li>Near-miss candidates: {_esc(len(plan.get('near_miss_candidates', [])))}</li>
        <li>Cross-task caution entries: {_esc(len(plan.get('cross_task_caution', [])))}</li>
      </ul>
    </section>

    <section>
      <h2>Step 5 — Tiny Follow-through</h2>
      {_table(["category", "checkpoint(s)", "task", "interpretation"], follow_rows)}
      <p>Workflow usefulness: QA={_esc(workflow_usefulness['qa_usefulness'])}, pairwise={_esc(workflow_usefulness['pairwise_comparison_usefulness'])}, action plan={_esc(workflow_usefulness['action_plan_usefulness'])}, report clarity={_esc(workflow_usefulness['report_clarity'])}.</p>
    </section>

    <section class=\"links\">
      <h2>Source Artifacts</h2>
      <ul>
        <li><code>../preflight_results.json</code></li>
        <li><code>../inventory/inventory_summary.json</code></li>
        <li><code>../inventory/inventory_action_plan.json</code></li>
        <li><code>../../evidence/bootstrap_results.json</code></li>
        <li><code>../../qa_artifacts/qa_summary.json</code></li>
        <li><code>../../pairwise/pairwise_results.json</code></li>
        <li><code>../../eval_results.json</code></li>
      </ul>
    </section>
  </div>
</body>
</html>
"""

    (ALPHA_DIR / "report.html").write_text(html_doc)


if __name__ == "__main__":
    build()
