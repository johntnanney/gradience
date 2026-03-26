"""Integration tests for corpus append/summarize scripts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
from gradience.vnext.inventory.corpus_manifest import CorpusManifest
from gradience.vnext.inventory.neighborhoods import suggest_merge_neighborhoods
from gradience.vnext.merge.qa_report import MergeQAReport


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_neighborhood_report(fixture_dir: Path, out_path: Path) -> None:
    qa_artifacts = [
        AdapterQAArtifact.from_dict(_load_json(p))
        for p in sorted((fixture_dir / "qa").glob("*.json"))
    ]
    pair_reports = [
        MergeQAReport.from_dict(_load_json(p))
        for p in sorted((fixture_dir / "reports").glob("*.json"))
    ]
    report = suggest_merge_neighborhoods(qa_artifacts, pair_reports)
    report.to_json(out_path)


def _write_qa_artifact(
    out_path: Path,
    *,
    adapter_name: str,
    adapter_path: str,
    base_model: str = "llama",
    instance_id: str | None = None,
) -> None:
    adapter_section: dict[str, object] = {
        "name": adapter_name,
        "path": adapter_path,
        "base_model": base_model,
        "rank_nominal": 8,
        "n_layers": 8,
    }
    if instance_id is not None:
        # Forward-compatible optional identity hook.
        adapter_section["instance_id"] = instance_id

    qa = {
        "schema": "gradience.adapter_qa/v1",
        "adapter": adapter_section,
        "structural_summary": {
            "utilization_mean": 0.55,
            "rank_waste_ratio": 0.42,
            "flags": [],
        },
        "behavioral_summary": {"eval_available": True},
        "eligibility": {"status": "eligible"},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")


def _append_manifest_from_single_qa(
    *,
    run_id: str,
    qa_artifact: Path,
    corpus_root: Path,
    pair_report: Path,
    neighborhood_report: Path,
) -> None:
    p = subprocess.run(
        [
            sys.executable,
            "scripts/append_corpus_entry.py",
            "--run-id",
            run_id,
            "--date",
            "2026-03-17",
            "--qa-artifact",
            str(qa_artifact),
            "--pair-report",
            str(pair_report),
            "--neighborhood-report",
            str(neighborhood_report),
            "--corpus-root",
            str(corpus_root),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert p.returncode == 0, p.stderr


def _run_summarize(corpus_root: Path, summary_json: Path, summary_md: Path | None = None) -> tuple[dict, str]:
    cmd = [
        sys.executable,
        "scripts/summarize_corpus.py",
        "--corpus-root",
        str(corpus_root),
        "--emit-json",
        str(summary_json),
    ]
    if summary_md is not None:
        cmd.extend(["--emit-md", str(summary_md)])

    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert p.returncode == 0, p.stderr
    return _load_json(summary_json), p.stdout


class TestAppendCorpusEntryScript:
    def test_appends_valid_manifest(self, tmp_path: Path) -> None:
        fixture_dir = Path("examples/inventories/inventory_safe_small")
        neighborhood_path = tmp_path / "safe_neighborhood.json"
        _build_neighborhood_report(fixture_dir, neighborhood_path)

        corpus_root = tmp_path / "corpus"
        p = subprocess.run(
            [
                sys.executable,
                "scripts/append_corpus_entry.py",
                "--run-id",
                "safe_fixture_001",
                "--date",
                "2026-03-17T12:00:00Z",
                "--qa-dir",
                str(fixture_dir / "qa"),
                "--report-dir",
                str(fixture_dir / "reports"),
                "--neighborhood-report",
                str(neighborhood_path),
                "--corpus-root",
                str(corpus_root),
                "--note",
                "safe fixture manifest",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert p.returncode == 0, p.stderr
        manifest_path = corpus_root / "manifests" / "safe_fixture_001.json"
        assert manifest_path.exists()

        manifest = CorpusManifest.from_dict(_load_json(manifest_path))
        assert manifest.run_id == "safe_fixture_001"
        assert manifest.base_model == "llama"
        assert len(manifest.adapter_names) == 3
        assert len(manifest.pair_report_paths) == 3

    def test_fails_on_malformed_qa_artifact(self, tmp_path: Path) -> None:
        bad_qa = tmp_path / "bad_qa.json"
        bad_qa.write_text(
            json.dumps(
                {
                    "schema": "gradience.adapter_qa/v1",
                    "adapter": {"name": "bad"},
                }
            ),
            encoding="utf-8",
        )
        valid_report = Path("examples/inventories/inventory_safe_small/reports/safe_ab.json")
        valid_neighborhood = Path("examples/neighborhoods/sample_merge_neighborhoods.json")
        corpus_root = tmp_path / "corpus"

        p = subprocess.run(
            [
                sys.executable,
                "scripts/append_corpus_entry.py",
                "--run-id",
                "bad_fixture_001",
                "--date",
                "2026-03-17",
                "--base-model",
                "llama",
                "--qa-artifact",
                str(bad_qa),
                "--pair-report",
                str(valid_report),
                "--neighborhood-report",
                str(valid_neighborhood),
                "--corpus-root",
                str(corpus_root),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert p.returncode != 0
        assert "Malformed adapter QA artifact" in p.stderr
        assert not (corpus_root / "manifests" / "bad_fixture_001.json").exists()


class TestSummarizeCorpusScript:
    def test_summarizes_multiple_manifests(self, tmp_path: Path) -> None:
        corpus_root = tmp_path / "corpus"

        fixture_safe = Path("examples/inventories/inventory_safe_small")
        fixture_weak = Path("examples/inventories/inventory_with_weak_sources")
        neighborhood_safe = tmp_path / "safe_neighborhood.json"
        neighborhood_weak = tmp_path / "weak_neighborhood.json"
        _build_neighborhood_report(fixture_safe, neighborhood_safe)
        _build_neighborhood_report(fixture_weak, neighborhood_weak)

        for run_id, fixture, neighborhood in [
            ("safe_fixture_001", fixture_safe, neighborhood_safe),
            ("weak_fixture_001", fixture_weak, neighborhood_weak),
        ]:
            p = subprocess.run(
                [
                    sys.executable,
                    "scripts/append_corpus_entry.py",
                    "--run-id",
                    run_id,
                    "--date",
                    "2026-03-17",
                    "--qa-dir",
                    str(fixture / "qa"),
                    "--report-dir",
                    str(fixture / "reports"),
                    "--neighborhood-report",
                    str(neighborhood),
                    "--corpus-root",
                    str(corpus_root),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert p.returncode == 0, p.stderr

        summary_json = tmp_path / "corpus_summary.json"
        summary_md = tmp_path / "corpus_summary.md"
        p = subprocess.run(
            [
                sys.executable,
                "scripts/summarize_corpus.py",
                "--corpus-root",
                str(corpus_root),
                "--emit-json",
                str(summary_json),
                "--emit-md",
                str(summary_md),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert p.returncode == 0, p.stderr
        assert "CORPUS SUMMARY" in p.stdout
        assert summary_json.exists()
        assert summary_md.exists()

        summary = _load_json(summary_json)
        assert summary["inventory_count"] == 2
        assert summary["pair_report_count"] == 6
        assert summary["strict_block_candidate_pairs"] >= 1
        assert summary["neighborhood_counts"]["groups_total"] >= 2
        assert summary["neighborhood_counts"]["excluded_total"] >= 1
        assert summary["strategy_counts"].get("linear", 0) >= 1

    def test_duplicate_checkpoint_labels_count_distinct_instances(self, tmp_path: Path) -> None:
        corpus_root = tmp_path / "corpus"
        qa_a = tmp_path / "qa" / "checkpoint_50_run_a.json"
        qa_b = tmp_path / "qa" / "checkpoint_50_run_b.json"
        _write_qa_artifact(
            qa_a,
            adapter_name="checkpoint-50",
            adapter_path="runs/a/checkpoint-50",
        )
        _write_qa_artifact(
            qa_b,
            adapter_name="checkpoint-50",
            adapter_path="runs/b/checkpoint-50",
        )

        pair_report = Path("examples/inventories/inventory_safe_small/reports/safe_ab.json")
        neighborhood_report = Path("examples/neighborhoods/sample_merge_neighborhoods.json")
        _append_manifest_from_single_qa(
            run_id="dup_checkpoint_a",
            qa_artifact=qa_a,
            corpus_root=corpus_root,
            pair_report=pair_report,
            neighborhood_report=neighborhood_report,
        )
        _append_manifest_from_single_qa(
            run_id="dup_checkpoint_b",
            qa_artifact=qa_b,
            corpus_root=corpus_root,
            pair_report=pair_report,
            neighborhood_report=neighborhood_report,
        )

        summary, _ = _run_summarize(corpus_root, tmp_path / "summary_dup_names.json")
        assert summary["adapter_instance_count"] == 2
        assert summary["unique_adapter_count"] == 2
        assert summary["unique_adapter_display_name_count"] == 1

    def test_repeated_reference_to_same_adapter_dedupes_by_identity_key(self, tmp_path: Path) -> None:
        corpus_root = tmp_path / "corpus"
        qa_shared = tmp_path / "qa" / "checkpoint_50_shared.json"
        _write_qa_artifact(
            qa_shared,
            adapter_name="checkpoint-50",
            adapter_path="runs/shared/checkpoint-50",
        )

        pair_report = Path("examples/inventories/inventory_safe_small/reports/safe_ab.json")
        neighborhood_report = Path("examples/neighborhoods/sample_merge_neighborhoods.json")
        _append_manifest_from_single_qa(
            run_id="shared_checkpoint_a",
            qa_artifact=qa_shared,
            corpus_root=corpus_root,
            pair_report=pair_report,
            neighborhood_report=neighborhood_report,
        )
        _append_manifest_from_single_qa(
            run_id="shared_checkpoint_b",
            qa_artifact=qa_shared,
            corpus_root=corpus_root,
            pair_report=pair_report,
            neighborhood_report=neighborhood_report,
        )

        summary, _ = _run_summarize(corpus_root, tmp_path / "summary_shared.json")
        assert summary["adapter_instance_count"] == 1
        assert summary["unique_adapter_count"] == 1
        assert summary["unique_adapter_display_name_count"] == 1

    def test_mixed_explicit_and_fallback_identity_sources_do_not_collide(self, tmp_path: Path) -> None:
        corpus_root = tmp_path / "corpus"
        qa_explicit = tmp_path / "qa" / "checkpoint_50_explicit.json"
        qa_fallback = tmp_path / "qa" / "checkpoint_50_fallback.json"
        shared_adapter_path = "runs/shared/checkpoint-50"
        _write_qa_artifact(
            qa_explicit,
            adapter_name="checkpoint-50",
            adapter_path=shared_adapter_path,
            instance_id="instance-explicit-001",
        )
        _write_qa_artifact(
            qa_fallback,
            adapter_name="checkpoint-50",
            adapter_path=shared_adapter_path,
        )

        pair_report = Path("examples/inventories/inventory_safe_small/reports/safe_ab.json")
        neighborhood_report = Path("examples/neighborhoods/sample_merge_neighborhoods.json")
        _append_manifest_from_single_qa(
            run_id="mixed_identity_explicit",
            qa_artifact=qa_explicit,
            corpus_root=corpus_root,
            pair_report=pair_report,
            neighborhood_report=neighborhood_report,
        )
        _append_manifest_from_single_qa(
            run_id="mixed_identity_fallback",
            qa_artifact=qa_fallback,
            corpus_root=corpus_root,
            pair_report=pair_report,
            neighborhood_report=neighborhood_report,
        )

        summary, _ = _run_summarize(corpus_root, tmp_path / "summary_mixed_identity.json")
        assert summary["adapter_instance_count"] == 2
        assert summary["unique_adapter_count"] == 2
        assert summary["unique_adapter_display_name_count"] == 1

    def test_human_readable_label_summary_remains_available(self, tmp_path: Path) -> None:
        corpus_root = tmp_path / "corpus"
        qa_a = tmp_path / "qa" / "checkpoint_50_name_a.json"
        qa_b = tmp_path / "qa" / "checkpoint_50_name_b.json"
        _write_qa_artifact(
            qa_a,
            adapter_name="checkpoint-50",
            adapter_path="runs/name/a/checkpoint-50",
        )
        _write_qa_artifact(
            qa_b,
            adapter_name="checkpoint-50",
            adapter_path="runs/name/b/checkpoint-50",
        )

        pair_report = Path("examples/inventories/inventory_safe_small/reports/safe_ab.json")
        neighborhood_report = Path("examples/neighborhoods/sample_merge_neighborhoods.json")
        _append_manifest_from_single_qa(
            run_id="display_names_a",
            qa_artifact=qa_a,
            corpus_root=corpus_root,
            pair_report=pair_report,
            neighborhood_report=neighborhood_report,
        )
        _append_manifest_from_single_qa(
            run_id="display_names_b",
            qa_artifact=qa_b,
            corpus_root=corpus_root,
            pair_report=pair_report,
            neighborhood_report=neighborhood_report,
        )

        summary_md = tmp_path / "summary_display_names.md"
        summary, terminal = _run_summarize(
            corpus_root,
            tmp_path / "summary_display_names.json",
            summary_md,
        )
        assert "unique adapter display names:" in terminal
        md_text = summary_md.read_text(encoding="utf-8")
        assert "Unique adapter display names" in md_text
        assert summary["unique_adapter_display_name_count"] == 1
