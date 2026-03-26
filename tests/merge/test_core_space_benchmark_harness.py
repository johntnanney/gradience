"""Integration tests for scripts/run_core_space_benchmark.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _latest_run_dir(out_root: Path) -> Path:
    runs = sorted([p for p in out_root.iterdir() if p.is_dir()])
    assert runs, f"No run directories found in {out_root}"
    return runs[-1]


class TestCoreSpaceBenchmarkHarness:
    def test_runs_all_fixtures_and_emits_decision_bundle(self, tmp_path: Path) -> None:
        out_root = tmp_path / "core_space_eval"
        p = subprocess.run(
            [
                sys.executable,
                "scripts/run_core_space_benchmark.py",
                "--fixtures-root",
                "examples/core_space_benchmark",
                "--out-root",
                str(out_root),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert p.returncode == 0, p.stderr

        run_dir = _latest_run_dir(out_root)
        summary_json = run_dir / "summary.json"
        summary_md = run_dir / "summary.md"
        decision_json = run_dir / "decision.json"
        assert summary_json.exists()
        assert summary_md.exists()
        assert decision_json.exists()

        with open(summary_json, encoding="utf-8") as f:
            summary = json.load(f)
        with open(decision_json, encoding="utf-8") as f:
            decision = json.load(f)

        fixtures = summary["fixtures"]
        fixture_map = {row["fixture_id"]: row for row in fixtures}
        assert set(fixture_map.keys()) == {
            "balanced_same_domain",
            "balanced_cross_domain",
            "moderate_risk_pairs",
            "same_rank_semantically_distant",
            "deliberately_mismatched_random",
            "realistic_ambiguous_pair",
            "realistic_semantic_mismatch",
        }

        assert fixture_map["balanced_same_domain"]["dominant_status"] == "compatible"
        assert fixture_map["same_rank_semantically_distant"]["dominant_status"] in {"marginal", "incompatible"}
        assert fixture_map["deliberately_mismatched_random"]["dominant_status"] in {"marginal", "incompatible"}
        for realistic_key in ("realistic_ambiguous_pair", "realistic_semantic_mismatch"):
            row = fixture_map[realistic_key]
            if row["available"]:
                assert row["dominant_status"] in {"compatible", "marginal", "incompatible"}
                assert row["verdict"] in {"passed", "partially passed", "unexpected_status_pattern"}
                assert row["trial_count"] == 3
            else:
                assert row["dominant_status"] == "not_applicable"
                assert row["verdict"] == "unavailable"
                assert row["trial_count"] == 0
                assert row["unavailable_reason"]

        for row in fixtures:
            if row["available"]:
                assert row["trial_count"] == 3
                assert row["verdict"] in {"passed", "partially passed", "unexpected_status_pattern"}
            else:
                assert row["trial_count"] == 0
                assert row["verdict"] == "unavailable"

        assert decision["decision"] in {"retain_optional", "promote_advanced", "shelve"}
        assert "criteria" in decision
        assert "status_spread_nontrivial" in decision["criteria"]

    def test_fixture_subset_option(self, tmp_path: Path) -> None:
        out_root = tmp_path / "core_space_subset"
        p = subprocess.run(
            [
                sys.executable,
                "scripts/run_core_space_benchmark.py",
                "--fixtures-root",
                "examples/core_space_benchmark",
                "--out-root",
                str(out_root),
                "--fixture",
                "balanced_same_domain",
                "--fixture",
                "same_rank_semantically_distant",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert p.returncode == 0, p.stderr

        run_dir = _latest_run_dir(out_root)
        with open(run_dir / "summary.json", encoding="utf-8") as f:
            summary = json.load(f)
        names = sorted([row["fixture_id"] for row in summary["fixtures"]])
        assert names == ["balanced_same_domain", "same_rank_semantically_distant"]

    def test_missing_realistic_fixture_defaults_to_unavailable(self, tmp_path: Path) -> None:
        fixtures_root = tmp_path / "fixtures"
        fixtures_root.mkdir(parents=True, exist_ok=True)

        (fixtures_root / "synthetic_ok.json").write_text(
            json.dumps(
                {
                    "fixture_id": "synthetic_ok",
                    "category": "synthetic",
                    "description": "simple synthetic fixture",
                    "evidence_tier": "synthetic",
                    "seeds": [1, 2],
                    "n_layers": 4,
                    "d_in": 16,
                    "d_out": 16,
                    "r_a": 4,
                    "r_b": 4,
                    "alpha_a": 8.0,
                    "alpha_b": 8.0,
                    "generator": {"mode": "same_domain_noise", "noise_scale": 0.03},
                    "expected_dominant_statuses": ["compatible", "marginal"],
                    "expected_trial_statuses": ["compatible", "marginal", "incompatible"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (fixtures_root / "real_missing.json").write_text(
            json.dumps(
                {
                    "fixture_id": "real_missing",
                    "category": "realistic missing",
                    "description": "missing local adapter pair",
                    "evidence_tier": "realistic",
                    "seeds": [1, 2],
                    "n_layers": 4,
                    "d_in": 16,
                    "d_out": 16,
                    "r_a": 8,
                    "r_b": 8,
                    "alpha_a": 16.0,
                    "alpha_b": 16.0,
                    "generator": {
                        "mode": "real_adapter_pair",
                        "adapter_a_dir": "does/not/exist/a",
                        "adapter_b_dir": "does/not/exist/b",
                        "max_layers": 4,
                    },
                    "expected_dominant_statuses": ["marginal", "incompatible"],
                    "expected_trial_statuses": ["marginal", "incompatible"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        out_root = tmp_path / "out"
        p = subprocess.run(
            [
                sys.executable,
                "scripts/run_core_space_benchmark.py",
                "--fixtures-root",
                str(fixtures_root),
                "--out-root",
                str(out_root),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert p.returncode == 0, p.stderr

        run_dir = _latest_run_dir(out_root)
        with open(run_dir / "summary.json", encoding="utf-8") as f:
            summary = json.load(f)
        rows = {row["fixture_id"]: row for row in summary["fixtures"]}
        assert rows["synthetic_ok"]["available"] is True
        assert rows["real_missing"]["available"] is False
        assert rows["real_missing"]["verdict"] == "unavailable"
        assert rows["real_missing"]["trial_count"] == 0
        assert rows["real_missing"]["unavailable_reason"]

    def test_require_realistic_fixtures_fails_on_missing_inputs(self, tmp_path: Path) -> None:
        fixtures_root = tmp_path / "fixtures"
        fixtures_root.mkdir(parents=True, exist_ok=True)
        (fixtures_root / "real_missing.json").write_text(
            json.dumps(
                {
                    "fixture_id": "real_missing",
                    "category": "realistic missing",
                    "description": "missing local adapter pair",
                    "evidence_tier": "realistic",
                    "seeds": [1],
                    "n_layers": 4,
                    "d_in": 16,
                    "d_out": 16,
                    "r_a": 8,
                    "r_b": 8,
                    "alpha_a": 16.0,
                    "alpha_b": 16.0,
                    "generator": {
                        "mode": "real_adapter_pair",
                        "adapter_a_dir": "does/not/exist/a",
                        "adapter_b_dir": "does/not/exist/b",
                        "max_layers": 4,
                    },
                    "expected_dominant_statuses": ["marginal", "incompatible"],
                    "expected_trial_statuses": ["marginal", "incompatible"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        out_root = tmp_path / "out"
        p = subprocess.run(
            [
                sys.executable,
                "scripts/run_core_space_benchmark.py",
                "--fixtures-root",
                str(fixtures_root),
                "--out-root",
                str(out_root),
                "--require-realistic-fixtures",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert p.returncode != 0
        assert "Required realistic fixture" in p.stderr
