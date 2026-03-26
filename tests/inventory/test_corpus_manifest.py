"""Tests for gradience.corpus_manifest/v1 validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gradience.exceptions import QASchemaError
from gradience.vnext.inventory.corpus_manifest import SCHEMA_ID, CorpusManifest


def _valid_dict() -> dict:
    return {
        "schema": SCHEMA_ID,
        "run_id": "inventory_safe_small_20260317",
        "date": "2026-03-17T10:20:30Z",
        "base_model": "llama",
        "adapter_names": ["safe_a", "safe_b"],
        "qa_artifact_paths": [
            "examples/inventories/inventory_safe_small/qa/safe_a.json",
            "examples/inventories/inventory_safe_small/qa/safe_b.json",
        ],
        "pair_report_paths": [
            "examples/inventories/inventory_safe_small/reports/safe_ab.json",
        ],
        "neighborhood_report_path": "examples/neighborhoods/sample_merge_neighborhoods.json",
        "downstream_outcome_paths": ["results/some_eval.json"],
        "notes": ["fixture entry"],
    }


class TestCorpusManifestValidation:
    def test_roundtrip(self) -> None:
        manifest = CorpusManifest.from_dict(_valid_dict())
        assert manifest.run_id == "inventory_safe_small_20260317"
        assert manifest.base_model == "llama"
        assert manifest.adapter_names == ("safe_a", "safe_b")
        reloaded = CorpusManifest.from_dict(manifest.to_dict())
        assert reloaded == manifest

    def test_to_json_roundtrip(self, tmp_path: Path) -> None:
        manifest = CorpusManifest.from_dict(_valid_dict())
        out = tmp_path / "manifest.json"
        manifest.to_json(out)
        with open(out, encoding="utf-8") as f:
            loaded = json.load(f)
        assert CorpusManifest.from_dict(loaded) == manifest

    def test_missing_required_field(self) -> None:
        d = _valid_dict()
        del d["run_id"]
        with pytest.raises(QASchemaError, match="Missing required field: run_id"):
            CorpusManifest.from_dict(d)

    def test_wrong_schema(self) -> None:
        d = _valid_dict()
        d["schema"] = "gradience.corpus_manifest/v99"
        with pytest.raises(QASchemaError, match="Expected schema"):
            CorpusManifest.from_dict(d)

    def test_rejects_empty_adapter_list(self) -> None:
        d = _valid_dict()
        d["adapter_names"] = []
        with pytest.raises(QASchemaError, match="adapter_names"):
            CorpusManifest.from_dict(d)

    def test_rejects_invalid_date(self) -> None:
        d = _valid_dict()
        d["date"] = "17-03-2026"
        with pytest.raises(QASchemaError, match="ISO date/time"):
            CorpusManifest.from_dict(d)
