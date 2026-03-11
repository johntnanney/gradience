"""Tests for InventorySummary dataclass and from_dict validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from gradience.exceptions import QASchemaError
from gradience.vnext.inventory.summary import SCHEMA_ID, InventorySummary


def _valid_dict() -> dict:
    """Return a minimal valid inventory_summary/v1 dict."""
    return {
        "schema": SCHEMA_ID,
        "sources": {"adapter_qa": 3, "merge_report": 2},
        "adapter_status_counts": {"eligible": 2, "flagged_weak": 1},
        "adapter_flag_counts": {"low_utilization": 1},
        "pair_risk_counts": {"low": 1, "high": 1},
        "recommended_strategy_counts": {"linear": 1, "audit_aware": 1},
        "dominant_issue_counts": {"none": 1, "subspace_conflict": 1},
        "strict_qa_block_candidates": 1,
        "notes": ["Example note"],
    }


class TestFromDictValidation:
    """Validate from_dict schema enforcement."""

    def test_valid_roundtrip(self) -> None:
        d = _valid_dict()
        obj = InventorySummary.from_dict(d)
        assert obj.sources == d["sources"]
        assert obj.adapter_status_counts == d["adapter_status_counts"]
        assert obj.adapter_flag_counts == d["adapter_flag_counts"]
        assert obj.pair_risk_counts == d["pair_risk_counts"]
        assert obj.recommended_strategy_counts == d["recommended_strategy_counts"]
        assert obj.dominant_issue_counts == d["dominant_issue_counts"]
        assert obj.strict_qa_block_candidates == 1
        assert obj.notes == ("Example note",)

    def test_to_dict_roundtrip(self) -> None:
        d = _valid_dict()
        obj = InventorySummary.from_dict(d)
        reconstructed = InventorySummary.from_dict(obj.to_dict())
        assert reconstructed == obj

    def test_to_json_roundtrip(self) -> None:
        d = _valid_dict()
        obj = InventorySummary.from_dict(d)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            obj.to_json(path)
            with open(path) as f:
                loaded = json.load(f)
        reconstructed = InventorySummary.from_dict(loaded)
        assert reconstructed == obj

    def test_missing_schema_raises(self) -> None:
        d = _valid_dict()
        del d["schema"]
        with pytest.raises(QASchemaError, match="Missing required field: schema"):
            InventorySummary.from_dict(d)

    def test_wrong_schema_raises(self) -> None:
        d = _valid_dict()
        d["schema"] = "gradience.other/v1"
        with pytest.raises(QASchemaError, match="Expected schema"):
            InventorySummary.from_dict(d)

    def test_missing_sources_raises(self) -> None:
        d = _valid_dict()
        del d["sources"]
        with pytest.raises(QASchemaError, match="Missing required section: sources"):
            InventorySummary.from_dict(d)

    def test_missing_adapter_status_counts_raises(self) -> None:
        d = _valid_dict()
        del d["adapter_status_counts"]
        with pytest.raises(QASchemaError, match="Missing required section: adapter_status_counts"):
            InventorySummary.from_dict(d)

    def test_missing_pair_risk_counts_raises(self) -> None:
        d = _valid_dict()
        del d["pair_risk_counts"]
        with pytest.raises(QASchemaError, match="Missing required section: pair_risk_counts"):
            InventorySummary.from_dict(d)

    def test_missing_strict_qa_block_candidates_raises(self) -> None:
        d = _valid_dict()
        del d["strict_qa_block_candidates"]
        with pytest.raises(QASchemaError, match="Missing required field: strict_qa_block_candidates"):
            InventorySummary.from_dict(d)

    def test_non_int_count_value_raises(self) -> None:
        d = _valid_dict()
        d["adapter_status_counts"]["eligible"] = 2.5
        with pytest.raises(QASchemaError, match="Values in 'adapter_status_counts' must be int"):
            InventorySummary.from_dict(d)

    def test_non_int_source_count_raises(self) -> None:
        d = _valid_dict()
        d["sources"]["adapter_qa"] = "three"
        with pytest.raises(QASchemaError, match="Values in 'sources' must be int"):
            InventorySummary.from_dict(d)

    def test_non_int_strict_qa_raises(self) -> None:
        d = _valid_dict()
        d["strict_qa_block_candidates"] = 1.0
        with pytest.raises(QASchemaError, match="must be int"):
            InventorySummary.from_dict(d)

    def test_notes_must_be_list_of_str(self) -> None:
        d = _valid_dict()
        d["notes"] = [1, 2]
        with pytest.raises(QASchemaError, match="Each note must be a str"):
            InventorySummary.from_dict(d)

    def test_notes_non_list_raises(self) -> None:
        d = _valid_dict()
        d["notes"] = "not a list"
        with pytest.raises(QASchemaError, match="must be a list of str"):
            InventorySummary.from_dict(d)

    def test_notes_backfilled_when_absent(self) -> None:
        d = _valid_dict()
        del d["notes"]
        obj = InventorySummary.from_dict(d)
        assert obj.notes == ()

    def test_extra_keys_ignored(self) -> None:
        d = _valid_dict()
        d["extra_field"] = "should be ignored"
        d["another"] = 42
        obj = InventorySummary.from_dict(d)
        assert obj.sources == d["sources"]

    def test_empty_count_dicts_accepted(self) -> None:
        d = _valid_dict()
        d["sources"] = {}
        d["adapter_status_counts"] = {}
        d["adapter_flag_counts"] = {}
        d["pair_risk_counts"] = {}
        d["recommended_strategy_counts"] = {}
        d["dominant_issue_counts"] = {}
        d["strict_qa_block_candidates"] = 0
        obj = InventorySummary.from_dict(d)
        assert obj.sources == {}
        assert obj.strict_qa_block_candidates == 0
