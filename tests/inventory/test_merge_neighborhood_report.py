"""Tests for merge neighborhood artifact schema validation."""

from __future__ import annotations

import glob
import json
import tempfile
from pathlib import Path

import pytest

from gradience.exceptions import QASchemaError
from gradience.vnext.inventory.merge_neighborhood_report import (
    CHARACTERIZATION_VALUES,
    SCHEMA_ID,
    MergeNeighborhoodReport,
)


def _valid_dict() -> dict:
    return {
        "schema": SCHEMA_ID,
        "groups": [
            {
                "group_id": "cluster_01",
                "members": ["adapter_a", "adapter_b"],
                "characterization": "likely-safe neighborhood",
                "common_strategy": "linear",
                "dominant_issue": "none",
            }
        ],
        "excluded": [
            {"adapter": "adapter_c", "reason": "flagged_weak"},
        ],
        "boundary_warnings": [
            {"between": ["cluster_01", "cluster_02"], "reason": "high cross-group merge risk"},
        ],
    }


class TestMergeNeighborhoodReportValidation:
    def test_roundtrip(self) -> None:
        d = _valid_dict()
        report = MergeNeighborhoodReport.from_dict(d)
        assert report.groups[0].group_id == "cluster_01"
        assert report.excluded[0].reason == "flagged_weak"
        assert report.boundary_warnings[0].between == ("cluster_01", "cluster_02")

        report2 = MergeNeighborhoodReport.from_dict(report.to_dict())
        assert report2 == report

    def test_to_json_roundtrip(self) -> None:
        report = MergeNeighborhoodReport.from_dict(_valid_dict())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "neighborhoods.json"
            report.to_json(path)
            with open(path) as f:
                loaded = json.load(f)
        reloaded = MergeNeighborhoodReport.from_dict(loaded)
        assert reloaded == report

    def test_missing_schema(self) -> None:
        d = _valid_dict()
        del d["schema"]
        with pytest.raises(QASchemaError, match="Missing required field: schema"):
            MergeNeighborhoodReport.from_dict(d)

    def test_wrong_schema(self) -> None:
        d = _valid_dict()
        d["schema"] = "gradience.merge_neighborhoods/v99"
        with pytest.raises(QASchemaError, match="Expected schema"):
            MergeNeighborhoodReport.from_dict(d)

    def test_missing_groups_section(self) -> None:
        d = _valid_dict()
        del d["groups"]
        with pytest.raises(QASchemaError, match="Missing required section: groups"):
            MergeNeighborhoodReport.from_dict(d)

    def test_invalid_characterization(self) -> None:
        d = _valid_dict()
        d["groups"][0]["characterization"] = "graph-cluster"
        with pytest.raises(QASchemaError, match="characterization"):
            MergeNeighborhoodReport.from_dict(d)

    def test_boundary_between_must_have_two_entries(self) -> None:
        d = _valid_dict()
        d["boundary_warnings"][0]["between"] = ["cluster_01"]
        with pytest.raises(QASchemaError, match="exactly two"):
            MergeNeighborhoodReport.from_dict(d)

    def test_characterization_vocabulary_frozen(self) -> None:
        assert CHARACTERIZATION_VALUES == {
            "likely-safe neighborhood",
            "caution neighborhood",
            "audit-aware neighborhood",
        }

    @pytest.mark.parametrize("path", sorted(glob.glob("examples/neighborhoods/*.json")))
    def test_example_file_loads(self, path: str) -> None:
        with open(Path(path)) as f:
            d = json.load(f)
        report = MergeNeighborhoodReport.from_dict(d)
        assert report.groups is not None
