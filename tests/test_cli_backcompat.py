"""Compatibility checks for legacy helper imports from ``gradience.cli``."""

from __future__ import annotations

from gradience.cli import (
    _analyze_policy_disagreements,
    _load_source_qa,
    _print_policy_disagreement_summary,
)
from gradience.cli_format import _print_policy_disagreement_summary as cli_format_print
from gradience.cli_utils import (
    _analyze_policy_disagreements as cli_utils_analyze,
)
from gradience.cli_utils import (
    _load_source_qa as cli_utils_load_source_qa,
)


def test_cli_reexports_legacy_analysis_helpers():
    assert _analyze_policy_disagreements is cli_utils_analyze
    assert _print_policy_disagreement_summary is cli_format_print
    assert _load_source_qa is cli_utils_load_source_qa
