"""
Tests for peft_utils normalization and pattern-completion functions.
"""

import pytest

from gradience.peft_utils import (
    create_complete_alpha_pattern,
    create_complete_rank_pattern,
    normalize_alpha_pattern,
    normalize_peft_module_name,
    normalize_rank_pattern,
)

# ---------------------------------------------------------------------------
# normalize_peft_module_name
# ---------------------------------------------------------------------------


class TestNormalizePeftModuleName:
    """Group of tests kept as plain functions via parametrize."""


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Strip "base_model.model." prefix
        (
            "base_model.model.bert.encoder.layer.0.attention.self.query",
            "bert.encoder.layer.0.attention.self.query",
        ),
        # Strip "base_model." prefix (but not "base_model.model.")
        (
            "base_model.transformer.h.0.attn.c_attn",
            "transformer.h.0.attn.c_attn",
        ),
        # Strip "model." prefix
        (
            "model.transformer.h.0.attn.c_attn",
            "transformer.h.0.attn.c_attn",
        ),
        # No matching prefix -- pass through unchanged
        (
            "distilbert.transformer.layer.0.attention.q_lin",
            "distilbert.transformer.layer.0.attention.q_lin",
        ),
        # Already clean name
        (
            "encoder.layer.3.output.dense",
            "encoder.layer.3.output.dense",
        ),
    ],
)
def test_normalize_peft_module_name(raw, expected):
    assert normalize_peft_module_name(raw) == expected


def test_normalize_peft_module_name_strips_longest_prefix_first():
    """'base_model.model.' is stripped before 'base_model.' is checked."""
    name = "base_model.model.layer.0.q"
    result = normalize_peft_module_name(name)
    assert result == "layer.0.q"
    # Confirm that 'base_model.' alone would *not* be re-applied
    assert not result.startswith("model.")


def test_normalize_peft_module_name_empty_after_strip():
    """Edge case: if the name *is* just a prefix, returns empty string."""
    # "model." stripped from "model." yields ""
    assert normalize_peft_module_name("model.") == ""


# ---------------------------------------------------------------------------
# normalize_rank_pattern
# ---------------------------------------------------------------------------


def test_normalize_rank_pattern_strips_prefixes():
    pattern = {
        "base_model.model.layer.0.q": 8,
        "base_model.model.layer.0.v": 4,
        "encoder.layer.1.dense": 16,
    }
    result = normalize_rank_pattern(pattern)

    assert result == {
        "layer.0.q": 8,
        "layer.0.v": 4,
        "encoder.layer.1.dense": 16,
    }


def test_normalize_rank_pattern_empty_dict():
    assert normalize_rank_pattern({}) == {}


def test_normalize_rank_pattern_preserves_values():
    pattern = {"model.a": 2, "model.b": 64}
    result = normalize_rank_pattern(pattern)
    assert result["a"] == 2
    assert result["b"] == 64


# ---------------------------------------------------------------------------
# normalize_alpha_pattern
# ---------------------------------------------------------------------------


def test_normalize_alpha_pattern_strips_prefixes():
    pattern = {
        "base_model.model.layer.0.q": 16,
        "model.layer.1.k": 32,
    }
    result = normalize_alpha_pattern(pattern)

    assert result == {
        "layer.0.q": 16,
        "layer.1.k": 32,
    }


def test_normalize_alpha_pattern_empty_dict():
    assert normalize_alpha_pattern({}) == {}


# ---------------------------------------------------------------------------
# create_complete_rank_pattern
# ---------------------------------------------------------------------------


def test_create_complete_rank_pattern_fills_defaults():
    """Modules absent from partial_rank_pattern get default_rank."""
    partial = {"layer.0.q": 8}
    audit_layers = [
        {"name": "model.layer.0.q"},
        {"name": "model.layer.0.v"},
        {"name": "model.layer.1.q"},
    ]
    result = create_complete_rank_pattern(partial, audit_layers, default_rank=4)

    # layer.0.q was specified => 8; others get default 4
    assert result == {
        "layer.0.q": 8,
        "layer.0.v": 4,
        "layer.1.q": 4,
    }


def test_create_complete_rank_pattern_normalizes_partial_and_audit():
    """Both the partial pattern keys and audit layer names are normalized."""
    partial = {"base_model.model.layer.0.q": 16}
    audit_layers = [
        {"name": "base_model.model.layer.0.q"},
        {"name": "base_model.model.layer.0.v"},
    ]
    result = create_complete_rank_pattern(partial, audit_layers, default_rank=4)

    assert result == {
        "layer.0.q": 16,
        "layer.0.v": 4,
    }


def test_create_complete_rank_pattern_empty_partial():
    """When partial is empty, every module gets the default rank."""
    audit_layers = [
        {"name": "layer.0.q"},
        {"name": "layer.0.v"},
    ]
    result = create_complete_rank_pattern({}, audit_layers, default_rank=8)
    assert all(rank == 8 for rank in result.values())
    assert len(result) == 2


def test_create_complete_rank_pattern_all_specified():
    """When partial covers every module, no defaults are used."""
    partial = {"layer.0.q": 8, "layer.0.v": 16}
    audit_layers = [
        {"name": "layer.0.q"},
        {"name": "layer.0.v"},
    ]
    result = create_complete_rank_pattern(partial, audit_layers, default_rank=4)
    assert result == {"layer.0.q": 8, "layer.0.v": 16}


# ---------------------------------------------------------------------------
# create_complete_alpha_pattern
# ---------------------------------------------------------------------------


def test_create_complete_alpha_pattern_fills_defaults():
    """Modules absent from partial_alpha_pattern get default_alpha."""
    partial = {"layer.0.q": 16}
    audit_layers = [
        {"name": "model.layer.0.q"},
        {"name": "model.layer.0.v"},
        {"name": "model.layer.1.q"},
    ]
    result = create_complete_alpha_pattern(partial, audit_layers, default_alpha=8)

    assert result == {
        "layer.0.q": 16,
        "layer.0.v": 8,
        "layer.1.q": 8,
    }


def test_create_complete_alpha_pattern_normalizes_partial_and_audit():
    """Both the partial pattern keys and audit layer names are normalized."""
    partial = {"base_model.model.layer.0.q": 32}
    audit_layers = [
        {"name": "base_model.model.layer.0.q"},
        {"name": "base_model.model.layer.0.v"},
    ]
    result = create_complete_alpha_pattern(partial, audit_layers, default_alpha=16)

    assert result == {
        "layer.0.q": 32,
        "layer.0.v": 16,
    }


def test_create_complete_alpha_pattern_empty_partial():
    """When partial is empty, every module gets the default alpha."""
    audit_layers = [
        {"name": "layer.0.q"},
        {"name": "layer.0.v"},
    ]
    result = create_complete_alpha_pattern({}, audit_layers, default_alpha=32)
    assert all(alpha == 32 for alpha in result.values())
    assert len(result) == 2


def test_create_complete_alpha_pattern_all_specified():
    """When partial covers every module, no defaults are used."""
    partial = {"layer.0.q": 8, "layer.0.v": 16}
    audit_layers = [
        {"name": "layer.0.q"},
        {"name": "layer.0.v"},
    ]
    result = create_complete_alpha_pattern(partial, audit_layers, default_alpha=4)
    assert result == {"layer.0.q": 8, "layer.0.v": 16}
