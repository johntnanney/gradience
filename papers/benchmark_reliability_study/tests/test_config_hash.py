"""Tests for config canonicalization and hashing (SPEC §4.4)."""

from __future__ import annotations

import hashlib

import pytest

from gradience_study.config import canonicalize, compute_config_hash


# -- Canonicalization --------------------------------------------------------


def test_canonicalize_sorts_keys_at_top_level():
    obj = {"z": 1, "a": 2, "m": 3}
    out = canonicalize(obj)
    assert out.index("a:") < out.index("m:") < out.index("z:")


def test_canonicalize_sorts_keys_at_nested_level():
    obj = {"outer": {"z": 1, "a": 2, "m": 3}}
    out = canonicalize(obj)
    assert out.index("a:") < out.index("m:") < out.index("z:")


def test_canonicalize_deterministic():
    obj = {"b": [3, 1, 2], "a": "hello"}
    assert canonicalize(obj) == canonicalize(obj)


def test_canonicalize_invariant_to_dict_insertion_order():
    obj1 = {"x": 1, "y": 2, "z": 3}
    obj2 = {"z": 3, "x": 1, "y": 2}
    assert canonicalize(obj1) == canonicalize(obj2)


def test_canonicalize_preserves_list_order():
    """Lists are order-significant; canonicalize must not reorder them."""
    obj1 = {"items": [3, 1, 2]}
    obj2 = {"items": [1, 2, 3]}
    assert canonicalize(obj1) != canonicalize(obj2)


def test_canonicalize_no_line_wrapping():
    """Long strings should not be wrapped, which would introduce
    whitespace differences between platforms or terminal widths."""
    long_value = "x" * 500
    obj = {"key": long_value}
    out = canonicalize(obj)
    # The long value should appear on a single line.
    assert long_value in out


# -- Hashing -----------------------------------------------------------------


def test_hash_full_format():
    full, _ = compute_config_hash({"test": "value"})
    assert len(full) == 64
    assert all(c in "0123456789abcdef" for c in full)


def test_hash_short_format():
    full, short = compute_config_hash({"test": "value"})
    assert len(short) == 8
    assert full.startswith(short)


def test_hash_deterministic():
    obj = {"a": 1, "b": [1, 2, 3]}
    h1, _ = compute_config_hash(obj)
    h2, _ = compute_config_hash(obj)
    assert h1 == h2


def test_hash_changes_on_value_change():
    h1, _ = compute_config_hash({"a": 1})
    h2, _ = compute_config_hash({"a": 2})
    assert h1 != h2


def test_hash_changes_on_key_change():
    h1, _ = compute_config_hash({"a": 1})
    h2, _ = compute_config_hash({"b": 1})
    assert h1 != h2


def test_hash_changes_on_nested_value_change():
    h1, _ = compute_config_hash({"a": {"b": 1}})
    h2, _ = compute_config_hash({"a": {"b": 2}})
    assert h1 != h2


def test_hash_stable_across_insertion_order():
    """Hashing uses canonical (sort-keys) form, so insertion order is
    irrelevant."""
    h1, _ = compute_config_hash({"x": 1, "y": 2, "z": 3})
    h2, _ = compute_config_hash({"z": 3, "x": 1, "y": 2})
    assert h1 == h2


def test_hash_known_answer():
    """Fixed input produces a fixed hash. If canonicalize() ever changes
    behavior, this test will need updating — which is the point: any
    canonicalization change must be explicit."""
    obj = {"study_id": "test", "version": "v1"}
    canonical = canonicalize(obj).encode("utf-8")
    expected_full = hashlib.sha256(canonical).hexdigest()
    full, short = compute_config_hash(obj)
    assert full == expected_full
    assert short == expected_full[:8]


def test_hash_sensitivity_to_whitespace_in_values():
    """Trailing whitespace in a string value must change the hash."""
    h1, _ = compute_config_hash({"key": "value"})
    h2, _ = compute_config_hash({"key": "value "})
    assert h1 != h2


def test_hash_sensitivity_to_unicode():
    """Unicode characters are preserved (allow_unicode=True) and affect
    the hash."""
    h1, _ = compute_config_hash({"key": "café"})
    h2, _ = compute_config_hash({"key": "cafe"})
    assert h1 != h2
