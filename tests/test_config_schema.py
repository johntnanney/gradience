"""Tests for gradience.bench.config_schema validation module."""

from __future__ import annotations

import warnings

import pytest

from gradience.bench.config_schema import _normalize_train_config, validate_config


def _minimal_valid_config():
    """Return a minimal config dict that passes validation."""
    return {
        "model": {"name": "distilbert-base-uncased"},
        "task": {"dataset": "sst2"},
        "lora": {"r": 8},
    }


# ---- validate_config: happy path ----


def test_valid_config_passes():
    """Minimal valid config passes through and returns a dict."""
    cfg = _minimal_valid_config()
    result = validate_config(cfg)
    assert result["model"]["name"] == "distilbert-base-uncased"
    assert result["task"]["dataset"] == "sst2"
    assert result["lora"] == {"r": 8}


# ---- validate_config: missing required keys ----


def test_missing_model_raises():
    """No model key raises ValueError."""
    cfg = _minimal_valid_config()
    del cfg["model"]
    with pytest.raises(ValueError, match="model"):
        validate_config(cfg)


def test_missing_model_name_raises():
    """model key present but no name raises ValueError."""
    cfg = _minimal_valid_config()
    cfg["model"] = {"type": "causal_lm"}
    with pytest.raises(ValueError, match="model.name"):
        validate_config(cfg)


def test_missing_task_raises():
    """No task key raises ValueError."""
    cfg = _minimal_valid_config()
    del cfg["task"]
    with pytest.raises(ValueError, match="task"):
        validate_config(cfg)


def test_missing_task_dataset_raises():
    """task key present but no dataset raises ValueError."""
    cfg = _minimal_valid_config()
    cfg["task"] = {"split": "train"}
    with pytest.raises(ValueError, match="task.dataset"):
        validate_config(cfg)


def test_missing_lora_raises():
    """No lora key raises ValueError."""
    cfg = _minimal_valid_config()
    del cfg["lora"]
    with pytest.raises(ValueError, match="lora"):
        validate_config(cfg)


def test_multiple_errors_reported():
    """Missing model AND task AND lora produces a single ValueError with all errors."""
    cfg = {}
    with pytest.raises(ValueError, match="model") as exc_info:
        validate_config(cfg)
    msg = str(exc_info.value)
    assert "model" in msg
    assert "task" in msg
    assert "lora" in msg


# ---- _normalize_train_config / train section handling ----


def test_lr_normalized_to_learning_rate():
    """train.lr gets renamed to train.learning_rate with DeprecationWarning."""
    cfg = _minimal_valid_config()
    cfg["train"] = {"lr": 1e-4, "epochs": 3}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = validate_config(cfg)

    assert "learning_rate" in result["train"]
    assert result["train"]["learning_rate"] == 1e-4
    assert "lr" not in result["train"]
    # Other keys preserved
    assert result["train"]["epochs"] == 3

    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 1
    assert "deprecated" in str(deprecation_warnings[0].message).lower()


def test_both_lr_and_learning_rate_keeps_learning_rate():
    """Both present keeps learning_rate, emits UserWarning, removes lr."""
    cfg = _minimal_valid_config()
    cfg["train"] = {"lr": 1e-4, "learning_rate": 5e-5}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = validate_config(cfg)

    assert result["train"]["learning_rate"] == 5e-5
    assert "lr" not in result["train"]

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "learning_rate" in str(user_warnings[0].message)


def test_no_train_section_passes():
    """Config without train section passes (train is optional)."""
    cfg = _minimal_valid_config()
    assert "train" not in cfg
    result = validate_config(cfg)
    assert "train" not in result


def test_config_without_lr_unchanged():
    """Config with learning_rate (not lr) passes without any warnings."""
    cfg = _minimal_valid_config()
    cfg["train"] = {"learning_rate": 3e-5}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = validate_config(cfg)

    assert result["train"]["learning_rate"] == 3e-5
    assert len(caught) == 0
