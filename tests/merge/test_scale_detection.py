"""Tests for gradience.vnext.merge.scale_detection — heuristic scale classification."""

from __future__ import annotations

import pytest

from gradience.vnext.merge.scale_detection import DetectedScale, detect_scale, needs_risk_provenance_warning


class TestDetectScale:
    """Test scale classification from base-model identifiers."""

    @pytest.mark.parametrize(
        "base_model",
        [
            "distilbert-base-uncased",
            "bert-base-uncased",
            "bert-large-cased",
            "roberta-base",
            "microsoft/deberta-v3-base",
            "albert-base-v2",
            "google/electra-small",
            "xlm-roberta-base",
        ],
    )
    def test_encoder_models_detected(self, base_model: str):
        assert detect_scale(base_model) == DetectedScale.ENCODER

    @pytest.mark.parametrize(
        "base_model",
        [
            "mistralai/Mistral-7B-v0.3",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-3-8B",
            "meta-llama/Llama-3.1-8B",
            "microsoft/phi-2",
            "microsoft/phi-3-mini",
            "google/gemma-7b",
            "google/gemma-2b",
            "Qwen/Qwen-7B",
            "tiiuae/falcon-7b",
            "01-ai/Yi-6B",
        ],
    )
    def test_decoder_7b_detected(self, base_model: str):
        assert detect_scale(base_model) == DetectedScale.DECODER_7B

    @pytest.mark.parametrize(
        "base_model",
        [
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Llama-3-70B",
            "meta-llama/Llama-2-13b-hf",
            "mistralai/Mixtral-8x7B-v0.1",
            "tiiuae/falcon-40b",
            "tiiuae/falcon-180b",
            "Qwen/Qwen-14B",
            "Qwen/Qwen-72B",
            "01-ai/Yi-34B",
        ],
    )
    def test_decoder_large_detected(self, base_model: str):
        assert detect_scale(base_model) == DetectedScale.DECODER_LARGE

    @pytest.mark.parametrize(
        "base_model",
        [
            "some-custom-model-v1",
            "my-org/proprietary-model",
            "",
        ],
    )
    def test_unknown_defaults_to_unknown(self, base_model: str):
        assert detect_scale(base_model) == DetectedScale.UNKNOWN


class TestNeedsRiskProvenanceWarning:
    """Test the provenance warning decision function."""

    @pytest.mark.parametrize(
        "base_model",
        [
            "distilbert-base-uncased",
            "microsoft/deberta-v3-base",
            "roberta-base",
        ],
    )
    def test_needs_warning_encoder_false(self, base_model: str):
        assert needs_risk_provenance_warning(base_model) is False

    @pytest.mark.parametrize(
        "base_model",
        [
            "mistralai/Mistral-7B-v0.3",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-70b-hf",
        ],
    )
    def test_needs_warning_decoder_true(self, base_model: str):
        assert needs_risk_provenance_warning(base_model) is True

    @pytest.mark.parametrize(
        "base_model",
        [
            "some-custom-model-v1",
            "my-org/proprietary-model",
        ],
    )
    def test_needs_warning_unknown_true(self, base_model: str):
        """Unknown models trigger warning (conservative default)."""
        assert needs_risk_provenance_warning(base_model) is True
