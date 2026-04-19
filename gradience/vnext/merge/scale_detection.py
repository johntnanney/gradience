"""
Heuristic scale detection from base-model identifiers.

This module classifies adapters as encoder-scale or decoder-scale based
on the base_model string in adapter metadata.  The classification drives
provenance warnings in the merge pipeline.

The heuristic is intentionally conservative: unknown models are classified
as UNKNOWN, which triggers the decoder-scale warning (erring on the side
of honesty).
"""

from __future__ import annotations

from enum import Enum


class DetectedScale(str, Enum):
    """Architecture scale category detected from base-model identifier."""

    ENCODER = "encoder"
    DECODER_7B = "decoder_7b"
    DECODER_LARGE = "decoder_large"
    UNKNOWN = "unknown"


# Known encoder-scale model families
_ENCODER_PATTERNS = [
    "distilbert",
    "bert-base",
    "bert-large",
    "roberta",
    "deberta",
    "albert",
    "electra",
    "xlm-roberta",
]

# Known decoder-scale model families (7B class)
_DECODER_7B_PATTERNS = [
    "mistral-7b",
    "llama-2-7b",
    "llama-3-8b",
    "llama-3.1-8b",
    "phi-2",
    "phi-3",
    "gemma-7b",
    "gemma-2b",
    "qwen-7b",
    "falcon-7b",
    "yi-6b",
]

# Known large decoder models
_DECODER_LARGE_PATTERNS = [
    "llama-2-13b",
    "llama-2-70b",
    "llama-3-70b",
    "mistral-8x7b",
    "mixtral",
    "falcon-40b",
    "falcon-180b",
    "qwen-14b",
    "qwen-72b",
    "yi-34b",
]


def detect_scale(base_model: str) -> DetectedScale:
    """Classify base model into scale category.

    Parameters
    ----------
    base_model : str
        The base_model identifier from adapter config (e.g.,
        ``"mistralai/Mistral-7B-v0.3"``).

    Returns
    -------
    DetectedScale
        The detected scale.  UNKNOWN triggers the same warnings as
        decoder-scale (conservative default).
    """
    lower = base_model.lower()
    for pat in _ENCODER_PATTERNS:
        if pat in lower:
            return DetectedScale.ENCODER
    for pat in _DECODER_LARGE_PATTERNS:
        if pat in lower:
            return DetectedScale.DECODER_LARGE
    for pat in _DECODER_7B_PATTERNS:
        if pat in lower:
            return DetectedScale.DECODER_7B
    return DetectedScale.UNKNOWN


def needs_risk_provenance_warning(base_model: str) -> bool:
    """Return True if risk predictions should carry a provenance warning.

    Currently returns True for everything except known encoder-scale models,
    because risk prediction has only been validated at encoder scale.
    """
    return detect_scale(base_model) != DetectedScale.ENCODER
