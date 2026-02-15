"""
Named constants for the bench protocol.

Centralises thresholds, defaults, and configuration values that were
previously scattered as magic numbers across protocol.py, compression.py,
reporting.py, metadata.py, and _util.py.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Smoke-test defaults
# ---------------------------------------------------------------------------
SMOKE_TRAIN_SAMPLES = 200
SMOKE_EVAL_SAMPLES = 200
SMOKE_MAX_POST_TUNE_STEPS = 20

# ---------------------------------------------------------------------------
# Post-tuning defaults (tiny-tune after SVD truncation)
# ---------------------------------------------------------------------------
DEFAULT_POST_TUNE_STEPS = 100
DEFAULT_POST_TUNE_LR_SCALE = 0.1
DEFAULT_BASE_LR = 5e-5
DEFAULT_TRAIN_BATCH_SIZE = 8
DEFAULT_EVAL_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Model / adapter defaults
# ---------------------------------------------------------------------------
DEFAULT_NUM_LABELS = 2        # Binary classification
DEFAULT_PROBE_RANK = 16       # Fallback probe rank when not specified

# ---------------------------------------------------------------------------
# Probe quality thresholds (minimum accuracy per GLUE task)
# ---------------------------------------------------------------------------
TASK_QUALITY_THRESHOLDS: dict[str, float] = {
    "sst2": 0.75,       # SST-2 sentiment
    "mnli": 0.70,       # MNLI entailment
    "qnli": 0.75,       # QNLI question entailment
    "qqp":  0.80,       # QQP paraphrase
    "rte":  0.60,       # RTE (small dataset)
    "wnli": 0.55,       # WNLI (very small)
    "cola": 0.70,       # CoLA linguistic
    "mrpc": 0.75,       # MRPC paraphrase
    "stsb": 0.80,       # STS-B regression (converted)
}
DEFAULT_TASK_THRESHOLD = 0.70  # Fallback for unknown tasks

# ---------------------------------------------------------------------------
# Compression candidate conservatism scores
# Higher = more conservative (retains more capacity)
# ---------------------------------------------------------------------------
CONSERVATISM_SCORES: dict[str, float] = {
    "erank_p90":       1.5,   # Least conservative (entropy-based)
    "knee_p90":        2.0,   # Less conservative (finds elbows)
    "per_layer":       2.0,   # Medium conservatism
    "oht_p90":         2.5,   # Medium
    "energy_p90":      3.0,   # Medium conservative
    "uniform_p90":     3.5,   # Conservative
    "energy_median":   3.5,   # More conservative than p90
    "uniform_median":  4.0,   # Very conservative
}

# ---------------------------------------------------------------------------
# Compression candidate control
# ---------------------------------------------------------------------------
DEFAULT_MAX_CANDIDATES = 4

# ---------------------------------------------------------------------------
# Seed / randomness
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42
SHUFFLE_SEED_OFFSET = 10000   # Deterministic offset for per_layer_shuffled

# ---------------------------------------------------------------------------
# Validation-level thresholds
# ---------------------------------------------------------------------------
MIN_SEEDS_CERTIFIABLE = 3
MIN_STEPS_CERTIFIABLE = 500

MIN_SEEDS_SCREENING_PLUS = 2
MIN_STEPS_SCREENING_PLUS = 200

MIN_SEEDS_SUFFICIENT_POWER = 3   # For "sufficient" statistical power label

# ---------------------------------------------------------------------------
# Multi-seed verdict thresholds
# ---------------------------------------------------------------------------
PASS_RATE_VALID = 0.8              # ≥80% seeds must pass for PASS verdict
PASS_RATE_AGGRESSIVE = 0.6         # ≥60% seeds for aggressive tier

DEFAULT_ACCURACY_TOLERANCE = 0.005  # ±0.005 accuracy tolerance

# ---------------------------------------------------------------------------
# Reporting / display thresholds
# ---------------------------------------------------------------------------
CONCENTRATION_INDEX_HIGH = 0.4      # HHI threshold: "highly concentrated"
CONCENTRATION_INDEX_MODERATE = 0.25 # HHI threshold: "moderately concentrated"

# ---------------------------------------------------------------------------
# Subprocess timeouts (seconds)
# ---------------------------------------------------------------------------
GIT_SUBPROCESS_TIMEOUT = 5

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
CONFIG_HASH_LENGTH = 16  # First N hex chars of SHA-256

# ---------------------------------------------------------------------------
# Escalation (safety fallback)
# ---------------------------------------------------------------------------
DEFAULT_CATASTROPHIC_MARGIN_MIN = 0.01     # Floor for catastrophic margin
DEFAULT_CATASTROPHIC_MARGIN_RATIO = 0.5    # Margin = max(MIN, RATIO * tolerance)
MAX_ESCALATION_ROUNDS = 1                  # At most 1 escalation round per run
