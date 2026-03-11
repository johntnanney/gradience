"""Gradience exception hierarchy.

All public Gradience exceptions inherit from :class:`GradienceError`,
allowing callers to catch all Gradience-specific errors with a single
``except GradienceError`` clause while still being able to handle
specific categories.

Backward compatibility: :class:`TelemetrySchemaError` and
:class:`TelemetryFormatError` remain importable from
``gradience.vnext.telemetry_reader`` as before.
"""

from __future__ import annotations


class GradienceError(Exception):
    """Base exception for all Gradience errors."""


class ConfigError(GradienceError, ValueError):
    """Invalid configuration (YAML parsing, missing fields, constraint violations)."""


class AuditError(GradienceError, ValueError):
    """Spectral audit failures (missing weights, incompatible shapes, SVD failures)."""


class MergeError(GradienceError, ValueError):
    """Merge audit failures (incompatible adapters, shape mismatches)."""


class QASchemaError(GradienceError, ValueError):
    """Raised when a QA artifact fails schema validation (missing fields, wrong types, unknown schema)."""


class TelemetryError(GradienceError, ValueError):
    """Telemetry read/write errors."""


class TelemetrySchemaError(TelemetryError, ValueError):
    """Raised when telemetry schema version does not match expectations."""


class TelemetryFormatError(TelemetryError, ValueError):
    """Raised when a telemetry record is missing required envelope fields."""


class DependencyError(GradienceError, RuntimeError):
    """Missing optional dependency required for the requested operation."""
