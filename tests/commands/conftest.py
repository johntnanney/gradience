"""Shared fixtures for command handler unit tests.

Re-exports merge adapter fixtures so they can be used directly in
command-level tests without duplicating the synthetic adapter builders.
"""

from tests.merge.conftest import orthogonal_pair  # noqa: F401
