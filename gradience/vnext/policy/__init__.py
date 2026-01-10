"""Gradience vNext policy package.

This is where *interpretation* lives (separate from measurement).

For v0, we start with a simple :func:`~gradience.vnext.policy.check.check_config`
validator that:

* consumes a :class:`~gradience.vnext.types.ConfigSnapshot`
* emits a list of :class:`~gradience.vnext.types.Recommendation`

Policies should remain:

* **cheap** (no model weights required),
* **honest** (heuristics with scope + confidence), and
* **composable** (check / audit / monitor can share the same types).
"""

from .check import check_config, check_run

__all__ = ["check_config", "check_run"]
