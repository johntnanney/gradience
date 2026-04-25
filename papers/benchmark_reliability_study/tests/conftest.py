"""pytest configuration for the benchmark reliability study.

Makes the paper-root `gradience_study` package importable without
requiring `pip install -e .` — useful for CI and quick local iteration.
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parent.parent
if str(PAPER_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_ROOT))
