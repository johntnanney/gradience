"""Gradience package entrypoint.

Enables: `python -m gradience ...` to behave the same as the `gradience` CLI.
"""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":
    main()
