"""Verify command — installation verification."""

from __future__ import annotations

import argparse


def cmd_verify(args: argparse.Namespace) -> None:
    """Run installation verification."""
    from gradience.verify import main as verify_main

    verify_main()


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------



def _setup_verify_command(subparsers):
    verify_parser = subparsers.add_parser("verify", help="[ADVANCED] Verify installation")
    verify_parser.set_defaults(func=cmd_verify)



def setup_verify_commands(subparsers) -> None:
    """Register verify commands with the argument parser."""
    _setup_verify_command(subparsers)
