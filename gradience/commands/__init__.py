"""Gradience CLI command handlers.

Each module contains one logical command group with its handler(s),
argparse setup function(s), and any command-specific helpers.
"""

from gradience.commands.audit import cmd_audit, cmd_audit_adapter, setup_audit_commands
from gradience.commands.check import cmd_check, setup_check_commands
from gradience.commands.inventory import (
    cmd_batch_summary,
    cmd_portfolio,
    cmd_preflight_report,
    cmd_suggest_neighborhoods,
    cmd_summarize_inventory,
    setup_inventory_commands,
)
from gradience.commands.merge import (
    cmd_explain,
    cmd_merge,
    cmd_merge_audit,
    cmd_merge_plan,
    setup_merge_commands,
)
from gradience.commands.monitor import cmd_monitor, cmd_report, setup_monitor_commands
from gradience.commands.truncate import cmd_truncate, setup_truncate_commands
from gradience.commands.verify import cmd_verify, setup_verify_commands

__all__ = [
    # Command handlers
    "cmd_audit",
    "cmd_audit_adapter",
    "cmd_batch_summary",
    "cmd_check",
    "cmd_explain",
    "cmd_merge",
    "cmd_merge_audit",
    "cmd_merge_plan",
    "cmd_monitor",
    "cmd_portfolio",
    "cmd_preflight_report",
    "cmd_report",
    "cmd_suggest_neighborhoods",
    "cmd_summarize_inventory",
    "cmd_truncate",
    "cmd_verify",
    # Setup functions
    "setup_audit_commands",
    "setup_check_commands",
    "setup_inventory_commands",
    "setup_merge_commands",
    "setup_monitor_commands",
    "setup_truncate_commands",
    "setup_verify_commands",
]
