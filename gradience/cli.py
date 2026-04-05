"""Gradience CLI — argument parsing and dispatch.

Inventory preflight for LoRA adapter merging, with advanced and research utilities.

Recommended workflow:
    1) audit one adapter:
       gradience audit-adapter --peft-dir ./adapters/a --out qa/a_qa.json ...
    2) audit one pair:
       gradience merge-audit --adapter-a ./adapters/a --adapter-b ./adapters/b --source-a-qa qa/a_qa.json --source-b-qa qa/b_qa.json --emit-report reports/a_vs_b.json
    3) analyze inventory:
       gradience summarize-inventory --qa-dir qa/ --report-dir reports/ --emit-report inventory/summary.json --emit-bundle runs/run_001
"""

from __future__ import annotations

import argparse
import sys

# ---------------------------------------------------------------------------
# Backward-compatible re-exports
# ---------------------------------------------------------------------------
# These symbols are imported by tests and external code.  Re-export them from
# their new canonical locations so existing ``from gradience.cli import X``
# continues to work.
from gradience.cli_utils import _load_source_qa as _load_source_qa  # noqa: F401, E402
from gradience.commands import (
    setup_audit_commands,
    setup_check_commands,
    setup_inventory_commands,
    setup_merge_commands,
    setup_monitor_commands,
    setup_truncate_commands,
    setup_verify_commands,
)
from gradience.commands.audit import cmd_audit as cmd_audit  # noqa: F401, E402
from gradience.commands.audit import cmd_audit_adapter as cmd_audit_adapter  # noqa: F401, E402
from gradience.commands.check import cmd_check as cmd_check  # noqa: F401, E402
from gradience.commands.inventory import cmd_batch_summary as cmd_batch_summary  # noqa: F401, E402
from gradience.commands.inventory import cmd_portfolio as cmd_portfolio  # noqa: F401, E402
from gradience.commands.inventory import cmd_preflight_report as cmd_preflight_report  # noqa: F401, E402
from gradience.commands.inventory import cmd_suggest_neighborhoods as cmd_suggest_neighborhoods  # noqa: F401, E402
from gradience.commands.inventory import cmd_summarize_inventory as cmd_summarize_inventory  # noqa: F401, E402
from gradience.commands.merge import cmd_explain as cmd_explain  # noqa: F401, E402
from gradience.commands.merge import cmd_merge as cmd_merge  # noqa: F401, E402
from gradience.commands.merge import cmd_merge_audit as cmd_merge_audit  # noqa: F401, E402
from gradience.commands.merge import cmd_merge_plan as cmd_merge_plan  # noqa: F401, E402
from gradience.commands.monitor import cmd_monitor as cmd_monitor  # noqa: F401, E402
from gradience.commands.monitor import cmd_report as cmd_report  # noqa: F401, E402
from gradience.commands.truncate import cmd_truncate as cmd_truncate  # noqa: F401, E402
from gradience.commands.verify import cmd_verify as cmd_verify  # noqa: F401, E402
from gradience.exceptions import GradienceError

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gradience",
        description="Inventory preflight for LoRA adapter merging: QA gate -> pair audit -> inventory action plan.",
        epilog=(
            "Recommended front door:\n"
            "  1) gradience audit-adapter ...\n"
            "  2) gradience merge-audit ...\n"
            "  3) gradience summarize-inventory ...\n"
            "Optional: gradience preflight-report --bundle-dir runs/run_001\n"
            "\n"
            "Command labels:\n"
            "  [RECOMMENDED] validated product workflow\n"
            "  [ADVANCED] optional power-user analysis\n"
            "  [EXPERIMENTAL] research/internal diagnostics"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        help="Command to run (recommended: audit-adapter -> merge-audit -> summarize-inventory)",
    )

    # Recommended product workflow (front door)
    setup_audit_commands(subparsers)
    setup_merge_commands(subparsers)
    setup_inventory_commands(subparsers)

    # Advanced / experimental commands
    setup_check_commands(subparsers)
    setup_verify_commands(subparsers)
    setup_monitor_commands(subparsers)
    setup_truncate_commands(subparsers)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except SystemExit:
        raise  # batch-summary uses sys.exit(0) for empty results
    except GradienceError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
