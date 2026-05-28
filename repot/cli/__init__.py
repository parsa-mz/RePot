"""``repot`` CLI — entry point.

Single console script registered via ``[project.scripts]`` in pyproject.toml::

    repot = "repot.cli:main"

Run ``repot --help`` for the full subcommand list.
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.traceback import install as install_traceback

from repot.cli import benchmark, derail, judge, run

_CONSOLE = Console()


def _load_dotenv_silent() -> None:
    """Load `.env` from the cwd if `python-dotenv` is installed; no-op otherwise."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def main(argv: list[str] | None = None) -> int:
    """Argparse-based CLI entry point. Returns an exit code."""
    _load_dotenv_silent()
    install_traceback(show_locals=False, console=_CONSOLE)

    parser = argparse.ArgumentParser(
        prog="repot",
        description="RePoT — Recoverable Program-of-Thought.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    run.add_parser(subparsers)
    derail.add_parser(subparsers)
    judge.add_parser(subparsers)
    benchmark.add_parser(subparsers)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
