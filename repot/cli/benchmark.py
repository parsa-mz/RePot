"""``repot benchmark`` — inspect bundled benchmark datasets.

Subcommands:
- ``repot benchmark list``           — show name, file, record count for each bundled dataset.
- ``repot benchmark info <name>``    — show schema + per-environment / per-complexity breakdown.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from repot.core.utils.jsonl import read_jsonl

_CONSOLE = Console()


# (name, relative path from repo root, one-line description)
_BENCHMARKS = {
    "puzzlezoo": (
        "data/problems/puzzlezoo_775.jsonl",
        "PuzzleZoo-775: stratified puzzle benchmark across 4 environments.",
    ),
    "planbench": (
        "data/problems/planbench_blocksworld.jsonl",
        "PlanBench Blocksworld adapter (3-12 blocks).",
    ),
    "derail": (
        "data/derail/derail_550.jsonl",
        "Derail-550: mid-rollout injection case definitions.",
    ),
}


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `benchmark` subcommand."""
    parser = subparsers.add_parser(
        "benchmark",
        help="Inspect bundled benchmark datasets.",
        description="List or inspect the PuzzleZoo / PlanBench / Derail datasets shipped in data/.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override the data/ directory (defaults to ./data relative to cwd).",
    )
    sub = parser.add_subparsers(dest="benchmark_command", metavar="ACTION")

    list_p = sub.add_parser("list", help="List bundled benchmarks with record counts.")
    list_p.set_defaults(func=lambda args: _cmd_list(args))

    info_p = sub.add_parser("info", help="Show schema + breakdown for a single benchmark.")
    info_p.add_argument("name", choices=sorted(_BENCHMARKS.keys()), help="Benchmark name.")
    info_p.set_defaults(func=lambda args: _cmd_info(args))

    parser.set_defaults(func=lambda args: _cmd_default(args, parser))


def _data_root(args: argparse.Namespace) -> Path:
    if args.data_root:
        return Path(args.data_root)
    return Path.cwd() / "data"


def _resolve(args: argparse.Namespace, rel_path: str) -> Path:
    root = _data_root(args)
    candidate = root / Path(rel_path).relative_to("data")
    if candidate.exists():
        return candidate
    # Fallback: treat rel_path as relative to cwd (e.g. running from a parent dir).
    return Path.cwd() / rel_path


def _cmd_default(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if getattr(args, "benchmark_command", None) is None:
        parser.print_help()
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    table = Table(title="Bundled benchmarks", show_lines=False)
    table.add_column("Name", style="bold")
    table.add_column("File")
    table.add_column("Records", justify="right")
    table.add_column("Description")
    for name, (rel_path, desc) in sorted(_BENCHMARKS.items()):
        path = _resolve(args, rel_path)
        n = _count_records(path)
        table.add_row(name, str(path), str(n) if n is not None else "(missing)", desc)
    _CONSOLE.print(table)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    rel_path, desc = _BENCHMARKS[args.name]
    path = _resolve(args, rel_path)
    if not path.exists():
        _CONSOLE.print(f"[red]File not found:[/red] {path}")
        return 2
    records = list(read_jsonl(path))
    _CONSOLE.print(f"[bold]{args.name}[/bold] — {desc}")
    _CONSOLE.print(f"[dim]File:[/dim] {path}")
    _CONSOLE.print(f"[dim]Records:[/dim] {len(records)}")

    if not records:
        return 0

    keys = sorted(records[0].keys())
    _CONSOLE.print(f"[dim]Schema:[/dim] {', '.join(keys)}")

    if "environment" in keys:
        env_counts = Counter(r.get("environment") for r in records)
        env_table = Table(title="By environment", show_lines=False)
        env_table.add_column("Environment", style="bold")
        env_table.add_column("Count", justify="right")
        for env, n in sorted(env_counts.items()):
            env_table.add_row(str(env), str(n))
        _CONSOLE.print(env_table)

    if "complexity" in keys:
        complexity_counts = Counter(r.get("complexity") for r in records)
        cx_table = Table(title="By complexity", show_lines=False)
        cx_table.add_column("Complexity", justify="right")
        cx_table.add_column("Count", justify="right")
        for cx, n in sorted(complexity_counts.items(), key=lambda kv: (kv[0] is None, kv[0])):
            cx_table.add_row(str(cx), str(n))
        _CONSOLE.print(cx_table)

    if "injection_type" in keys:
        inj_counts = Counter(r.get("injection_type") for r in records)
        inj_table = Table(title="By injection type", show_lines=False)
        inj_table.add_column("Injection type", style="bold")
        inj_table.add_column("Count", justify="right")
        for inj, n in sorted(inj_counts.items()):
            inj_table.add_row(str(inj), str(n))
        _CONSOLE.print(inj_table)
    return 0


def _count_records(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open() as fh:
        return sum(1 for line in fh if line.strip())
