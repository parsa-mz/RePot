"""``repot judge`` — extract metrics from a trace JSONL and save them.

Reads a ``repot run`` trace file (one JSON record per line), aggregates
success rate / token cost / failure modes by (model, method, environment),
and writes a compact summary JSON.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.table import Table

_CONSOLE = Console()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `judge` subcommand."""
    parser = subparsers.add_parser(
        "judge",
        help="Extract metrics from a trace JSONL and save a summary JSON.",
        description=(
            "Read a `repot run` trace file, aggregate success rate / tokens / "
            "failure modes, and write a compact summary."
        ),
    )
    parser.add_argument(
        "traces",
        help="Path to a trace JSONL file (or a directory of trace JSONL files).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output summary JSON path. Defaults to <traces>.summary.json.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Also print a per-method success-rate table to stdout.",
    )
    parser.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    traces_path = Path(args.traces)
    records = _load(traces_path)
    if not records:
        _CONSOLE.print(f"[red]No records found at {traces_path}[/red]")
        return 1

    summary = _summarize(records)

    out = Path(args.output or _default_summary_path(traces_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _CONSOLE.print(f"[green]Wrote summary[/green] {out}  [dim]({len(records)} records)[/dim]")

    if args.print:
        _print_method_table(summary)
    return 0


# ---------------------------------------------------------------------------


def _load(path: Path) -> list[dict]:
    """Read a JSONL file or every ``*.jsonl`` under a directory."""
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            return []
    else:
        files = [path]
    rows: list[dict] = []
    for f in files:
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _summarize(records: list[dict]) -> dict:
    """Aggregate by (model, method) and (model, method, environment)."""
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_triple: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in records:
        model = str(row.get("model", "unknown"))
        method = str(row.get("method", "unknown"))
        env = str(row.get("environment", "unknown"))
        by_pair[(model, method)].append(row)
        by_triple[(model, method, env)].append(row)

    methods = sorted(
        {(m, meth): None for (m, meth) in by_pair}.keys(),
        key=lambda x: (x[0], x[1]),
    )

    return {
        "total_records": len(records),
        "models": sorted({m for m, _ in by_pair}),
        "methods": sorted({meth for _, meth in by_pair}),
        "by_method": [
            {
                "model": model,
                "method": method,
                **_metrics(by_pair[(model, method)]),
            }
            for model, method in methods
        ],
        "by_method_environment": [
            {
                "model": model,
                "method": method,
                "environment": env,
                **_metrics(rows),
            }
            for (model, method, env), rows in sorted(by_triple.items())
        ],
        "failure_modes": dict(
            Counter(
                row.get("stopped_reason", "unknown")
                for row in records
                if not _is_success(row)
            )
        ),
    }


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    successes = sum(_is_success(row) for row in rows)
    tokens_in = [int(r.get("tokens_in", 0) or 0) for r in rows]
    tokens_out = [int(r.get("tokens_out", 0) or 0) for r in rows]
    return {
        "n": n,
        "n_success": successes,
        "success_rate": round(successes / n, 4) if n else 0.0,
        "tokens_in_mean": round(mean(tokens_in), 1) if tokens_in else 0.0,
        "tokens_out_mean": round(mean(tokens_out), 1) if tokens_out else 0.0,
    }


def _is_success(row: dict) -> bool:
    val = row.get("success", False)
    return bool(val) if val is not None else False


def _default_summary_path(traces: Path) -> Path:
    if traces.is_dir():
        return traces / "summary.json"
    return traces.with_suffix(".summary.json")


def _print_method_table(summary: dict) -> None:
    table = Table(title="Per-method success rate")
    table.add_column("Model")
    table.add_column("Method")
    table.add_column("N", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Rate", justify="right")
    for row in summary["by_method"]:
        table.add_row(
            row["model"],
            row["method"],
            str(row["n"]),
            str(row["n_success"]),
            f"{row['success_rate']*100:.1f}%",
        )
    _CONSOLE.print(table)
