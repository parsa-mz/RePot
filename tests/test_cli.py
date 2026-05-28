"""CLI smoke tests — exercises argparse + import wiring without external APIs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from repot.cli import main


def test_help_renders(capsys):
    """`repot --help` prints help and exits 0."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "RePoT" in out
    assert "run" in out
    assert "derail" in out
    assert "judge" in out


@pytest.mark.parametrize("subcommand", ["run", "derail", "judge"])
def test_subcommand_help_renders(capsys, subcommand):
    """Each subcommand's `--help` works without imports failing."""
    with pytest.raises(SystemExit) as excinfo:
        main([subcommand, "--help"])
    assert excinfo.value.code == 0


def test_no_args_prints_help_and_exits_zero(capsys):
    """`repot` with no args prints help and exits 0 (not an error)."""
    rc = main([])
    assert rc == 0


def test_judge_aggregates_minimal_trace(tmp_path: Path):
    """`repot judge` on a tiny synthetic JSONL emits a sane summary."""
    traces = tmp_path / "traces.jsonl"
    rows = [
        {"model": "m", "method": "pot", "environment": "hanoi", "success": True,
         "tokens_in": 100, "tokens_out": 50},
        {"model": "m", "method": "pot", "environment": "hanoi", "success": False,
         "tokens_in": 110, "tokens_out": 60, "stopped_reason": "invalid_action"},
        {"model": "m", "method": "repot", "environment": "hanoi", "success": True,
         "tokens_in": 120, "tokens_out": 55},
    ]
    traces.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    out = tmp_path / "summary.json"
    rc = main(["judge", str(traces), "--output", str(out)])
    assert rc == 0
    summary = json.loads(out.read_text())

    assert summary["total_records"] == 3
    by_method = {(r["model"], r["method"]): r for r in summary["by_method"]}
    assert by_method[("m", "pot")]["n"] == 2
    assert by_method[("m", "pot")]["n_success"] == 1
    assert by_method[("m", "pot")]["success_rate"] == 0.5
    assert by_method[("m", "repot")]["success_rate"] == 1.0
    assert summary["failure_modes"] == {"invalid_action": 1}
