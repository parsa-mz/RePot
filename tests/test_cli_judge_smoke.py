"""End-to-end smoke test for `repot judge` over a hand-crafted trace JSONL."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _repot_cmd() -> list[str]:
    script = shutil.which("repot")
    if script:
        return [script]
    candidate = Path(sys.executable).with_name("repot")
    if candidate.exists():
        return [str(candidate)]
    pytest.skip("`repot` console script not on PATH; install with `pip install -e .`")


def _write_trace(path: Path) -> None:
    """Write a minimal but realistic trace: 2 methods x 2 envs, mixed outcomes."""
    rows = [
        {"model": "m", "method": "program_of_thought", "environment": "tower_of_hanoi",
         "success": True, "tokens_in": 100, "tokens_out": 50,
         "stopped_reason": "goal"},
        {"model": "m", "method": "program_of_thought", "environment": "blocksworld",
         "success": False, "tokens_in": 120, "tokens_out": 70,
         "stopped_reason": "invalid_transition"},
        {"model": "m", "method": "repot", "environment": "tower_of_hanoi",
         "success": True, "tokens_in": 130, "tokens_out": 55,
         "stopped_reason": "goal"},
        {"model": "m", "method": "repot", "environment": "blocksworld",
         "success": True, "tokens_in": 140, "tokens_out": 60,
         "stopped_reason": "goal"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_judge_emits_summary_with_aggregates(tmp_path: Path) -> None:
    """`repot judge --print` aggregates a small trace and writes a summary JSON.

    Confirms the judge CLI loads JSONL, computes per-method and
    per-(method,environment) aggregates, captures failure modes, and writes
    the default ``<traces>.summary.json`` companion file.
    """
    traces = tmp_path / "traces.jsonl"
    _write_trace(traces)

    cmd = _repot_cmd() + ["judge", str(traces), "--print"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, (
        f"repot judge failed (rc={result.returncode})\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    summary_path = traces.with_suffix(".summary.json")
    assert summary_path.exists(), f"expected {summary_path} to be written"
    summary = json.loads(summary_path.read_text())

    assert summary["total_records"] == 4
    assert sorted(summary["methods"]) == ["program_of_thought", "repot"]
    assert summary["models"] == ["m"]

    by_method = {(r["model"], r["method"]): r for r in summary["by_method"]}
    pot = by_method[("m", "program_of_thought")]
    repot = by_method[("m", "repot")]
    assert pot["n"] == 2 and pot["n_success"] == 1
    assert pot["success_rate"] == 0.5
    assert repot["n"] == 2 and repot["n_success"] == 2
    assert repot["success_rate"] == 1.0

    by_triple = {(r["model"], r["method"], r["environment"]): r
                 for r in summary["by_method_environment"]}
    assert ("m", "program_of_thought", "tower_of_hanoi") in by_triple
    assert ("m", "program_of_thought", "blocksworld") in by_triple
    assert ("m", "repot", "tower_of_hanoi") in by_triple
    assert ("m", "repot", "blocksworld") in by_triple

    assert summary["failure_modes"] == {"invalid_transition": 1}
