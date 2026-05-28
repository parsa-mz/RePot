"""End-to-end smoke test for `repot run` with the offline local_dummy client."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _repot_cmd() -> list[str]:
    """Resolve the installed `repot` console script.

    Prefer the console script (matches what real users invoke); fall back to
    the venv's `repot` next to ``sys.executable`` so the test still works
    inside an unactivated virtualenv.
    """
    script = shutil.which("repot")
    if script:
        return [script]
    candidate = Path(sys.executable).with_name("repot")
    if candidate.exists():
        return [str(candidate)]
    pytest.skip("`repot` console script not on PATH; install with `pip install -e .`")


def test_run_local_dummy_emits_valid_trace(tmp_path: Path) -> None:
    """`repot run --model local_dummy` produces a well-formed JSONL trace.

    Exercises the full CLI -> argparse -> runner -> trace-writer path with
    no network calls (the local_dummy client returns canned oracle output).
    """
    traces = tmp_path / "smoke.jsonl"
    cmd = _repot_cmd() + [
        "run",
        "--model", "local_dummy",
        "--max-items", "2",
        "--traces", str(traces),
        "--methods-only", "program_of_thought,repot",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, (
        f"repot run failed (rc={result.returncode})\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert traces.exists(), "trace file was not created"
    raw = traces.read_text().strip()
    assert raw, "trace file is empty"

    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert records, "no records parsed from trace file"

    required_fields = {"model", "method", "environment", "problem_id", "success",
                       "tokens_in", "tokens_out"}
    for rec in records:
        missing = required_fields - rec.keys()
        assert not missing, f"record missing fields {missing}: {rec}"
        assert rec["model"] == "local_dummy"
        assert isinstance(rec["success"], bool)
        assert isinstance(rec["tokens_in"], int)
        assert isinstance(rec["tokens_out"], int)

    methods = {rec["method"] for rec in records}
    assert "program_of_thought" in methods, f"missing program_of_thought records: {methods}"
    assert "repot" in methods, f"missing repot records: {methods}"

    environments = {rec["environment"] for rec in records}
    assert environments, "no environments observed in trace"
    # local_dummy hits at least one configured environment per problem.
    assert all(env for env in environments), "blank environment field"
