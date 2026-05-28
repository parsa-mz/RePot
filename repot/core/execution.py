"""Sandboxed Python execution and deterministic verified replay.

Two primitives that together make RePoT's "verified prefix" possible:

- ``execute_python_code`` runs an LLM-emitted program in a locked-down
  subprocess (`-I` + AST allowlist) and captures stdout/stderr.
- ``replay_until_failure`` walks the parsed action list through the env's
  verifier and stops at the first invalid transition, returning the valid
  prefix and the verifier's error so the caller can condition a repair.

The same ``replay_until_failure`` is reused by ``core.evaluation`` for the
prefix-execution-up-to-injection path in Derail.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repot.core.env import PuzzleEnv
from repot.core.schemas import Problem, State


# ---------------------------------------------------------------------------
# Sandboxed Python execution
# ---------------------------------------------------------------------------

PYTHON_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

ALLOWED_IMPORTS = {"collections", "functools", "itertools", "json", "math"}
BANNED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
BANNED_MODULES = {
    "builtins",
    "ctypes",
    "importlib",
    "inspect",
    "os",
    "pathlib",
    "resource",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
}


@dataclass(frozen=True)
class PythonExecutionResult:
    """Outcome of running an LLM-emitted Python program in the sandbox."""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    timed_out: bool = False


def extract_python_code(text: str) -> str:
    """Return the longest ```python``` fenced block in ``text``, or the full text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    fences = [match.group(1).strip() for match in PYTHON_FENCE_RE.finditer(text)]
    if fences:
        return max(fences, key=len)
    return text


def execute_python_code(code: str, timeout_s: float = 3.0, max_stdout_chars: int = 200_000) -> PythonExecutionResult:
    """Run ``code`` in an isolated subprocess; return stdout/stderr or the failure reason."""
    try:
        _validate_python_code(code)
    except ValueError as exc:
        return PythonExecutionResult(ok=False, error=str(exc))

    with tempfile.TemporaryDirectory(prefix="repot_pot_") as tmp:
        script = Path(tmp) / "program.py"
        script.write_text(code)
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=tmp,
                env={},
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return PythonExecutionResult(
                ok=False,
                stdout=(exc.stdout or "")[:max_stdout_chars],
                stderr=(exc.stderr or "")[:max_stdout_chars],
                error=f"Python execution timed out after {timeout_s:g}s.",
                timed_out=True,
            )

    stdout = completed.stdout[:max_stdout_chars]
    stderr = completed.stderr[:max_stdout_chars]
    if completed.returncode != 0:
        return PythonExecutionResult(
            ok=False,
            stdout=stdout,
            stderr=stderr,
            error=f"Python execution failed with exit code {completed.returncode}.",
        )
    return PythonExecutionResult(ok=True, stdout=stdout, stderr=stderr)


def _validate_python_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Python code has syntax error: {exc}") from exc
    _SafetyVisitor().visit(tree)


class _SafetyVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        """Reject `import X` for any X outside the allow-list."""
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root not in ALLOWED_IMPORTS or root in BANNED_MODULES:
                raise ValueError(f"Import is not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        """Reject `from X import …` for X outside the allow-list and any relative imports."""
        root = (node.module or "").split(".", 1)[0]
        if node.level or root not in ALLOWED_IMPORTS or root in BANNED_MODULES:
            raise ValueError(f"Import is not allowed: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        """Reject calls to banned built-ins (eval, exec, open, etc.)."""
        name = _call_name(node.func)
        if name in BANNED_CALLS:
            raise ValueError(f"Function call is not allowed: {name}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        """Reject dunder attribute access (e.g. `obj.__class__`)."""
        if node.attr.startswith("__"):
            raise ValueError(f"Dunder attribute access is not allowed: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        """Reject any reference to a dunder name (e.g. `__import__`)."""
        if node.id.startswith("__"):
            raise ValueError(f"Dunder name access is not allowed: {node.id}")
        self.generic_visit(node)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


# ---------------------------------------------------------------------------
# Deterministic verified replay
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    """Outcome of replaying a candidate move list through the verifier."""

    success: bool
    valid_prefix_actions: list = field(default_factory=list)
    invalid_action: Any | None = None
    first_failure_step: int | None = None
    final_state: State | None = None
    error_type: str | None = None
    error_message: str = ""
    valid_prefix_fraction: float = 0.0


def replay_until_failure(
    env: PuzzleEnv,
    problem: Problem,
    actions: list,
    start_state: State | None = None,
) -> ReplayResult:
    """Replay ``actions`` through ``env`` from ``start_state``; stop at the first invalid transition."""
    state: State = start_state if start_state is not None else problem.initial_state
    valid: list = []
    n = len(actions)

    for idx, raw in enumerate(actions):
        try:
            action = env.normalize_candidate_move(problem, raw, state)
        except Exception as exc:  # noqa: BLE001 — surface the env's reason
            return ReplayResult(
                success=False,
                valid_prefix_actions=valid,
                invalid_action=raw,
                first_failure_step=idx,
                final_state=state,
                error_type="normalization_error",
                error_message=str(exc),
                valid_prefix_fraction=idx / n if n else 0.0,
            )

        result = env.step(state, action)
        if not result.valid:
            return ReplayResult(
                success=False,
                valid_prefix_actions=valid,
                invalid_action=action,
                first_failure_step=idx,
                final_state=state,
                error_type=result.error_type.value if result.error_type else "invalid_transition",
                error_message=result.message or "invalid transition",
                valid_prefix_fraction=idx / n if n else 0.0,
            )

        state = result.next_state
        valid.append(action)

    return ReplayResult(
        success=True,
        valid_prefix_actions=valid,
        invalid_action=None,
        first_failure_step=None,
        final_state=state,
        error_type=None,
        error_message="",
        valid_prefix_fraction=1.0,
    )
