"""Shared prompt templates for code-emitting agents (RePoT and VEX).

The single template here, ``code_prompt``, is the verified-prefix-conditioned
code-emission prompt that asks the model to print

    moves = [...]

containing primitive moves to apply *from the current verified state*. Both
RePoT (Algorithm 1, suffix repair) and VEX v6 (per-chunk code proposer) use it
unchanged. Cache-friendly layout: the stable byte-identical block (problem
text, goal, format instruction) sits on top; the dynamic verifier-checkpoint
block (state, legal moves, last error, blocked actions, prefix tail) sits
below the marker line.
"""

from __future__ import annotations

import json
from typing import Any

from repot.core.schemas import Problem
from repot.core.env import PuzzleEnv


_CHECKPOINT_MARKER = "--- verifier checkpoint below ---"


def code_prompt(
    problem: Problem,
    env: PuzzleEnv,
    state: Any,
    verified_prefix: list,
    chunk_size: int,
    last_error: str,
    *,
    blocked: list[tuple] | None = None,
    hide_prefix: bool = False,
    legal_summary: str | None = None,
    blocked_summary: str | None = None,
) -> str:
    """Verified-prefix-conditioned code-emission prompt.

    Parameters
    ----------
    chunk_size:
        Maximum moves the model should emit (RePoT: remaining horizon;
        VEX: K).
    verified_prefix:
        The full list of already-committed actions. Only the recency tail is
        rendered into the prompt (cap=4 by default; 0 when ``hide_prefix``).
    last_error:
        The verifier's message from the most recent failed attempt, if any.
    blocked:
        Per-state tabu list; rendered as text the model is told not to repeat.
    hide_prefix:
        Recovery-ablation flag. When True the prefix tail is omitted entirely
        (the ``repot_no_prefix`` / ``vex_no_prefix`` ablation conditions).
    legal_summary, blocked_summary:
        Optional pre-rendered strings — VEX passes these from
        ``ChunkedPlanPolicy._legal_action_summary`` and ``format_blocked``.
    """
    stable_block = (
        f"{problem.natural_language_prompt}\n"
        f"Goal state: {json.dumps(problem.goal_state, sort_keys=True)}\n"
        f"Write Python code that prints exactly one line:\n"
        f"  moves = [...]\n"
        f"where the list contains up to {chunk_size} primitive moves to apply *from the "
        f"current verified state*. Do NOT repeat moves already in the verified prefix. "
        f"It is acceptable to print fewer than {chunk_size} moves if you are not confident. "
        f"Each move must be in the same shape as the puzzle's primitive-move format.\n"
        f"{_CHECKPOINT_MARKER}\n"
    )
    prefix_summary = "(hidden — ablation)" if hide_prefix else format_verified_prefix(verified_prefix)
    if legal_summary is None:
        legal_summary = _legal_summary_default(env, state)
    if blocked_summary is None:
        blocked_summary = _blocked_summary_default(blocked or [])
    dynamic_block = (
        f"You have already executed {len(verified_prefix)} verified moves. "
        f"Recent verified moves: {prefix_summary}\n"
        f"Current verified state: {json.dumps(state, sort_keys=True)}\n"
        f"Legal moves from this state: {legal_summary}\n"
        f"Blocked from this state (do not repeat): {blocked_summary}\n"
        f"Verifier message from last attempt: {last_error or 'none'}"
    )
    return stable_block + dynamic_block


def format_verified_prefix(verified_prefix: list, cap: int = 4) -> str:
    """Render only the last ``cap`` committed moves.

    The verifier owns the full prefix; the model only needs a recency tail to
    maintain continuity. Default cap=4 (lever D from v7).
    """
    if not verified_prefix:
        return "(none — at the initial state)"
    tail = verified_prefix[-cap:]
    earlier = max(0, len(verified_prefix) - cap)
    more = "" if earlier == 0 else f" ... (+{earlier} earlier moves elided)"
    try:
        rendered = json.dumps(tail, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = str(tail)
    return rendered + more


def _legal_summary_default(env: PuzzleEnv, state: Any) -> str:
    actions = env.legal_actions(state)
    if not actions:
        return "(no legal actions)"
    rendered = [json.dumps(a, sort_keys=True) for a in actions[:24]]
    more = "" if len(actions) <= 24 else f" ... (+{len(actions) - 24} more)"
    return ", ".join(rendered) + more


def _blocked_summary_default(blocked: list[tuple]) -> str:
    if not blocked:
        return "(none)"
    parts = []
    for action, message in blocked[:8]:
        parts.append(f"{json.dumps(action, sort_keys=True)} ({message})")
    more = "" if len(blocked) <= 8 else f" ... (+{len(blocked) - 8} more)"
    return "; ".join(parts) + more
