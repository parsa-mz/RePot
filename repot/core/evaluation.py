"""Trace evaluation, metric aggregation, and Derail recovery driver.

Sections:
  1. Failure classification          — classify_failure
  2. Candidate extraction + replay   — extract_candidate_solutions / extract_and_replay_candidates
  3. Trace evaluation                — evaluate_run (one TraceRun → one TraceEvaluation)
  4. Metric aggregation              — aggregate_metrics, collapse_taxonomy
  5. Derail recovery                 — RecoveryRecord / RecoveryCase / make_recovery_case / run_recovery_condition
"""

from __future__ import annotations

import ast
import json
import re
import time
from collections import defaultdict
from dataclasses import replace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from repot.core.env import PuzzleEnv
from repot.core.llm import GenerateRequest, ModelClient
from repot.core.parsing import extract_actions
from repot.core.schemas import (
    Action,
    CandidateSolution,
    FailureType,
    Problem,
    TraceEvaluation,
    TraceRun,
)


# =============================================================================
# 1. Failure classification
# =============================================================================


def classify_failure(run: TraceRun) -> FailureType | None:
    """Pick the dominant FailureType for a failed TraceRun; None if it succeeded."""
    if run.success:
        return None
    if not run.steps:
        return FailureType.OUTPUT_FORMAT_ERROR

    first_bad = next((s for s in run.steps if not s.valid), None)
    if first_bad is not None:
        return first_bad.error_type or FailureType.INVALID_TRANSITION

    drift = next(
        (
            s
            for s in run.steps
            if s.predicted_next_state is not None
            and s.actual_next_state is not None
            and s.predicted_next_state != s.actual_next_state
        ),
        None,
    )
    if drift is not None:
        return FailureType.STATE_DRIFT

    if run.stopped_reason == "repetition_loop":
        return FailureType.REPETITION_LOOP
    if run.stopped_reason in {"max_steps", "max_steps_or_premature_stop"}:
        return FailureType.REPETITION_LOOP if _has_repeated_states(run) else FailureType.PREMATURE_STOP
    return FailureType.WRONG_GOAL


def _has_repeated_states(run: TraceRun) -> bool:
    seen: set[str] = set()
    for step in run.steps:
        key = repr(step.actual_next_state or step.current_state)
        if key in seen:
            return True
        seen.add(key)
    return False


# =============================================================================
# 2. Candidate extraction + replay
# =============================================================================


FINAL_RE = re.compile(r"\b(final|answer|therefore|solution)\b", re.IGNORECASE)
MOVES_RE = re.compile(r"\bmoves\s*=", re.IGNORECASE)
TOKEN_RE = re.compile(r"\S+")


def extract_and_replay_candidates(text: str, problem: Problem, env: PuzzleEnv) -> list[CandidateSolution]:
    """Extract every candidate solution in ``text`` and replay each one through ``env``."""
    candidates = extract_candidate_solutions(text, problem, env)
    replayed: list[CandidateSolution] = []
    for candidate in candidates:
        replayed.append(_replay_candidate(candidate, problem, env))
    return replayed


def extract_candidate_solutions(text: str, problem: Problem, env: PuzzleEnv) -> list[CandidateSolution]:
    """Find every move-list candidate in ``text`` (deduped, with token positions)."""
    raw_candidates = _raw_candidates(text)
    final_marker = _last_final_marker(text)
    total_tokens = len(TOKEN_RE.findall(text))
    deduped: list[CandidateSolution] = []
    seen: set[str] = set()

    for raw_moves, start, end in raw_candidates:
        source = _candidate_source(start, final_marker, raw_candidates)
        try:
            actions = _normalize_moves(raw_moves, problem, env)
        except (KeyError, TypeError, ValueError):
            continue
        dedupe_key = json.dumps(actions, sort_keys=True, separators=(",", ":"))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        token_start = len(TOKEN_RE.findall(text[:start]))
        token_end = len(TOKEN_RE.findall(text[:end]))
        deduped.append(
            CandidateSolution(
                candidate_id=len(deduped),
                source=source,
                raw_moves=raw_moves,
                actions=actions,
                char_start=start,
                char_end=end,
                token_start=token_start,
                token_end=token_end,
                normalized_token_position=(token_start / total_tokens) if total_tokens else None,
                valid_format=True,
                dedupe_key=dedupe_key,
            )
        )
    return deduped


def _raw_candidates(text: str) -> list[tuple[list[Any], int, int]]:
    candidates: list[tuple[list[Any], int, int]] = []
    for raw_moves in _json_action_candidates(text):
        candidates.append((raw_moves, 0, len(text)))
    for start, end in _balanced_list_spans(text):
        snippet = text[start:end]
        parsed = _parse_literal(snippet)
        if _looks_like_move_sequence(parsed):
            candidates.append((parsed, start, end))
    return candidates


def _json_action_candidates(text: str) -> list[list[Any]]:
    try:
        actions = extract_actions(text)
    except Exception:
        return []
    return [actions] if actions else []


def _balanced_list_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    starts = [match.end() for match in MOVES_RE.finditer(text)]
    starts.extend(match.start() for match in re.finditer(r"\[", text))
    for start_hint in sorted(set(starts)):
        start = text.find("[", start_hint)
        if start < 0:
            continue
        depth = 0
        in_string: str | None = None
        escaped = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == in_string:
                    in_string = None
                continue
            if char in {"'", '"'}:
                in_string = char
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    spans.append((start, idx + 1))
                    break
    return sorted(set(spans), key=lambda item: (item[0], item[1]))


def _parse_literal(snippet: str) -> Any:
    cleaned = _strip_code_fence(snippet.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(cleaned)
    except (SyntaxError, ValueError):
        return None


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1])
    return text


def _looks_like_move_sequence(parsed: Any) -> bool:
    if not isinstance(parsed, list) or not parsed:
        return False
    if all(isinstance(item, dict) for item in parsed):
        return True
    return all(isinstance(item, (list, tuple)) for item in parsed)


def _normalize_moves(raw_moves: list[Any], problem: Problem, env: PuzzleEnv) -> list[dict[str, Any]]:
    state = json.loads(json.dumps(problem.initial_state))
    actions: list[dict[str, Any]] = []
    for raw_move in raw_moves:
        action = env.normalize_candidate_move(problem, raw_move, state)
        actions.append(action)
        result = env.step(state, action)
        if result.valid:
            state = result.next_state
    return actions


def _replay_candidate(candidate: CandidateSolution, problem: Problem, env: PuzzleEnv) -> CandidateSolution:
    state = json.loads(json.dumps(problem.initial_state))
    first_failure_move: int | None = None
    failure_type: FailureType | None = None
    for idx, action in enumerate(candidate.actions):
        result = env.step(state, action)
        if not result.valid:
            first_failure_move = idx
            failure_type = result.error_type or FailureType.INVALID_TRANSITION
            break
        state = result.next_state
    replay_success = first_failure_move is None and env.is_goal(state, problem.goal_state)
    if not replay_success and first_failure_move is None:
        first_failure_move = len(candidate.actions)
        failure_type = FailureType.WRONG_GOAL
    denominator = max(1, problem.min_steps or len(candidate.actions) or 1)
    return candidate.model_copy(
        update={
            "replay_success": replay_success,
            "first_failure_move": None if replay_success else first_failure_move,
            "first_failure_fraction": None if replay_success else first_failure_move / denominator,
            "failure_type": None if replay_success else failure_type,
        }
    )


def _last_final_marker(text: str) -> int | None:
    matches = list(FINAL_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].start()


def _candidate_source(
    start: int,
    final_marker: int | None,
    raw_candidates: list[tuple[list[Any], int, int]],
) -> str:
    if final_marker is not None and start >= final_marker:
        return "final"
    last_start = max(item[1] for item in raw_candidates) if raw_candidates else start
    if start == last_start:
        return "final"
    return "thought"


# =============================================================================
# 3. Trace evaluation — one TraceRun -> one TraceEvaluation
# =============================================================================


def evaluate_run(run: TraceRun, problem: Problem | None = None, env: PuzzleEnv | None = None) -> TraceEvaluation:
    """Score a TraceRun. With ``problem`` + ``env`` uses the strict verifier path; otherwise lightweight."""
    if problem is not None and env is not None:
        evaluation = env.verify_trace(problem, run.steps)
        evaluation = evaluation.model_copy(
            update={
                "method": run.method,
                "model": run.model,
                "success": run.success and evaluation.success,
                "failure_type": None if run.success and evaluation.success else (evaluation.failure_type or classify_failure(run)),
                "tokens_total": run.tokens_in + run.tokens_out,
                "recovery_success_rate": _repair_successes(run),
                "rollback_count": sum(1 for step in run.steps if not step.valid),
                "pot_success": bool(run.metadata.get("pot_success", run.method == "program_of_thought" and run.success)),
                "pot_plan_valid_prefix_fraction": run.metadata.get("tool_plan_valid_prefix_fraction"),
                "pot_first_failure_step": _tool_plan_first_failure(run),
                "pot_repaired_success": bool(run.metadata.get("pot_repaired_success", False)),
                "auto_start_policy": run.metadata.get("auto_start_policy"),
                "auto_policy_route": run.metadata.get("auto_policy_route", []),
                "auto_switch_reasons": run.metadata.get("auto_switch_reasons", []),
                "auto_tool_attempted": bool(run.metadata.get("auto_tool_attempted", False)),
                "auto_tool_valid_prefix_fraction": run.metadata.get("auto_tool_valid_prefix_fraction"),
                "auto_lookahead_steps": int(run.metadata.get("auto_lookahead_steps", 0)),
                "auto_model_rank_steps": int(run.metadata.get("auto_model_rank_steps", 0)),
                "auto_policy_switch_count": int(run.metadata.get("auto_policy_switch_count", 0)),
                "auto_v2_chunk_size_history": list(run.metadata.get("auto_v2_chunk_size_history", []) or []),
                "llm_calls": _llm_calls(run),
                "verified_steps_per_llm_call": _verified_steps_per_llm_call(run),
                "valid_prefix_reuse_rate": float(run.metadata.get("valid_prefix_reuse_rate", run.metadata.get("auto_v2_valid_prefix_reuse_rate", 0.0)) or 0.0),
                "repair_call_rate": _repair_call_rate(run),
                "call_budget_exhaustion_rate": 1.0 if run.metadata.get("call_budget_exhausted", run.metadata.get("auto_v2_call_budget_exhausted", False)) else 0.0,
                "success_under_call_budget": bool(run.metadata.get("success_under_call_budget", run.success)),
                "provider_rejection": _detect_provider_rejection(run),
            }
        )
        return _with_candidate_evaluation(evaluation, run, problem, env)

    # Reconstruct a lightweight evaluation from trace fields when no env is supplied.
    if not run.steps:
        return TraceEvaluation(
            problem_id=run.problem_id,
            environment=run.environment,
            complexity=run.complexity,
            method=run.method,
            model=run.model,
            success=False,
            failure_type=classify_failure(run),
            explanation="Trace contains no steps.",
        )

    valid = sum(1 for s in run.steps if s.valid)
    invalid = [s for s in run.steps if not s.valid]
    drift = [
        s
        for s in run.steps
        if s.predicted_next_state is not None
        and s.actual_next_state is not None
        and s.predicted_next_state != s.actual_next_state
    ]
    constraint = [s for s in invalid if s.error_type and s.error_type.value == "CONSTRAINT_VIOLATION"]
    tokens = run.tokens_in + run.tokens_out
    failure = classify_failure(run)
    repeated = _repeated_state_count(run)
    recoveries = _repair_successes(run)

    return TraceEvaluation(
        problem_id=run.problem_id,
        environment=run.environment,
        complexity=run.complexity,
        method=run.method,
        model=run.model,
        success=run.success,
        failure_type=None if run.success else failure,
        first_failure_step=None if run.success else (invalid[0].step if invalid else run.steps[-1].step),
        valid_move_rate=valid / len(run.steps) if run.steps else 0.0,
        first_invalid_step=invalid[0].step if invalid else None,
        state_drift_rate=len(drift) / len(run.steps) if run.steps else 0.0,
        constraint_violation_rate=len(constraint) / len(run.steps) if run.steps else 0.0,
        recovery_success_rate=recoveries,
        rollback_count=len(invalid),
        repeated_state_count=repeated,
        premature_stop=not run.success and failure and failure.value == "PREMATURE_STOP",
        legal_action_adherence_rate=_legal_action_adherence_rate(run),
        unknown_action_id_rate=_unknown_action_id_rate(run),
        invalid_after_resolution_rate=_invalid_after_resolution_rate(run),
        loop_entry_rate=repeated / len(run.steps) if run.steps else 0.0,
        loop_recovery_success_rate=_loop_recovery_success_rate(run),
        state_drift_detected_rate=len(drift) / len(run.steps) if run.steps else 0.0,
        state_drift_corrected_rate=_state_drift_corrected_rate(run),
        normalized_first_failure_step=None,
        solution_length_ratio=None,
        parse_schema_success_rate=_parse_schema_success_rate(run),
        pot_success=bool(run.metadata.get("pot_success", run.method == "program_of_thought" and run.success)),
        pot_plan_valid_prefix_fraction=run.metadata.get("tool_plan_valid_prefix_fraction"),
        pot_first_failure_step=_tool_plan_first_failure(run),
        pot_repaired_success=bool(run.metadata.get("pot_repaired_success", False)),
        auto_start_policy=run.metadata.get("auto_start_policy"),
        auto_policy_route=run.metadata.get("auto_policy_route", []),
        auto_switch_reasons=run.metadata.get("auto_switch_reasons", []),
        auto_tool_attempted=bool(run.metadata.get("auto_tool_attempted", False)),
        auto_tool_valid_prefix_fraction=run.metadata.get("auto_tool_valid_prefix_fraction"),
        auto_lookahead_steps=int(run.metadata.get("auto_lookahead_steps", 0)),
        auto_model_rank_steps=int(run.metadata.get("auto_model_rank_steps", 0)),
        auto_policy_switch_count=int(run.metadata.get("auto_policy_switch_count", 0)),
        llm_calls=_llm_calls(run),
        verified_steps_per_llm_call=_verified_steps_per_llm_call(run),
        valid_prefix_reuse_rate=float(run.metadata.get("valid_prefix_reuse_rate", run.metadata.get("auto_v2_valid_prefix_reuse_rate", 0.0)) or 0.0),
        repair_call_rate=_repair_call_rate(run),
        call_budget_exhaustion_rate=1.0 if run.metadata.get("call_budget_exhausted", run.metadata.get("auto_v2_call_budget_exhausted", False)) else 0.0,
        success_under_call_budget=bool(run.metadata.get("success_under_call_budget", run.success)),
        provider_rejection=_detect_provider_rejection(run),
        auto_v2_chunk_size_history=list(run.metadata.get("auto_v2_chunk_size_history", []) or []),
        tokens_total=tokens,
        tokens_per_valid_step=tokens / valid if valid else None,
        explanation="" if run.success else f"Primary failure: {failure}",
    )


def _with_candidate_evaluation(
    evaluation: TraceEvaluation,
    run: TraceRun,
    problem: Problem,
    env: PuzzleEnv,
) -> TraceEvaluation:
    if not _is_one_shot_baseline(run.method):
        return evaluation
    raw_output = "\n\n".join(step.raw_model_output for step in run.steps if step.raw_model_output)
    if not raw_output:
        return evaluation
    candidates = extract_and_replay_candidates(raw_output, problem, env)
    if not candidates:
        return evaluation.model_copy(update={"format_filter_pass": False})

    final_candidates = [candidate for candidate in candidates if candidate.source == "final"]
    chosen = _chosen_final_candidate(final_candidates, candidates)
    final_success = any(candidate.replay_success for candidate in final_candidates) if final_candidates else chosen.replay_success
    thought_contains_success = any(candidate.replay_success for candidate in candidates if candidate.source != "final")
    first_solution_positions = [
        candidate.normalized_token_position for candidate in candidates if candidate.replay_success
    ]
    first_solution_positions = [pos for pos in first_solution_positions if pos is not None]
    valid_moves = _candidate_valid_moves(chosen)
    candidate_total = len(chosen.actions)
    failure_type = None if final_success else (chosen.failure_type or evaluation.failure_type or classify_failure(run))
    first_failure_step = None if final_success else chosen.first_failure_move
    return evaluation.model_copy(
        update={
            "success": final_success,
            "failure_type": failure_type,
            "first_failure_step": first_failure_step,
            "valid_move_rate": (valid_moves / candidate_total) if candidate_total else evaluation.valid_move_rate,
            "first_invalid_step": None if final_success else chosen.first_failure_move,
            "final_success": final_success,
            "thought_contains_success": thought_contains_success,
            "correct_in_thought_wrong_final": thought_contains_success and not final_success,
            "first_solution_token_position": min(first_solution_positions) if first_solution_positions else None,
            "first_failure_move_fraction": None if final_success else chosen.first_failure_fraction,
            "num_unique_candidate_solutions": len(candidates),
            "format_filter_pass": True,
            "candidate_solutions": candidates,
            "explanation": "" if final_success else f"Primary failure: {failure_type}",
        }
    )


def _is_one_shot_baseline(method: str) -> bool:
    return method in {
        "cot",
        "long_cot",
        "algorithm_prompted",
        "state_table",
        "verifier_only",
        "self_consistency",
        "tool_only",
        "program_of_thought",
    }


def _chosen_final_candidate(final_candidates, candidates):
    for candidate in final_candidates:
        if candidate.replay_success:
            return candidate
    if final_candidates:
        return final_candidates[-1]
    return candidates[-1]


def _candidate_valid_moves(candidate) -> int:
    if candidate.replay_success:
        return len(candidate.actions)
    if candidate.first_failure_move is None:
        return 0
    return min(candidate.first_failure_move, len(candidate.actions))


def _repeated_state_count(run: TraceRun) -> int:
    seen: set[str] = set()
    repeated = 0
    for step in run.steps:
        key = repr(step.actual_next_state or step.current_state)
        if key in seen:
            repeated += 1
        seen.add(key)
    return repeated


def _repair_successes(run: TraceRun) -> float:
    invalid_indices = [i for i, step in enumerate(run.steps) if not step.valid]
    if not invalid_indices:
        return 0.0
    repaired = 0
    for idx in invalid_indices:
        if any(step.valid for step in run.steps[idx + 1 :]):
            repaired += 1
    return repaired / len(invalid_indices)


def _legal_action_adherence_rate(run: TraceRun) -> float:
    candidates = [s for s in run.steps if s.legal_action_ids]
    if not candidates:
        return 0.0
    return sum(1 for s in candidates if s.action_id in s.legal_action_ids) / len(candidates)


def _unknown_action_id_rate(run: TraceRun) -> float:
    candidates = [s for s in run.steps if s.action_id is not None or s.legal_action_ids]
    if not candidates:
        return 0.0
    return sum(1 for s in candidates if s.error_type == FailureType.UNKNOWN_ACTION_ID) / len(candidates)


def _invalid_after_resolution_rate(run: TraceRun) -> float:
    resolved = [
        s
        for s in run.steps
        if s.action_id is not None
        and s.legal_action_ids
        and s.action_id in s.legal_action_ids
        and s.error_type != FailureType.REPETITION_LOOP
    ]
    if not resolved:
        return 0.0
    return sum(1 for s in resolved if not s.valid) / len(resolved)


def _loop_recovery_success_rate(run: TraceRun) -> float:
    attempts = [idx for idx, s in enumerate(run.steps) if s.loop_recovery_attempted]
    if not attempts:
        return 0.0
    recovered = 0
    for idx in attempts:
        if run.success or any(s.valid and s.error_type is None for s in run.steps[idx + 1 :]):
            recovered += 1
    return recovered / len(attempts)


def _state_drift_corrected_rate(run: TraceRun) -> float:
    drift_indices = [
        idx
        for idx, s in enumerate(run.steps)
        if s.predicted_next_state is not None
        and s.actual_next_state is not None
        and s.predicted_next_state != s.actual_next_state
    ]
    if not drift_indices:
        return 0.0
    corrected = 0
    for idx in drift_indices:
        if run.success or any(s.valid for s in run.steps[idx + 1 :]):
            corrected += 1
    return corrected / len(drift_indices)


def _parse_schema_success_rate(run: TraceRun) -> float:
    if not run.steps:
        return 0.0
    bad = {
        FailureType.OUTPUT_FORMAT_ERROR,
        FailureType.TOKEN_TRUNCATION,
        FailureType.UNKNOWN_ACTION_ID,
    }
    return sum(1 for s in run.steps if s.error_type not in bad) / len(run.steps)


def _llm_calls(run: TraceRun) -> int:
    explicit = run.metadata.get("llm_call_count", run.metadata.get("auto_v2_llm_call_count"))
    if explicit is not None:
        return int(explicit)
    return sum(1 for step in run.steps if step.tokens_in or step.tokens_out or step.raw_model_output)


def _verified_steps_per_llm_call(run: TraceRun) -> float | None:
    explicit = run.metadata.get(
        "verified_steps_per_llm_call",
        run.metadata.get("auto_v2_verified_steps_per_llm_call"),
    )
    if explicit is not None:
        return float(explicit)
    calls = _llm_calls(run)
    if calls <= 0:
        return None
    return sum(1 for step in run.steps if step.valid) / calls


def _repair_call_rate(run: TraceRun) -> float:
    repairs = run.metadata.get("repair_call_count", run.metadata.get("auto_v2_repair_call_count"))
    calls = _llm_calls(run)
    if calls <= 0:
        return 0.0
    return int(repairs or 0) / calls


def _tool_plan_first_failure(run: TraceRun) -> int | None:
    if "tool_plan_length" not in run.metadata:
        return None
    plan_length = int(run.metadata.get("tool_plan_length") or 0)
    for step in run.steps[:plan_length]:
        if not step.valid:
            return step.step
    if run.success:
        return None
    return min(plan_length, len(run.steps)) if plan_length else None


def _detect_provider_rejection(run: TraceRun) -> bool:
    for step in run.steps:
        if step.error_type in {FailureType.PROVIDER_REJECTION, FailureType.API_ERROR}:
            return True
    if run.stopped_reason == "runner_exception":
        return True
    return False


# =============================================================================
# 4. Metric aggregation
# =============================================================================


def aggregate_metrics(evals: list[TraceEvaluation]) -> dict[str, Any]:
    """Aggregate per-run evaluations into per-group summaries + boundary curves + paired comparisons."""
    groups: dict[tuple[str, str, str, int], list[TraceEvaluation]] = defaultdict(list)
    for ev in evals:
        groups[(ev.model, ev.environment, ev.method, ev.complexity)].append(ev)

    by_group: list[dict[str, Any]] = []
    method_complexities: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    collapse_group_counts: dict[tuple[str, str, str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ev in evals:
        collapse_type = collapse_taxonomy(ev)
        if collapse_type:
            collapse_group_counts[(ev.model, ev.environment, ev.method, ev.complexity)][collapse_type] += 1

    for (model, env, method, complexity), rows in sorted(groups.items()):
        total_rows = len(rows)
        rejection_count = sum(1 for row in rows if _is_provider_rejection(row))
        attempted_rows = max(0, total_rows - rejection_count)
        successes = sum(row.success for row in rows)
        success_rate = successes / total_rows if total_rows else 0.0
        success_rate_excluding_rejections = (
            successes / attempted_rows if attempted_rows else None
        )
        method_complexities[(model, env, method)][complexity] = success_rate
        taxonomy_counts = dict(sorted(collapse_group_counts[(model, env, method, complexity)].items()))
        success_rows = [row for row in rows if row.success]
        attempt_rows = [row for row in rows if not _is_provider_rejection(row)]
        tokens_per_success_mean = _mean(row.tokens_total for row in success_rows)
        tokens_per_success_amortised = (
            _sum(row.tokens_total for row in attempt_rows) / successes if successes else None
        )
        llm_calls_per_success_mean = _mean(row.llm_calls for row in success_rows)
        llm_calls_per_success_amortised = (
            _sum(row.llm_calls for row in attempt_rows) / successes if successes else None
        )
        by_group.append(
            {
                "model": model,
                "environment": env,
                "method": method,
                "complexity": complexity,
                "n": total_rows,
                "n_attempts": attempted_rows,
                "n_successes": successes,
                "success_rate": success_rate,
                "success_rate_excluding_rejections": success_rate_excluding_rejections,
                "provider_rejection_rate": rejection_count / total_rows if total_rows else 0.0,
                "valid_move_rate": _mean(row.valid_move_rate for row in rows),
                "state_drift_rate": _mean(row.state_drift_rate for row in rows),
                "constraint_violation_rate": _mean(row.constraint_violation_rate for row in rows),
                "recovery_success_rate": _mean(row.recovery_success_rate for row in rows),
                "legal_action_adherence_rate": _mean(row.legal_action_adherence_rate for row in rows),
                "unknown_action_id_rate": _mean(row.unknown_action_id_rate for row in rows),
                "invalid_after_resolution_rate": _mean(row.invalid_after_resolution_rate for row in rows),
                "loop_entry_rate": _mean(row.loop_entry_rate for row in rows),
                "loop_recovery_success_rate": _mean(row.loop_recovery_success_rate for row in rows),
                "state_drift_detected_rate": _mean(row.state_drift_detected_rate for row in rows),
                "state_drift_corrected_rate": _mean(row.state_drift_corrected_rate for row in rows),
                "normalized_first_failure_step": _mean(
                    row.normalized_first_failure_step
                    for row in rows
                    if row.normalized_first_failure_step is not None
                ),
                "solution_length_ratio": _mean(
                    row.solution_length_ratio for row in rows if row.solution_length_ratio is not None
                ),
                "parse_schema_success_rate": _mean(row.parse_schema_success_rate for row in rows),
                "final_success_rate": _mean(float(row.final_success) for row in rows),
                "thought_contains_success_rate": _mean(float(row.thought_contains_success) for row in rows),
                "correct_in_thought_wrong_final_rate": _mean(
                    float(row.correct_in_thought_wrong_final) for row in rows
                ),
                "first_solution_token_position": _mean(
                    row.first_solution_token_position
                    for row in rows
                    if row.first_solution_token_position is not None
                ),
                "first_failure_move_fraction": _mean(
                    row.first_failure_move_fraction for row in rows if row.first_failure_move_fraction is not None
                ),
                "num_unique_candidate_solutions": _mean(row.num_unique_candidate_solutions for row in rows),
                "format_filter_pass_rate": _mean(float(row.format_filter_pass) for row in rows),
                "pot_success_rate": _mean(float(row.pot_success) for row in rows),
                "pot_plan_valid_prefix_fraction": _mean(
                    row.pot_plan_valid_prefix_fraction
                    for row in rows
                    if row.pot_plan_valid_prefix_fraction is not None
                ),
                "pot_repaired_success_rate": _mean(float(row.pot_repaired_success) for row in rows),
                "auto_tool_attempt_rate": _mean(float(row.auto_tool_attempted) for row in rows),
                "auto_tool_valid_prefix_fraction": _mean(
                    row.auto_tool_valid_prefix_fraction
                    for row in rows
                    if row.auto_tool_valid_prefix_fraction is not None
                ),
                "auto_lookahead_steps": _mean(row.auto_lookahead_steps for row in rows),
                "auto_model_rank_steps": _mean(row.auto_model_rank_steps for row in rows),
                "auto_policy_switch_count": _mean(row.auto_policy_switch_count for row in rows),
                "auto_policy_route_counts": _route_counts(rows),
                "auto_v2_chunk_size_history_max": _mean(
                    max(row.auto_v2_chunk_size_history) for row in rows if row.auto_v2_chunk_size_history
                ),
                "auto_v2_chunk_size_history_min": _mean(
                    min(row.auto_v2_chunk_size_history) for row in rows if row.auto_v2_chunk_size_history
                ),
                "llm_calls": _mean(row.llm_calls for row in rows),
                "llm_calls_per_success": llm_calls_per_success_mean,
                "llm_calls_per_success_mean": llm_calls_per_success_mean,
                "llm_calls_per_success_amortised": llm_calls_per_success_amortised,
                "verified_steps_per_llm_call": _mean(
                    row.verified_steps_per_llm_call
                    for row in rows
                    if row.verified_steps_per_llm_call is not None
                ),
                "valid_prefix_reuse_rate": _mean(row.valid_prefix_reuse_rate for row in rows),
                "repair_call_rate": _mean(row.repair_call_rate for row in rows),
                "call_budget_exhaustion_rate": _mean(row.call_budget_exhaustion_rate for row in rows),
                "success_under_call_budget_rate": _mean(float(row.success_under_call_budget) for row in rows),
                "tokens_per_success": tokens_per_success_mean,
                "tokens_per_success_mean": tokens_per_success_mean,
                "tokens_per_success_amortised": tokens_per_success_amortised,
                "tokens_per_valid_step": _mean(
                    row.tokens_per_valid_step for row in rows if row.tokens_per_valid_step is not None
                ),
                "collapse_taxonomy_counts": taxonomy_counts,
                "dominant_collapse_type": _dominant_key(taxonomy_counts),
            }
        )

    boundaries = []
    for (model, env, method), curve in sorted(method_complexities.items()):
        robust = [c for c, rate in curve.items() if rate >= 0.8]
        collapse = [c for c, rate in sorted(curve.items()) if rate < 0.1]
        auc = _auc_complexity(curve)
        boundaries.append(
            {
                "model": model,
                "environment": env,
                "method": method,
                "robust_n": max(robust) if robust else None,
                "robust_level": max(robust) if robust else None,
                "collapse_n": collapse[0] if collapse else None,
                "collapse_level": collapse[0] if collapse else None,
                "auc_complexity": auc,
            }
        )

    failures: dict[str, int] = defaultdict(int)
    collapse_failures: dict[str, int] = defaultdict(int)
    for ev in evals:
        if ev.failure_type is not None:
            failures[ev.failure_type.value] += 1
        collapse_type = collapse_taxonomy(ev)
        if collapse_type:
            collapse_failures[collapse_type] += 1

    return {
        "groups": by_group,
        "boundaries": boundaries,
        "collapse_boundary_shifts": _collapse_boundary_shifts(boundaries),
        "failure_counts": dict(sorted(failures.items())),
        "collapse_taxonomy_counts": dict(sorted(collapse_failures.items())),
        "paired": _paired_comparisons(evals),
    }


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _sum(values) -> float:
    return sum(v for v in values if v is not None)


def collapse_taxonomy(ev: TraceEvaluation) -> str | None:
    """Classify a failed evaluation by failure signature, not by environment name.

    Order matters: provider/format errors are infrastructure issues; recovery and loop
    signals come from the trace itself; finally fall through to state-execution
    collapse for any remaining transition/state/action error.
    """
    if ev.success:
        return None
    failure = ev.failure_type.value if ev.failure_type is not None else ""
    if failure == "TOKEN_TRUNCATION":
        return "TOKEN_BUDGET_COLLAPSE"
    if failure in {"OUTPUT_FORMAT_ERROR", "PROVIDER_REJECTION", "API_ERROR"}:
        return "FORMAT_COLLAPSE"
    if failure == "REPETITION_LOOP" or ev.repeated_state_count > 0:
        return "LOOP_COLLAPSE"
    if failure == "NO_RECOVERY_AFTER_ERROR" or (ev.recovery_success_rate == 0 and ev.rollback_count > 0):
        return "NO_RECOVERY_COLLAPSE"
    if failure == "CONSTRAINT_VIOLATION":
        return "CONSTRAINT_MAINTENANCE_COLLAPSE"
    if failure == "POLICY_DEVIATION":
        return "POLICY_SELECTION_COLLAPSE"
    if failure in {
        "INVALID_TRANSITION",
        "STATE_DRIFT",
        "UNKNOWN_ACTION_ID",
        "WRONG_GOAL",
        "PREMATURE_STOP",
        "VERIFIER_DISAGREEMENT",
    }:
        return "STATE_EXECUTION_COLLAPSE"
    return "UNCLASSIFIED_FAILURE"


def _is_provider_rejection(ev: TraceEvaluation) -> bool:
    if ev.provider_rejection:
        return True
    return ev.failure_type in {FailureType.PROVIDER_REJECTION, FailureType.API_ERROR}


def _dominant_key(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _auc_complexity(curve: dict[int, float]) -> float | None:
    if not curve:
        return None
    return sum(curve.values()) / len(curve)


def _collapse_boundary_shifts(boundaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["model"], row["environment"], row["method"]): row
        for row in boundaries
    }
    baseline_methods = ["cot", "algorithm_prompted", "self_consistency", "program_of_thought"]
    rows: list[dict[str, Any]] = []
    for row in boundaries:
        method = row["method"]
        if not method.startswith("stateguard_"):
            continue
        for baseline in baseline_methods:
            other = by_key.get((row["model"], row["environment"], baseline))
            if other is None:
                continue
            rows.append(
                {
                    "model": row["model"],
                    "environment": row["environment"],
                    "method": method,
                    "baseline": baseline,
                    "robust_level_delta": _delta(row.get("robust_level"), other.get("robust_level")),
                    "collapse_level_delta": _delta(row.get("collapse_level"), other.get("collapse_level")),
                    "auc_delta": _delta(row.get("auc_complexity"), other.get("auc_complexity")),
                }
            )
    return rows


def _delta(left, right):
    if left is None or right is None:
        return None
    return left - right


def _route_counts(rows: list[TraceEvaluation]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.auto_policy_route:
            counts[" -> ".join(row.auto_policy_route)] += 1
    return dict(sorted(counts.items()))


def _paired_comparisons(evals: list[TraceEvaluation]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int, str, str], TraceEvaluation] = {}
    for ev in evals:
        by_key[(ev.model, ev.environment, ev.complexity, ev.problem_id, ev.method)] = ev

    methods = sorted({ev.method for ev in evals})
    target_methods = [method for method in methods if method.startswith("stateguard_")]
    baselines = [m for m in methods if m not in target_methods]
    rows: list[dict[str, Any]] = []
    for target in target_methods:
        for baseline in baselines:
            wins = losses = ties = total = 0
            for key, sg in by_key.items():
                model, env, complexity, problem_id, method = key
                if method != target:
                    continue
                other = by_key.get((model, env, complexity, problem_id, baseline))
                if other is None:
                    continue
                total += 1
                if sg.success and not other.success:
                    wins += 1
                elif other.success and not sg.success:
                    losses += 1
                else:
                    ties += 1
            if total:
                rows.append(
                    {
                        "baseline": baseline,
                        "method": target,
                        "n": total,
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                    }
                )
    return rows


# =============================================================================
# 5. Derail recovery
# =============================================================================


RECOVERY_CONDITIONS = [
    "no_feedback",
    "error_only",
    "state_feedback",
    "state_plus_legal_actions",
    "stateguard_rollback",
    # VEX prefix-conditioning ablation. Three conditions share the same VEX
    # loop / verifier / tabu / code proposer; the only thing that differs is
    # what the proposer sees on call #1.
    #   vex_full                   : starts at the verified checkpoint, prefix exposed
    #   vex_no_prefix              : starts at the verified checkpoint, prefix hidden
    #   vex_restart_from_initial   : starts at the *original* initial state,
    #                                same max_llm_calls budget. Tests whether
    #                                the win comes from "more budget" vs
    #                                "verified-prefix repair".
    "vex_full",
    "vex_no_prefix",
    "vex_restart_from_initial",
    # RePoT (Algorithm 1) prefix-conditioning ablation — paper headline.
    # Same shape as the vex_* conditions, but the agent is RePoT (one-shot
    # PoT then suffix repair) instead of the chunked VEX controller.
    #   repot_full                 : starts at the verified checkpoint, prefix exposed
    #   repot_no_prefix            : starts at the verified checkpoint, prefix hidden
    #   repot_restart              : starts at the *original* initial state,
    #                                same max_repair_calls budget. Tests whether
    #                                the win comes from "another sample" vs
    #                                "verified-prefix repair".
    # Expected ordering if the novelty is load-bearing:
    #   repot_full  >=  repot_no_prefix  >>  repot_restart
    "repot_full",
    "repot_no_prefix",
    "repot_restart",
]


class RecoveryRecord(BaseModel):
    """One Derail recovery attempt: per-case × per-condition outcome row."""

    model_config = ConfigDict(extra="forbid")

    problem_id: str
    environment: str
    complexity: int
    model: str
    condition: str
    success: bool
    failure_type: FailureType | None = None
    injection_step: int
    injection_type: str
    last_valid_state: dict[str, Any]
    injected_action: Action
    injected_state: dict[str, Any]
    returned_action_count: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_time: float = 0.0
    raw_model_output: str = ""
    stopped_reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecoveryCase(BaseModel):
    """Method-agnostic Derail case: problem + checkpoint + the injected derailment."""

    model_config = ConfigDict(extra="forbid")

    problem: Problem
    checkpoint_problem: Problem
    injection_step: int
    injection_type: str
    last_valid_state: dict[str, Any]
    injected_action: Action
    injected_state: dict[str, Any]


def make_recovery_case(problem: Problem, env: PuzzleEnv) -> RecoveryCase | None:
    """Build a Derail case for ``problem``: pick an injection step, derail it, derive the checkpoint problem."""
    if len(problem.oracle_solution) < 3:
        return None
    injection_step = max(1, min(len(problem.oracle_solution) - 2, len(problem.oracle_solution) // 3))
    state = json.loads(json.dumps(problem.initial_state))
    for action in problem.oracle_solution[:injection_step]:
        result = env.step(state, action)
        if not result.valid:
            return None
        state = result.next_state
    oracle_next = problem.oracle_solution[injection_step]
    injected_action, injected_state, injection_type = _choose_injected_transition(env, state, oracle_next)
    suffix = problem.oracle_solution[injection_step:]
    checkpoint_problem = problem.model_copy(
        update={
            "problem_id": f"{problem.problem_id}_recovery_step{injection_step}",
            "initial_state": state,
            "oracle_solution": suffix,
            "min_steps": len(suffix),
            "max_steps": max(problem.max_steps, len(suffix) * 3),
            "natural_language_prompt": (
                f"Continue this {problem.environment} puzzle from the checkpoint state. "
                f"Goal state: {json.dumps(problem.goal_state, sort_keys=True)}"
            ),
        }
    )
    return RecoveryCase(
        problem=problem,
        checkpoint_problem=checkpoint_problem,
        injection_step=injection_step,
        injection_type=injection_type,
        last_valid_state=state,
        injected_action=injected_action,
        injected_state=injected_state,
    )


def run_recovery_condition(
    case: RecoveryCase,
    env: PuzzleEnv,
    client: ModelClient,
    condition: str,
    agent_config,
) -> RecoveryRecord:
    """Dispatch one Derail condition: prompted, stateguard rollback, or vex/repot agent."""
    if condition not in RECOVERY_CONDITIONS:
        raise ValueError(f"Unknown recovery condition: {condition}")
    if condition == "stateguard_rollback":
        return _run_stateguard_recovery(case, env, client, agent_config)
    if condition in {"vex_full", "vex_no_prefix", "vex_restart_from_initial"}:
        return _run_agent_recovery(
            case,
            env,
            client,
            agent_config,
            method="vex",
            disable_prefix=(condition == "vex_no_prefix"),
            restart_from_initial=(condition == "vex_restart_from_initial"),
            condition_name=condition,
        )
    if condition in {"repot_full", "repot_no_prefix", "repot_restart"}:
        return _run_agent_recovery(
            case,
            env,
            client,
            agent_config,
            method="repot",
            disable_prefix=(condition == "repot_no_prefix"),
            restart_from_initial=(condition == "repot_restart"),
            condition_name=condition,
        )
    return _run_prompted_recovery(case, env, client, condition, agent_config)


def _choose_injected_transition(
    env: PuzzleEnv,
    state: dict[str, Any],
    oracle_next: Action,
) -> tuple[Action, dict[str, Any], str]:
    for action in env.legal_actions(state):
        if action == oracle_next:
            continue
        result = env.step(state, action)
        if result.valid:
            return action, result.next_state, "legal_wrong_action"
    injected = {"type": "invalid_recovery_probe"}
    return injected, state, "invalid_action"


def _run_prompted_recovery(
    case: RecoveryCase,
    env: PuzzleEnv,
    client: ModelClient,
    condition: str,
    agent_config,
) -> RecoveryRecord:
    prompt = _recovery_prompt(case, env, condition)
    t0 = time.perf_counter()
    response = client.generate(
        GenerateRequest(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return a complete continuation as JSON exactly like "
                        '{"actions":[...]}. Do not use markdown.'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=agent_config.temperature,
            max_tokens=agent_config.max_tokens,
            metadata={
                "problem_id": case.problem.problem_id,
                "method": f"recovery_{condition}",
                "oracle_actions": case.checkpoint_problem.oracle_solution,
            },
        )
    )
    wall_time = time.perf_counter() - t0
    candidates = extract_and_replay_candidates(response.text, case.checkpoint_problem, env)
    chosen = _choose_candidate(candidates)
    success = bool(chosen and chosen.replay_success)
    failure = None if success else (chosen.failure_type if chosen else FailureType.OUTPUT_FORMAT_ERROR)
    return RecoveryRecord(
        problem_id=case.problem.problem_id,
        environment=case.problem.environment,
        complexity=case.problem.complexity,
        model=client.model,
        condition=condition,
        success=success,
        failure_type=failure,
        injection_step=case.injection_step,
        injection_type=case.injection_type,
        last_valid_state=case.last_valid_state,
        injected_action=case.injected_action,
        injected_state=case.injected_state,
        returned_action_count=len(chosen.actions) if chosen else 0,
        tokens_in=response.prompt_tokens,
        tokens_out=response.completion_tokens,
        wall_time=wall_time,
        raw_model_output=response.text,
        stopped_reason="goal" if success else "failed_recovery",
        metadata={
            "candidate_count": len(candidates),
            "condition_prompt_kind": condition,
        },
    )


def _run_stateguard_recovery(
    case: RecoveryCase,
    env: PuzzleEnv,
    client: ModelClient,
    agent_config,
) -> RecoveryRecord:
    from repot.core.agent import make_agent

    t0 = time.perf_counter()
    run = make_agent("stateguard_search").run(case.checkpoint_problem, env, client, agent_config)
    return RecoveryRecord(
        problem_id=case.problem.problem_id,
        environment=case.problem.environment,
        complexity=case.problem.complexity,
        model=client.model,
        condition="stateguard_rollback",
        success=run.success,
        failure_type=None if run.success else _failure_from_run(run),
        injection_step=case.injection_step,
        injection_type=case.injection_type,
        last_valid_state=case.last_valid_state,
        injected_action=case.injected_action,
        injected_state=case.injected_state,
        returned_action_count=sum(1 for step in run.steps if step.valid),
        tokens_in=run.tokens_in,
        tokens_out=run.tokens_out,
        wall_time=time.perf_counter() - t0,
        raw_model_output=run.steps[-1].raw_model_output if run.steps else "",
        stopped_reason=run.stopped_reason,
        metadata={
            "trace_steps": len(run.steps),
            "rollback_count": run.metadata.get("rollback_count", 0),
            "loop_recovery_attempts": run.metadata.get("loop_recovery_attempts", 0),
        },
    )


def _run_agent_recovery(
    case: RecoveryCase,
    env: PuzzleEnv,
    client: ModelClient,
    agent_config,
    *,
    method: str,
    disable_prefix: bool,
    restart_from_initial: bool = False,
    condition_name: str,
) -> RecoveryRecord:
    """Derail wedge: the *_full / *_no_prefix / *_restart triple, parameterized by method."""
    from repot.core.agent import make_agent

    cfg = replace(agent_config, vex_disable_prefix_in_prompt=disable_prefix)
    target_problem = case.problem if restart_from_initial else case.checkpoint_problem
    t0 = time.perf_counter()
    run = make_agent(method).run(target_problem, env, client, cfg)
    base_meta = {
        "trace_steps": len(run.steps),
        "method": method,
        "disable_prefix": disable_prefix,
        "restart_from_initial": restart_from_initial,
    }
    if method == "vex":
        base_meta.update(
            {
                "vex_llm_calls": run.metadata.get("vex_llm_calls", 0),
                "vex_rollback_count": run.metadata.get("vex_rollback_count", 0),
                "vex_committed_actions": run.metadata.get("vex_committed_actions", 0),
                "vex_proposer": run.metadata.get("vex_proposer", "unknown"),
            }
        )
    elif method == "repot":
        base_meta.update(
            {
                "repot_repair_calls": run.metadata.get("repot_repair_calls", 0),
                "repot_committed_actions": run.metadata.get("repot_committed_actions", 0),
                "repot_initial_pot_success": run.metadata.get("repot_initial_pot_success", False),
                "repot_first_failure_step": run.metadata.get("repot_first_failure_step"),
                "repot_initial_plan_len": run.metadata.get("repot_initial_plan_len", 0),
                "repot_initial_valid_prefix_fraction": run.metadata.get(
                    "repot_initial_valid_prefix_fraction", 0.0
                ),
            }
        )
    return RecoveryRecord(
        problem_id=case.problem.problem_id,
        environment=case.problem.environment,
        complexity=case.problem.complexity,
        model=client.model,
        condition=condition_name,
        success=run.success,
        failure_type=None if run.success else _failure_from_run(run),
        injection_step=case.injection_step,
        injection_type=case.injection_type,
        last_valid_state=case.last_valid_state,
        injected_action=case.injected_action,
        injected_state=case.injected_state,
        returned_action_count=sum(1 for step in run.steps if step.valid),
        tokens_in=run.tokens_in,
        tokens_out=run.tokens_out,
        wall_time=time.perf_counter() - t0,
        raw_model_output=run.steps[-1].raw_model_output if run.steps else "",
        stopped_reason=run.stopped_reason,
        metadata=base_meta,
    )


def _recovery_prompt(case: RecoveryCase, env: PuzzleEnv, condition: str) -> str:
    prefix = case.problem.oracle_solution[: case.injection_step]
    legal_actions = env.legal_actions(case.last_valid_state)
    base = (
        f"Puzzle prompt:\n{case.problem.natural_language_prompt}\n\n"
        f"Goal state:\n{json.dumps(case.problem.goal_state, sort_keys=True)}\n\n"
        f"Verified prefix moves already taken:\n{json.dumps(prefix, sort_keys=True)}\n\n"
        f"Then the trace contains this next move:\n{json.dumps(case.injected_action, sort_keys=True)}\n\n"
        f"The trace's current state is:\n{json.dumps(case.injected_state, sort_keys=True)}\n\n"
    )
    if condition == "no_feedback":
        return (
            base
            + "Continue from the trace's current state and return the remaining primitive moves as JSON."
        )
    if condition == "error_only":
        return (
            base
            + "The previous move or state was flagged as wrong, but no checkpoint is provided. "
            "Recover if possible and return the remaining primitive moves as JSON."
        )
    if condition == "state_feedback":
        return (
            base
            + "Rollback to this last valid checkpoint state before continuing:\n"
            + json.dumps(case.last_valid_state, sort_keys=True)
            + "\nReturn the remaining primitive moves as JSON."
        )
    if condition == "state_plus_legal_actions":
        return (
            base
            + "Rollback to this last valid checkpoint state before continuing:\n"
            + json.dumps(case.last_valid_state, sort_keys=True)
            + "\nLegal next actions from that checkpoint are:\n"
            + json.dumps(legal_actions, sort_keys=True)
            + "\nReturn the remaining primitive moves as JSON."
        )
    raise ValueError(f"Unsupported prompted recovery condition: {condition}")


def _choose_candidate(candidates):
    if not candidates:
        return None
    for candidate in candidates:
        if candidate.source == "final" and candidate.replay_success:
            return candidate
    for candidate in candidates:
        if candidate.replay_success:
            return candidate
    final_candidates = [candidate for candidate in candidates if candidate.source == "final"]
    return final_candidates[-1] if final_candidates else candidates[-1]


def _failure_from_run(run) -> FailureType:
    for step in run.steps:
        if step.error_type is not None:
            return step.error_type
    return FailureType.PREMATURE_STOP
