"""Trace assembly for ``RePoTAgent``.

Projects (initial PoT call, repair calls, committed actions, replays) into the
project's TraceStep / TraceRun schema. Lives in its own module so
``repot/agents/repot/agent.py`` reads as Algorithm 1 — the algorithm code does
not need to know about TraceStep field shapes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from repot.core.schemas import FailureType, Problem, State, TraceRun, TraceStep
from repot.core.env import PuzzleEnv
from repot.core.llm import ModelClient
from repot.core.agent import AgentConfig
from repot.core.execution import ReplayResult


METHOD_LABEL = "repot"


@dataclass
class CallTrace:
    """One LLM call boundary — captured for trace accounting.

    The PoT call and each repair call produces one ``CallTrace``. The trace
    builder pairs each ``CallTrace`` with the ``ReplayResult`` it triggered
    (PoT always replays; a repair call only replays if it emitted parseable
    actions).
    """

    raw_text: str
    actions: list
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    wall_time: float
    finish_reason: str | None
    error: str = ""


def build_trace_run(
    *,
    env: PuzzleEnv,
    problem: Problem,
    client: ModelClient,
    config: AgentConfig,
    t0: float,
    pot: CallTrace,
    repair_calls: list[CallTrace],
    committed: list,
    replays: list[ReplayResult],
    success: bool,
) -> TraceRun:
    """Build the project-schema TraceRun for a RePoT execution.

    One TraceStep per committed action, with token accounting on the first
    step of each proposal's chunk; one failure TraceStep at the rollback
    boundary if the proposal triggered one. Mirrors VEX's accounting so
    trace-level metrics are directly comparable.
    """
    proposal_calls = [pot, *repair_calls]
    # Pair each proposal with the replay it produced. The PoT call always
    # produced replays[0]; later proposals only produce a replay if they
    # emitted parseable actions.
    proposals_with_replay: list[tuple[CallTrace, ReplayResult | None]] = []
    replay_iter = iter(replays)
    for prop in proposal_calls:
        if prop is pot or prop.actions:
            proposals_with_replay.append((prop, next(replay_iter, None)))
        else:
            proposals_with_replay.append((prop, None))

    steps: list[TraceStep] = []
    state: State = json.loads(json.dumps(problem.initial_state))

    for prop_i, (proposal, replay) in enumerate(proposals_with_replay):
        first_step_in_chunk = True
        chunk_actions = list(replay.valid_prefix_actions) if replay else []

        for action in chunk_actions:
            result = env.step(state, action)
            steps.append(
                TraceStep(
                    problem_id=problem.problem_id,
                    model=client.model,
                    method=METHOD_LABEL,
                    step=len(steps),
                    current_state=state,
                    model_action=action,
                    actual_next_state=result.next_state if result.valid else state,
                    valid=True,
                    tokens_in=proposal.prompt_tokens if first_step_in_chunk else 0,
                    tokens_out=proposal.completion_tokens if first_step_in_chunk else 0,
                    wall_time=proposal.wall_time if first_step_in_chunk else 0.0,
                    raw_model_output=proposal.raw_text if first_step_in_chunk else "",
                    finish_reason=proposal.finish_reason if first_step_in_chunk else None,
                )
            )
            state = result.next_state
            first_step_in_chunk = False

        if replay is not None and not replay.success and replay.first_failure_step is not None:
            steps.append(
                TraceStep(
                    problem_id=problem.problem_id,
                    model=client.model,
                    method=METHOD_LABEL,
                    step=len(steps),
                    current_state=state,
                    model_action=replay.invalid_action if isinstance(replay.invalid_action, dict) else None,
                    valid=False,
                    error_type=_failure_type_from_replay(replay),
                    message=replay.error_message,
                    tokens_in=proposal.prompt_tokens if first_step_in_chunk else 0,
                    tokens_out=proposal.completion_tokens if first_step_in_chunk else 0,
                    wall_time=proposal.wall_time if first_step_in_chunk else 0.0,
                    raw_model_output=proposal.raw_text if first_step_in_chunk else "",
                    finish_reason=proposal.finish_reason if first_step_in_chunk else None,
                    retry_reason=f"repot_proposal_{prop_i}_invalid",
                )
            )
        elif replay is None:
            # Proposal emitted no parseable actions; emit a format-error step.
            steps.append(
                TraceStep(
                    problem_id=problem.problem_id,
                    model=client.model,
                    method=METHOD_LABEL,
                    step=len(steps),
                    current_state=state,
                    valid=False,
                    error_type=FailureType.OUTPUT_FORMAT_ERROR,
                    message=proposal.error or "no parseable moves",
                    tokens_in=proposal.prompt_tokens,
                    tokens_out=proposal.completion_tokens,
                    wall_time=proposal.wall_time,
                    raw_model_output=proposal.raw_text,
                    finish_reason=proposal.finish_reason,
                    retry_reason=f"repot_proposal_{prop_i}_format_error",
                )
            )

    if not steps:
        steps.append(
            TraceStep(
                problem_id=problem.problem_id,
                model=client.model,
                method=METHOD_LABEL,
                step=0,
                current_state=problem.initial_state,
                valid=False,
                error_type=FailureType.OUTPUT_FORMAT_ERROR,
                message=pot.error or "no parseable moves",
                tokens_in=pot.prompt_tokens,
                tokens_out=pot.completion_tokens,
                wall_time=pot.wall_time,
                raw_model_output=pot.raw_text,
                finish_reason=pot.finish_reason,
            )
        )

    initial_replay = replays[0] if replays else None
    initial_pot_success = bool(
        initial_replay is not None
        and initial_replay.success
        and initial_replay.final_state is not None
        and env.is_goal(initial_replay.final_state, problem.goal_state)
    )

    return TraceRun(
        problem_id=problem.problem_id,
        environment=problem.environment,
        complexity=problem.complexity,
        model=client.model,
        method=METHOD_LABEL,
        success=success,
        stopped_reason="goal" if success else stopped_reason(repair_calls, replays, config),
        steps=steps,
        final_state=state,
        tokens_in=sum(s.tokens_in for s in steps),
        tokens_out=sum(s.tokens_out for s in steps),
        wall_time=time.perf_counter() - t0,
        metadata={
            "repot_initial_pot_success": initial_pot_success,
            "repot_repair_calls": len(repair_calls),
            "repot_committed_actions": len(committed),
            "repot_first_failure_step": initial_replay.first_failure_step if initial_replay else None,
            "repot_initial_plan_len": len(pot.actions),
            "repot_initial_valid_prefix_fraction": initial_replay.valid_prefix_fraction if initial_replay else 0.0,
            "repot_cached_tokens": sum(c.cached_tokens for c in proposal_calls),
            "repot_disable_prefix_in_prompt": bool(config.vex_disable_prefix_in_prompt),
        },
    )


def stopped_reason(
    repair_calls: list[CallTrace],
    replays: list[ReplayResult],
    config: AgentConfig,
) -> str:
    """Classify why the RePoT loop stopped (replay-failure, budget exhaustion, format error)."""
    if not repair_calls and replays and not replays[0].success:
        return "pot_replay_failed"
    if len(repair_calls) >= config.max_repair_calls:
        return "repair_budget_exhausted"
    if repair_calls and not repair_calls[-1].actions:
        return "repair_format_error"
    return "max_steps_or_premature_stop"


def render_raw_output(model_text: str, stdout: str, stderr: str) -> str:
    """Render the canonical MODEL_CODE/STDOUT/STDERR block stored in `TraceStep.raw_model_output`."""
    return f"MODEL_CODE:\n{model_text}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"


def _failure_type_from_replay(replay: ReplayResult) -> FailureType:
    if replay.error_type == "normalization_error":
        return FailureType.OUTPUT_FORMAT_ERROR
    return FailureType.INVALID_TRANSITION
