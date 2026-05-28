"""Adaptive RePoT: rule-based dispatcher between suffix-repair and fresh-retry.

Algorithm:

    1. Run PoT once.
    2. Verified replay → (verified_prefix, state, error, prefix_fraction).
    3. Goal already? → return success in 1 LLM call.
    4. Policy decision:
         - empty plan (len == 0)            → fresh PoT retry
         - very-short prefix (frac < 0.15)  → fresh PoT retry
         - otherwise                        → suffix repair (standard RePoT)
    5. Spend the second LLM call accordingly.
    6. Return.

The total LLM-call budget is 2, matching ``PoTRetryAgent`` and ``RePoTAgent``
(both R=1). The contribution over standard RePoT is the policy decision in
step 4: when the verified prefix is empty or misleading, anchoring on it is
worse than a fresh independent sample.

This module reuses the same ``_pot_once`` and ``_suffix_repair_once``
primitives as ``repot.agents.repot.agent``, so the underlying prompts and
sandboxing match exactly. The only difference is the dispatcher in step 4.
"""

from __future__ import annotations

import json
import time

from repot.core.schemas import FailureType, Problem, TraceRun, TraceStep
from repot.core.env import PuzzleEnv
from repot.core.llm import ModelClient
from repot.core.agent import Agent, AgentConfig, apply_actions
from repot.agents.repot.agent import _pot_once, _suffix_repair_once
from repot.core.tracing import CallTrace
from repot.core.execution import replay_until_failure


# Policy thresholds. Tuned conservatively on inspection of the existing
# RePoT-vs-PoT-retry head-to-head: empty plans and very-short prefixes are
# where the verified-prefix anchor is most likely to mislead the repair call.
EMPTY_PLAN_THRESHOLD = 0  # len(plan) == 0
SHORT_PREFIX_FRACTION = 0.15  # prefix < 15% of plan length

METHOD_LABEL = "repot_adaptive"


class AdaptiveRePoTAgent(Agent):
    """RePoT with a rule-based recovery-policy dispatcher (R=1 budget)."""

    method = METHOD_LABEL

    def run(
        self,
        problem: Problem,
        env: PuzzleEnv,
        client: ModelClient,
        config: AgentConfig,
    ) -> TraceRun:
        """Run PoT once, then dispatch a fresh retry or suffix repair based on the verified prefix."""
        t0 = time.perf_counter()

        # Step 1 — initial PoT call.
        pot1 = _pot_once(problem, client, config)

        # Step 2 — verified replay.
        replay1 = replay_until_failure(env, problem, pot1.actions)
        committed = list(replay1.valid_prefix_actions)
        state = replay1.final_state if replay1.final_state is not None else problem.initial_state
        prefix_frac = replay1.valid_prefix_fraction
        plan_len = len(pot1.actions)

        # Step 3 — initial PoT already reached the goal.
        if env.is_goal(state, problem.goal_state):
            return _finalize(
                problem=problem, env=env, model=client.model,
                committed=committed, success=True,
                t0=t0, pot_calls=[pot1], second_call=None,
                policy="initial_pot_success",
                prefix_frac=prefix_frac, plan_len=plan_len,
            )

        # Step 4 — policy decision.
        if plan_len == EMPTY_PLAN_THRESHOLD:
            policy = "fresh_retry_empty_plan"
        elif prefix_frac < SHORT_PREFIX_FRACTION:
            policy = "fresh_retry_short_prefix"
        else:
            policy = "suffix_repair"

        if config.max_repair_calls < 1:
            return _finalize(
                problem=problem, env=env, model=client.model,
                committed=committed, success=False,
                t0=t0, pot_calls=[pot1], second_call=None,
                policy=f"{policy}_no_budget",
                prefix_frac=prefix_frac, plan_len=plan_len,
            )

        # Step 5 — spend the second LLM call.
        if policy.startswith("fresh_retry"):
            pot2 = _pot_once(problem, client, config)
            replay2 = replay_until_failure(env, problem, pot2.actions)
            # Pick the better of the two attempts (by success then valid steps),
            # mirroring PoTRetryAgent's choice logic.
            attempt1_valid = len(replay1.valid_prefix_actions)
            attempt2_valid = len(replay2.valid_prefix_actions)
            picked_second = (
                replay2.success
                or (not replay1.success and attempt2_valid >= attempt1_valid)
            )
            if picked_second:
                committed = list(replay2.valid_prefix_actions)
                state = replay2.final_state if replay2.final_state is not None else problem.initial_state
            # else: keep committed/state from attempt 1
            success = env.is_goal(state, problem.goal_state)
            return _finalize(
                problem=problem, env=env, model=client.model,
                committed=committed, success=success,
                t0=t0, pot_calls=[pot1, pot2], second_call="fresh_retry",
                policy=policy, prefix_frac=prefix_frac, plan_len=plan_len,
                picked_attempt=(2 if picked_second else 1),
            )

        # policy == "suffix_repair" — standard RePoT branch.
        repair = _suffix_repair_once(
            problem, env, client, config,
            verified_prefix=committed, state=state,
            last_error=replay1.error_message,
        )
        if repair.actions:
            replay_repair = replay_until_failure(
                env, problem, repair.actions, start_state=state,
            )
            committed.extend(replay_repair.valid_prefix_actions)
            state = replay_repair.final_state if replay_repair.final_state is not None else state
        success = env.is_goal(state, problem.goal_state)
        return _finalize(
            problem=problem, env=env, model=client.model,
            committed=committed, success=success,
            t0=t0, pot_calls=[pot1], second_call="suffix_repair",
            repair_call=repair, policy=policy,
            prefix_frac=prefix_frac, plan_len=plan_len,
        )


def _finalize(
    *,
    problem: Problem,
    env: PuzzleEnv,
    model: str,
    committed: list,
    success: bool,
    t0: float,
    pot_calls: list[CallTrace],
    second_call: str | None,
    policy: str,
    prefix_frac: float,
    plan_len: int,
    repair_call: CallTrace | None = None,
    picked_attempt: int | None = None,
) -> TraceRun:
    all_calls = list(pot_calls) + ([repair_call] if repair_call else [])
    tokens_in = sum(c.prompt_tokens for c in all_calls)
    tokens_out = sum(c.completion_tokens for c in all_calls)
    cached = sum(c.cached_tokens for c in all_calls)
    raw_output = "\n\n=== ADAPTIVE BOUNDARY ===\n\n".join(c.raw_text for c in all_calls)
    finish_reason = all_calls[-1].finish_reason if all_calls else None

    if committed:
        steps, final_state, success_replay, stopped = apply_actions(
            problem=problem, env=env, actions=committed,
            model=model, method=METHOD_LABEL,
            prompt_tokens=tokens_in, completion_tokens=tokens_out,
            wall_time=time.perf_counter() - t0,
            raw_model_output=raw_output, finish_reason=finish_reason,
        )
        # Trust our success determination; apply_actions only knows about
        # the committed list, not the goal predicate context.
        success = success or success_replay
    else:
        steps = [TraceStep(
            problem_id=problem.problem_id, model=model, method=METHOD_LABEL,
            step=0, current_state=problem.initial_state, valid=False,
            error_type=FailureType.OUTPUT_FORMAT_ERROR,
            message="no committed actions",
            raw_model_output=raw_output, finish_reason=finish_reason,
            tokens_in=tokens_in, tokens_out=tokens_out,
            wall_time=time.perf_counter() - t0,
        )]
        final_state = json.loads(json.dumps(problem.initial_state))
        stopped = "program_error"

    return TraceRun(
        problem_id=problem.problem_id,
        environment=problem.environment,
        complexity=problem.complexity,
        model=model, method=METHOD_LABEL,
        success=success,
        stopped_reason="goal" if success else stopped,
        steps=steps, final_state=final_state,
        tokens_in=tokens_in, tokens_out=tokens_out,
        wall_time=time.perf_counter() - t0,
        metadata={
            "repot_adaptive_policy": policy,
            "repot_adaptive_second_call": second_call,
            "repot_adaptive_picked_attempt": picked_attempt,
            "repot_adaptive_initial_prefix_fraction": prefix_frac,
            "repot_adaptive_initial_plan_len": plan_len,
            "repot_adaptive_committed_actions": len(committed),
            "repot_adaptive_total_llm_calls": len(all_calls),
            "repot_adaptive_cached_tokens": cached,
            # Mirror RePoT/PoT-retry metadata names where it makes sense, so
            # downstream analysis scripts can treat adaptive uniformly.
            "repot_initial_pot_success": policy == "initial_pot_success",
            "repot_initial_plan_len": plan_len,
            "repot_initial_valid_prefix_fraction": prefix_frac,
            "repot_committed_actions": len(committed),
        },
    )
