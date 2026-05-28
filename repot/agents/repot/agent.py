"""RePoT: Recoverable Program-of-Thought.

Algorithm 1 (paper):

    1. Run Program-of-Thought once.
    2. Replay the proposed moves through the verifier; commit moves[0:k]
       up to the first failure index k.
    3. If the plan reaches the goal, return.
    4. Otherwise repair the suffix from the verified checkpoint, conditioned
       on (verified_prefix_tail, current_state, goal, last_error). Bound to
       R = config.max_repair_calls calls.

Layout in this file mirrors the algorithm exactly:
- ``RePoTAgent.run`` is the loop.
- ``_pot_once`` is step 1 (initial PoT call).
- ``_suffix_repair_once`` is step 4 (one repair call).
- ``replay_until_failure`` (in ``repot.core.execution``) is step 2 / inner replay.

Trace-schema assembly (``CallTrace``, ``build_trace_run``) lives in
``repot.core.tracing`` so this file stays close to the algorithm.

The novelty over plain PoT is the *suffix-repair* prompt: it is conditioned
on the verifier-owned state at the failure point, so the model resumes from
a trusted checkpoint instead of restarting from the initial state.
"""

from __future__ import annotations

import time

from repot.core.schemas import Problem, State, TraceRun
from repot.core.env import PuzzleEnv
from repot.core.llm import GenerateRequest, ModelClient
from repot.core.tracing import CallTrace, build_trace_run, render_raw_output
from repot.core.agent import Agent, AgentConfig
from repot.core.parsing import parse_moves_with_fallback
from repot.core.prompts import code_prompt
from repot.core.execution import ReplayResult, replay_until_failure
from repot.core.execution import execute_python_code, extract_python_code


_POT_SYSTEM = (
    "Write Python code to solve the puzzle. Return only Python code, no markdown. "
    "The code must print one final line beginning with `moves = ` followed by the "
    "complete primitive move list in the requested format."
)

_REPAIR_SYSTEM = (
    "Write Python code to repair a partially verified puzzle plan. Return only "
    "Python code, no markdown. The code MUST print exactly one line beginning "
    "with `moves = ` followed by the suffix moves to apply *from the current "
    "verified state*. Do NOT restart from the initial state."
)


class RePoTAgent(Agent):
    """RePoT: Recoverable Program-of-Thought (Algorithm 1)."""

    method = "repot"

    def run(
        self,
        problem: Problem,
        env: PuzzleEnv,
        client: ModelClient,
        config: AgentConfig,
    ) -> TraceRun:
        """Run RePoT Algorithm 1: PoT once, verified replay, then up to R suffix-repair calls."""
        t0 = time.perf_counter()

        # Step 1 — one-shot Program-of-Thought.
        pot = _pot_once(problem, client, config)

        # Step 2 — deterministic verified replay.
        replay = replay_until_failure(env, problem, pot.actions)
        committed: list = list(replay.valid_prefix_actions)
        state: State = replay.final_state if replay.final_state is not None else problem.initial_state
        last_error: str = replay.error_message
        repair_calls: list[CallTrace] = []
        replays: list[ReplayResult] = [replay]

        if env.is_goal(state, problem.goal_state):
            return build_trace_run(
                env=env, problem=problem, client=client, config=config, t0=t0,
                pot=pot, repair_calls=repair_calls, committed=committed,
                replays=replays, success=True,
            )

        # Step 3 — suffix repair from the verified checkpoint.
        for _ in range(config.max_repair_calls):
            suffix_call = _suffix_repair_once(
                problem, env, client, config,
                verified_prefix=committed, state=state, last_error=last_error,
            )
            repair_calls.append(suffix_call)
            if not suffix_call.actions:
                # No usable suffix — surface the error but keep best-effort prefix.
                last_error = suffix_call.error or last_error
                continue

            replay = replay_until_failure(env, problem, suffix_call.actions, start_state=state)
            replays.append(replay)
            committed.extend(replay.valid_prefix_actions)
            state = replay.final_state if replay.final_state is not None else state
            last_error = replay.error_message
            if env.is_goal(state, problem.goal_state):
                return build_trace_run(
                    env=env, problem=problem, client=client, config=config, t0=t0,
                    pot=pot, repair_calls=repair_calls, committed=committed,
                    replays=replays, success=True,
                )

        return build_trace_run(
            env=env, problem=problem, client=client, config=config, t0=t0,
            pot=pot, repair_calls=repair_calls, committed=committed,
            replays=replays, success=False,
        )


# ----- Step 1 ---------------------------------------------------------------


def _pot_once(problem: Problem, client: ModelClient, config: AgentConfig) -> CallTrace:
    """Initial Program-of-Thought call. Same prompt as the PoT baseline so the
    headline comparison is apples-to-apples; only the *use* of the output
    differs (we replay deterministically instead of declaring success blindly).
    """
    user_prompt = (
        f"{problem.natural_language_prompt}\n\n"
        "Solve this by writing Python code. The program should compute the move sequence, "
        "then print exactly one line like:\n"
        "moves = [[...], [...]]\n"
        "Do not import unsafe modules, read files, use network, or call external programs."
    )
    return _emit_code_and_extract(
        client=client, config=config,
        system=_POT_SYSTEM, user=user_prompt,
        problem_id=problem.problem_id,
        method_label="repot:pot",
        extra_metadata={
            "program_of_thought": True,
            "oracle_actions": problem.oracle_solution,
        },
    )


# ----- Step 4 ---------------------------------------------------------------


def _suffix_repair_once(
    problem: Problem,
    env: PuzzleEnv,
    client: ModelClient,
    config: AgentConfig,
    *,
    verified_prefix: list,
    state: State,
    last_error: str,
) -> CallTrace:
    """One suffix-repair call. The model sees the verifier-owned current state,
    a tail of the committed prefix, the goal, and the last verifier error; it
    emits Python code that prints a `moves = [...]` suffix to apply from
    ``state``. Replay is the caller's responsibility.
    """
    chunk_size = max(1, problem.max_steps - len(verified_prefix))
    user_prompt = code_prompt(
        problem, env, state,
        verified_prefix,
        chunk_size=chunk_size,
        last_error=last_error,
        hide_prefix=config.vex_disable_prefix_in_prompt,
    )
    return _emit_code_and_extract(
        client=client, config=config,
        system=_REPAIR_SYSTEM, user=user_prompt,
        problem_id=problem.problem_id,
        method_label="repot:repair",
        extra_metadata={
            # Tell the OpenAIResponsesClient (openai_responses_client.py:77)
            # we are emitting Python code, not JSON. Without this the client
            # enforces text.format=json_object and the API rejects the
            # request because the prompt doesn't contain the word "json".
            "program_of_thought": True,
        },
    )


# ----- Sandbox + extraction (shared between Step 1 and Step 4) --------------


def _emit_code_and_extract(
    *,
    client: ModelClient,
    config: AgentConfig,
    system: str,
    user: str,
    problem_id: str,
    method_label: str,
    extra_metadata: dict | None = None,
) -> CallTrace:
    t0 = time.perf_counter()
    metadata = {"problem_id": problem_id, "method": method_label}
    if extra_metadata:
        metadata.update(extra_metadata)
    response = client.generate(
        GenerateRequest(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            metadata=metadata,
        )
    )
    elapsed = time.perf_counter() - t0
    code = extract_python_code(response.text)
    execution = execute_python_code(code)
    actions: list = []
    error = ""
    if not execution.ok:
        error = execution.error or "python execution failed"
    else:
        actions = parse_moves_with_fallback(execution.stdout, code)
        if not actions:
            error = "Python stdout did not contain a parseable moves list."
    return CallTrace(
        raw_text=render_raw_output(response.text, execution.stdout, execution.stderr),
        actions=actions,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        cached_tokens=getattr(response, "cached_tokens", 0) or 0,
        wall_time=elapsed,
        finish_reason=response.finish_reason,
        error=error,
    )
