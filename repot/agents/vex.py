"""VEX agent — wraps the chunked verified executor in the Agent / TraceRun interface.

VEX (Verified Execution): one-loop controller that proposes K moves, lets the
verifier commit/rollback, repeat. Two knobs: K per env, max_llm_calls.

Thin wrapper around ``repot.agents.repot.executor`` (the chunked verified
execution loop). Supplies:
- a Verifier shim around the project's PuzzleEnv,
- two proposer kinds:
  * "json": raw-action chunked plan (re-uses ChunkedPlanPolicy),
  * "code": Python-emitting proposer conditioned on the *verified prefix*
    plus current state — the v6 novelty: code resumes from a verified
    checkpoint instead of restarting from scratch (PoT cannot do this).
- TraceRun emission so traces line up with every other method's schema.
"""

from __future__ import annotations

import json
import time
from typing import Any

from repot.core.agent import Agent, AgentConfig
from repot.agents._chunked_plan import ChunkedPlanPolicy
from repot.agents.repot.executor import ChunkProposal, VEXExecutor, format_blocked
from repot.core.verifier import StepOutcome, Verifier
from repot.core.schemas import FailureType, Problem, TraceRun, TraceStep
from repot.core.env import PuzzleEnv
from repot.core.llm import GenerateRequest, ModelClient
from repot.core.parsing import parse_moves_with_fallback
from repot.core.prompts import code_prompt
from repot.core.execution import execute_python_code, extract_python_code


class _PuzzleEnvVerifier(Verifier):
    """Adapt a project ``PuzzleEnv`` to the public Verifier protocol."""

    def __init__(self, env: PuzzleEnv, problem: Problem) -> None:
        self.env = env
        self.problem = problem

    def legal_actions(self, state):
        """Delegate to the wrapped PuzzleEnv's `legal_actions`."""
        return self.env.legal_actions(state)

    def step(self, state, action):
        """Run env.step and convert the env's StepResult into a `StepOutcome`."""
        result = self.env.step(state, action)
        return StepOutcome(
            valid=result.valid,
            next_state=result.next_state,
            error_type=result.error_type.value if result.error_type else None,
            message=result.message,
        )

    def is_goal(self, state, goal):
        """Delegate to the wrapped PuzzleEnv's goal predicate."""
        return self.env.is_goal(state, goal)

    def progress_score(self, state, goal):
        """Delegate to the wrapped PuzzleEnv's progress score."""
        return self.env.progress_score(state, goal)

    def normalize(self, raw, state=None):
        """Delegate to the env's `normalize_candidate_move` to coerce a raw model action into canonical form."""
        return self.env.normalize_candidate_move(self.problem, raw, state)


class VEXAgent(Agent):
    """Verified-execution baseline (chunked verified plans).

    Reuses ``ChunkedPlanPolicy.propose_chunk`` for the LLM call (prompt already
    includes the verifier-grounded fields). The executor is the core loop.
    """

    method = "vex"

    def run(self, problem: Problem, env: PuzzleEnv, client: ModelClient, config: AgentConfig) -> TraceRun:
        """Drive the chunked verified-execution loop and project its result into a `TraceRun`."""
        t0 = time.perf_counter()
        chunk_size = config.resolve_chunk_size(problem.environment)
        proposer_kind = config.resolve_proposer(problem.environment)
        verifier = _PuzzleEnvVerifier(env, problem)
        proposals_log: list[dict[str, Any]] = []

        if proposer_kind == "code":
            proposer = _make_code_proposer(problem, env, client, config, proposals_log)
        else:
            proposer = _make_json_proposer(problem, env, client, config, proposals_log)

        executor = VEXExecutor(
            verifier=verifier,
            proposer=proposer,
            chunk_size=chunk_size,
            max_llm_calls=config.max_llm_calls,
        )
        result = executor.run(problem.initial_state, problem.goal_state)
        steps = _build_trace_steps(problem, client.model, result, proposals_log)
        for idx, step in enumerate(steps):
            step.step = idx

        wall_time = time.perf_counter() - t0
        cached_total = sum(p.get("cached_tokens", 0) for p in proposals_log)
        tokens_in_total = sum(p["tokens_in"] for p in proposals_log)
        return TraceRun(
            problem_id=problem.problem_id,
            environment=problem.environment,
            complexity=problem.complexity,
            model=client.model,
            method=self.method,
            success=result.success,
            stopped_reason=result.stopped_reason,
            steps=steps,
            final_state=result.final_state,
            tokens_in=tokens_in_total,
            tokens_out=sum(p["tokens_out"] for p in proposals_log),
            wall_time=wall_time,
            metadata={
                "vex_chunk_size": chunk_size,
                "vex_max_llm_calls": config.max_llm_calls,
                "vex_llm_calls": result.llm_calls,
                "vex_chunk_attempts": result.chunk_attempts,
                "vex_committed_actions": len(result.committed),
                "vex_rollback_count": result.rollback_count,
                "vex_rollback_rate": (
                    result.rollback_count / result.chunk_attempts if result.chunk_attempts else 0.0
                ),
                "vex_proposer": proposer_kind,
                "vex_cached_tokens_total": cached_total,
                "vex_cache_hit_rate": (cached_total / tokens_in_total) if tokens_in_total else 0.0,
                # Legacy key kept for downstream eval scripts that look for the
                # auto_v2 history field. With adaptive K reverted, this is a
                # single-element list of the constant chunk size.
                "auto_v2_chunk_size_history": [chunk_size],
            },
        )


def _make_json_proposer(problem, env, client, config, proposals_log):
    """Raw-action chunked proposer. Verifier-grounded prompt already includes
    legal_actions, current verified state, goal, strategy hint, last error."""
    policy = ChunkedPlanPolicy()

    def proposer(state, verified_prefix, chunk_size, last_error, blocked=()) -> ChunkProposal:
        """Run the JSON chunked policy and adapt its `PlanProposal` into a `ChunkProposal`."""
        proposal = policy.propose_chunk(
            problem=problem,
            env=env,
            state=state,
            client=client,
            config=config,
            retry_reason="verifier_rollback" if last_error else None,
            last_error=last_error,
            oracle_index=len(verified_prefix),
            chunk_size=chunk_size,
            blocked=list(blocked) if blocked else None,
        )
        proposals_log.append(_log_entry(list(proposal.actions), proposal.raw_model_output, proposal))
        return ChunkProposal(
            actions=list(proposal.actions),
            raw_text=proposal.raw_model_output,
            prompt_tokens=proposal.prompt_tokens,
            completion_tokens=proposal.completion_tokens,
            cached_tokens=getattr(proposal, "cached_tokens", 0) or 0,
            latency_s=proposal.latency_s,
            finish_reason=proposal.finish_reason,
            error=proposal.error,
        )

    return proposer


def _make_code_proposer(problem, env, client, config, proposals_log):
    """Python-emitting proposer conditioned on the verified prefix.

    The novelty: each call gives the LLM (verified_prefix, current_state,
    goal, last_error) and asks for code that produces moves *from the current
    state*, not from initial. PoT generates from initial every time. VEX's
    verifier commits the valid prefix; the next code call resumes — making
    long-horizon puzzles tractable across multiple LLM calls without losing
    the algorithmic structure code naturally captures.
    """

    def proposer(state, verified_prefix, chunk_size, last_error, blocked=()) -> ChunkProposal:
        """Emit Python conditioned on the verified prefix, execute it, and return parsed moves."""
        # Ablation hook: when config disables prefix conditioning, the prompt
        # sees an empty prefix and the "N verified moves" counter resets to 0.
        # Recovery harness uses this to isolate the prefix-conditioned-code
        # novelty from "VEX simply has more budget".
        prompt_prefix = [] if config.vex_disable_prefix_in_prompt else verified_prefix
        legal_summary = ChunkedPlanPolicy._legal_action_summary(env, state)
        blocked_summary = format_blocked(list(blocked) if blocked else [])
        prompt_user = code_prompt(
            problem,
            env,
            state,
            prompt_prefix,
            chunk_size,
            last_error,
            blocked=list(blocked) if blocked else [],
            hide_prefix=config.vex_disable_prefix_in_prompt,
            legal_summary=legal_summary,
            blocked_summary=blocked_summary,
        )
        response = client.generate(
            GenerateRequest(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Write Python code to solve a sub-segment of a verifiable puzzle. "
                            "Return only Python code, no markdown. The code MUST print exactly one "
                            "line beginning with `moves = ` followed by a Python list of next "
                            "primitive moves from the *given current state*. Do NOT include moves "
                            "already executed before the current state. Do not import unsafe modules."
                        ),
                    },
                    {"role": "user", "content": prompt_user},
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                metadata={
                    "problem_id": problem.problem_id,
                    "method": "vex",
                    "program_of_thought": True,
                    "vex_code_proposer": True,
                    "oracle_actions": problem.oracle_solution,
                    "oracle_index": len(verified_prefix),
                },
            )
        )
        code = extract_python_code(response.text)
        execution = execute_python_code(code)
        if not execution.ok:
            err = execution.error or "python execution failed"
            proposals_log.append(_log_entry([], response.text, response, error=err))
            return ChunkProposal(
                actions=[],
                raw_text=response.text,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                cached_tokens=getattr(response, "cached_tokens", 0) or 0,
                latency_s=response.latency_s,
                finish_reason=response.finish_reason,
                error=err,
            )
        raw_actions = parse_moves_with_fallback(execution.stdout, code)
        if not raw_actions:
            err = "Python stdout did not contain a parseable moves list."
            proposals_log.append(_log_entry([], response.text, response, error=err))
            return ChunkProposal(
                actions=[],
                raw_text=response.text,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                cached_tokens=getattr(response, "cached_tokens", 0) or 0,
                latency_s=response.latency_s,
                finish_reason=response.finish_reason,
                error=err,
            )
        proposals_log.append(_log_entry(list(raw_actions), response.text, response))
        return ChunkProposal(
            actions=list(raw_actions),
            raw_text=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cached_tokens=getattr(response, "cached_tokens", 0) or 0,
            latency_s=response.latency_s,
            finish_reason=response.finish_reason,
        )

    return proposer


def _log_entry(actions, raw_text, response_or_proposal, error: str = "") -> dict[str, Any]:
    return {
        "actions": list(actions),
        "raw_text": raw_text or "",
        "tokens_in": getattr(response_or_proposal, "prompt_tokens", 0) or 0,
        "tokens_out": getattr(response_or_proposal, "completion_tokens", 0) or 0,
        "cached_tokens": getattr(response_or_proposal, "cached_tokens", 0) or 0,
        "latency_s": getattr(response_or_proposal, "latency_s", 0.0) or 0.0,
        "finish_reason": getattr(response_or_proposal, "finish_reason", None),
        "error": error or getattr(response_or_proposal, "error", "") or "",
    }


def _build_trace_steps(
    problem: Problem,
    model: str,
    result,
    proposals_log: list[dict[str, Any]],
) -> list[TraceStep]:
    """Project the VEXResult into the project's per-action TraceStep schema.

    Each LLM proposal contributes one step boundary worth of token accounting;
    the committed and rolled-back actions inside that proposal become their
    own TraceSteps so the eval pipeline can compute valid_move_rate etc.
    """
    steps: list[TraceStep] = []
    consumed_committed: list = list(result.committed)
    consumed_rollbacks: list = list(result.rollbacks)
    # Reconstruct chunk-by-chunk: we consume committed actions until a rollback
    # matches the next chunk boundary. Safe because executor commits before
    # rolling back, and at most one rollback per chunk.
    committed_cursor = 0
    rollback_cursor = 0
    for chunk_idx, proposal_meta in enumerate(proposals_log):
        chunk_actions = proposal_meta.get("actions") or []
        if not chunk_actions:
            steps.append(
                TraceStep(
                    problem_id=problem.problem_id,
                    model=model,
                    method=VEXAgent.method,
                    step=0,
                    current_state=consumed_committed[committed_cursor - 1].post_state
                    if committed_cursor
                    else problem.initial_state,
                    valid=False,
                    error_type=FailureType.OUTPUT_FORMAT_ERROR,
                    message=proposal_meta.get("error") or "empty proposal",
                    tokens_in=proposal_meta["tokens_in"],
                    tokens_out=proposal_meta["tokens_out"],
                    wall_time=proposal_meta["latency_s"],
                    raw_model_output=proposal_meta["raw_text"],
                    finish_reason=proposal_meta["finish_reason"],
                    retry_reason="vex_format_error",
                )
            )
            continue
        # Greedily peel off committed actions matching this chunk; stop on
        # rollback.
        first_in_chunk = True
        for slot, _raw in enumerate(chunk_actions):
            if committed_cursor < len(consumed_committed):
                committed = consumed_committed[committed_cursor]
                # Heuristic: if a rollback exists and its pre_state equals the
                # state we'd be at NOW, the rollback belongs at this slot.
                if (
                    rollback_cursor < len(consumed_rollbacks)
                    and consumed_rollbacks[rollback_cursor].pre_state
                    == committed.pre_state
                ):
                    rb = consumed_rollbacks[rollback_cursor]
                    rollback_cursor += 1
                    steps.append(_rollback_step(problem, model, rb, proposal_meta, slot, first_in_chunk))
                    first_in_chunk = False
                    break
                steps.append(_commit_step(problem, model, committed, proposal_meta, slot, first_in_chunk))
                committed_cursor += 1
                first_in_chunk = False
            else:
                # No more committed; remaining slots are rollback-then-bail.
                if rollback_cursor < len(consumed_rollbacks):
                    rb = consumed_rollbacks[rollback_cursor]
                    rollback_cursor += 1
                    steps.append(_rollback_step(problem, model, rb, proposal_meta, slot, first_in_chunk))
                    first_in_chunk = False
                break
    return steps


def _commit_step(problem, model, committed, proposal_meta, slot, first_in_chunk) -> TraceStep:
    return TraceStep(
        problem_id=problem.problem_id,
        model=model,
        method=VEXAgent.method,
        step=0,
        current_state=committed.pre_state,
        model_action=committed.action,
        actual_next_state=committed.post_state,
        valid=True,
        tokens_in=proposal_meta["tokens_in"] if first_in_chunk else 0,
        tokens_out=proposal_meta["tokens_out"] if first_in_chunk else 0,
        wall_time=proposal_meta["latency_s"] if first_in_chunk else 0.0,
        raw_model_output=proposal_meta["raw_text"] if first_in_chunk else "",
        finish_reason=proposal_meta["finish_reason"] if first_in_chunk else None,
        retry_reason="vex_chunk",
    )


def _rollback_step(problem, model, rb, proposal_meta, slot, first_in_chunk) -> TraceStep:
    err = rb.error_type
    failure = (
        FailureType[err] if isinstance(err, str) and err in FailureType.__members__ else FailureType.INVALID_TRANSITION
    )
    return TraceStep(
        problem_id=problem.problem_id,
        model=model,
        method=VEXAgent.method,
        step=0,
        current_state=rb.pre_state,
        model_action=rb.action,
        actual_next_state=rb.pre_state,
        valid=False,
        error_type=failure,
        message=rb.message,
        tokens_in=proposal_meta["tokens_in"] if first_in_chunk else 0,
        tokens_out=proposal_meta["tokens_out"] if first_in_chunk else 0,
        wall_time=proposal_meta["latency_s"] if first_in_chunk else 0.0,
        raw_model_output=proposal_meta["raw_text"] if first_in_chunk else "",
        finish_reason=proposal_meta["finish_reason"] if first_in_chunk else None,
        retry_reason="vex_rollback",
    )


def _ensure_state(state):
    """Defensive deep copy via JSON; project states are JSON-friendly."""
    return json.loads(json.dumps(state))
