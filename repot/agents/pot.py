"""One-shot Program-of-Thought + the retry-on-failure baseline.

- ``ProgramOfThoughtAgent`` (method ``program_of_thought``) — emit one Python
  program, execute it, replay the printed move list once.
- ``PoTRetryAgent`` (method ``pot_retry``) — same as above, plus one
  independent retry on failure (fresh sample, no verified prefix, no error
  feedback). Total budget is 2 LLM calls in the worst case, matching RePoT's
  R=1 repair budget so the comparison is "second sample" vs "verified-prefix
  repair", not "more calls".
"""

from __future__ import annotations

import json
import time

from repot.core.agent import Agent, AgentConfig, apply_actions
from repot.core.env import PuzzleEnv
from repot.core.evaluation import extract_and_replay_candidates
from repot.core.execution import execute_python_code, extract_python_code
from repot.core.llm import GenerateRequest, ModelClient
from repot.core.schemas import FailureType, Problem, TraceRun, TraceStep


_SYSTEM_PROMPT = (
    "Write Python code to solve the puzzle. Return only Python code, no markdown. "
    "The code must print one final line beginning with moves = followed by the "
    "complete primitive move list in the requested format."
)


def _user_prompt(problem: Problem) -> str:
    return (
        f"{problem.natural_language_prompt}\n\n"
        "Solve this by writing Python code. The program should compute the move sequence, "
        "then print exactly one line like:\n"
        "moves = [[...], [...]]\n"
        "Do not import unsafe modules, read files, use network, or call external programs."
    )


# ---------------------------------------------------------------------------
# ProgramOfThoughtAgent — one-shot
# ---------------------------------------------------------------------------


class ProgramOfThoughtAgent(Agent):
    """One-shot Program-of-Thought: emit code, execute it, replay the printed moves."""

    method = "program_of_thought"

    def run(self, problem: Problem, env: PuzzleEnv, client: ModelClient, config: AgentConfig) -> TraceRun:
        """Generate one Python program, execute it, and replay the printed `moves =` list."""
        t0 = time.perf_counter()
        response = client.generate(
            GenerateRequest(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _user_prompt(problem)},
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                metadata={
                    "problem_id": problem.problem_id,
                    "method": self.method,
                    "program_of_thought": True,
                    "oracle_actions": problem.oracle_solution,
                },
            )
        )
        code = extract_python_code(response.text)
        execution = execute_python_code(code)
        raw_output = _raw_output(response.text, execution.stdout, execution.stderr)
        if not execution.ok:
            return _failure_run(
                problem=problem,
                model=client.model,
                method=self.method,
                error_type=FailureType.TOKEN_TRUNCATION if execution.timed_out else FailureType.OUTPUT_FORMAT_ERROR,
                message=execution.error,
                raw_output=raw_output,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                wall_time=time.perf_counter() - t0,
                finish_reason=response.finish_reason,
            )

        # Gemini-style models sometimes emit `moves = [...]` directly without
        # wrapping in print(); the sandbox runs that as an assignment with no
        # stdout. Scan both stdout and the model's emitted code.
        candidates_text = execution.stdout + ("\n" + code if code else "")
        candidates = extract_and_replay_candidates(candidates_text, problem, env)
        if not candidates:
            return _failure_run(
                problem=problem,
                model=client.model,
                method=self.method,
                error_type=FailureType.OUTPUT_FORMAT_ERROR,
                message="Python stdout did not contain a parseable moves list.",
                raw_output=raw_output,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                wall_time=time.perf_counter() - t0,
                finish_reason=response.finish_reason,
            )
        chosen = _choose_candidate(candidates)
        steps, final_state, success, stopped = apply_actions(
            problem=problem,
            env=env,
            actions=chosen.actions,
            model=client.model,
            method=self.method,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            wall_time=time.perf_counter() - t0,
            raw_model_output=raw_output,
            finish_reason=response.finish_reason,
        )
        return TraceRun(
            problem_id=problem.problem_id,
            environment=problem.environment,
            complexity=problem.complexity,
            model=client.model,
            method=self.method,
            success=success,
            stopped_reason=stopped,
            steps=steps,
            final_state=final_state,
            tokens_in=sum(step.tokens_in for step in steps),
            tokens_out=sum(step.tokens_out for step in steps),
            wall_time=time.perf_counter() - t0,
            metadata={
                "program_executed": True,
                "candidate_count": len(candidates),
                "chosen_candidate_id": chosen.candidate_id,
            },
        )


# ---------------------------------------------------------------------------
# PoTRetryAgent — one-shot + one independent retry (R=1 budget)
# ---------------------------------------------------------------------------


class PoTRetryAgent(Agent):
    """One-shot PoT plus one independent retry on failure (R=1 budget)."""

    method = "pot_retry"

    def run(self, problem: Problem, env: PuzzleEnv, client: ModelClient,
            config: AgentConfig) -> TraceRun:
        """Run one PoT attempt; on failure, try one independent fresh sample (R=1 budget)."""
        t_start = time.perf_counter()

        attempt1 = _one_pot_call(problem, env, client, config, attempt=1)
        if attempt1["success"]:
            return _retry_finalize(problem, env, client.model,
                                   [attempt1], chosen=attempt1, t_start=t_start,
                                   retry_used=False)

        attempt2 = _one_pot_call(problem, env, client, config, attempt=2)
        chosen = attempt2 if attempt2["success"] or \
                 attempt2["valid_steps"] >= attempt1["valid_steps"] else attempt1
        return _retry_finalize(problem, env, client.model,
                               [attempt1, attempt2], chosen=chosen, t_start=t_start,
                               retry_used=True)


def _one_pot_call(problem: Problem, env: PuzzleEnv, client: ModelClient,
                  config: AgentConfig, attempt: int) -> dict:
    response = client.generate(
        GenerateRequest(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(problem)},
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            metadata={
                "problem_id": problem.problem_id,
                "method": "pot_retry",
                "pot_retry_attempt": attempt,
                "program_of_thought": True,
                "oracle_actions": problem.oracle_solution,
            },
        )
    )
    code = extract_python_code(response.text)
    execution = execute_python_code(code)
    raw_output = (
        f"ATTEMPT {attempt} MODEL_CODE:\n{response.text}\n\n"
        f"STDOUT:\n{execution.stdout}\n\nSTDERR:\n{execution.stderr}"
    )
    base = {
        "raw_output": raw_output,
        "tokens_in": response.prompt_tokens,
        "tokens_out": response.completion_tokens,
        "finish_reason": response.finish_reason,
    }

    if not execution.ok:
        return {**base, "success": False, "valid_steps": 0, "actions": [],
                "error_type": FailureType.TOKEN_TRUNCATION if execution.timed_out
                              else FailureType.OUTPUT_FORMAT_ERROR,
                "message": execution.error}

    candidates_text = execution.stdout + ("\n" + code if code else "")
    candidates = extract_and_replay_candidates(candidates_text, problem, env)
    if not candidates:
        return {**base, "success": False, "valid_steps": 0, "actions": [],
                "error_type": FailureType.OUTPUT_FORMAT_ERROR,
                "message": "Python stdout did not contain a parseable moves list."}

    chosen = max(candidates,
                 key=lambda c: (int(c.replay_success), len(c.actions or [])))
    return {**base,
            "success": bool(chosen.replay_success),
            "valid_steps": len(chosen.actions or []),
            "actions": chosen.actions or [],
            "candidate_count": len(candidates),
            "chosen_candidate_id": chosen.candidate_id}


def _retry_finalize(problem: Problem, env: PuzzleEnv, model: str,
                    attempts: list[dict], chosen: dict, t_start: float,
                    retry_used: bool) -> TraceRun:
    total_tokens_in  = sum(a["tokens_in"]  for a in attempts)
    total_tokens_out = sum(a["tokens_out"] for a in attempts)

    if chosen.get("actions"):
        steps, final_state, success, stopped = apply_actions(
            problem=problem, env=env, actions=chosen["actions"],
            model=model, method="pot_retry",
            prompt_tokens=total_tokens_in,
            completion_tokens=total_tokens_out,
            wall_time=time.perf_counter() - t_start,
            raw_model_output=chosen["raw_output"],
            finish_reason=chosen["finish_reason"],
        )
    else:
        steps = [TraceStep(
            problem_id=problem.problem_id, model=model, method="pot_retry",
            step=0, current_state=problem.initial_state, valid=False,
            error_type=chosen.get("error_type", FailureType.OUTPUT_FORMAT_ERROR),
            message=chosen.get("message", "no plan parsed"),
            raw_model_output=chosen["raw_output"],
            finish_reason=chosen["finish_reason"],
            tokens_in=total_tokens_in, tokens_out=total_tokens_out,
            wall_time=time.perf_counter() - t_start,
        )]
        final_state = json.loads(json.dumps(problem.initial_state))
        success, stopped = False, "program_error"

    return TraceRun(
        problem_id=problem.problem_id,
        environment=problem.environment,
        complexity=problem.complexity,
        model=model, method="pot_retry",
        success=success, stopped_reason=stopped,
        steps=steps, final_state=final_state,
        tokens_in=total_tokens_in, tokens_out=total_tokens_out,
        wall_time=time.perf_counter() - t_start,
        metadata={
            "pot_retry_used": retry_used,
            "pot_retry_attempt1_success": attempts[0]["success"],
            "pot_retry_attempt2_success": attempts[1]["success"] if len(attempts) > 1 else None,
            "pot_retry_attempt1_valid_steps": attempts[0]["valid_steps"],
            "pot_retry_attempt2_valid_steps": attempts[1]["valid_steps"] if len(attempts) > 1 else None,
            "pot_retry_total_llm_calls": len(attempts),
        },
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _choose_candidate(candidates):
    for candidate in candidates:
        if candidate.replay_success:
            return candidate
    final_candidates = [candidate for candidate in candidates if candidate.source == "final"]
    if final_candidates:
        return final_candidates[-1]
    return candidates[-1]


def _failure_run(
    problem: Problem,
    model: str,
    method: str,
    error_type: FailureType,
    message: str,
    raw_output: str,
    prompt_tokens: int,
    completion_tokens: int,
    wall_time: float,
    finish_reason: str | None,
) -> TraceRun:
    step = TraceStep(
        problem_id=problem.problem_id,
        model=model,
        method=method,
        step=0,
        current_state=problem.initial_state,
        valid=False,
        error_type=error_type,
        message=message,
        raw_model_output=raw_output,
        finish_reason=finish_reason,
        tokens_in=prompt_tokens,
        tokens_out=completion_tokens,
        wall_time=wall_time,
    )
    return TraceRun(
        problem_id=problem.problem_id,
        environment=problem.environment,
        complexity=problem.complexity,
        model=model,
        method=method,
        success=False,
        stopped_reason="program_error",
        steps=[step],
        final_state=json.loads(json.dumps(problem.initial_state)),
        tokens_in=prompt_tokens,
        tokens_out=completion_tokens,
        wall_time=wall_time,
        metadata={"program_executed": False},
    )


def _raw_output(model_text: str, stdout: str, stderr: str) -> str:
    return (
        "MODEL_CODE:\n"
        f"{model_text}\n\n"
        "STDOUT:\n"
        f"{stdout}\n\n"
        "STDERR:\n"
        f"{stderr}"
    )
