from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from repot.core.schemas import Action, FailureType, Problem, State, TraceRun, TraceStep
from repot.core.env import PuzzleEnv
from repot.core.llm import GenerateRequest, ModelClient
from repot.core.parsing import extract_actions, extract_json_object


@dataclass
class AgentConfig:
    """Shared runtime configuration for every Agent (budgets, per-env overrides, ablation flags)."""

    max_steps_multiplier: int = 2
    max_retries: int = 3
    self_consistency_k: int = 8
    max_state_visits: int = 4
    chunk_size: int = 4
    chunk_size_by_env: dict[str, int] | None = None
    max_llm_calls: int = 4
    max_repair_calls: int = 2
    chunk_max_tokens: int = 8192
    # Per-env override for chunk_max_tokens. Blocksworld/River usually need
    # ~2K, Hanoi/Checker the full ceiling. Tightening per-env stops the
    # reasoning model from burning hidden-reasoning budget on chunks that
    # only need a handful of tokens.
    chunk_max_tokens_by_env: dict[str, int] | None = None
    action_choice_max_tokens: int = 2048
    temperature: float = 0.0
    max_tokens: int = 16384
    # VEX/RePoT proposer routing. Either "json" (raw-action chunked policy)
    # or "code" (Python-emitting, conditioned on the verified prefix).
    # Defaults to "json" if unset / env not in table.
    vex_default_proposer: str = "json"
    vex_proposer_by_env: dict[str, str] | None = None
    # Derail ablation flag: when True, the code-prompt omits the
    # verified-prefix tail and the "you have already executed N verified
    # moves" line. Flipped on by the `repot_no_prefix` / `vex_no_prefix`
    # ablation conditions to A/B test whether prefix conditioning is the
    # load-bearing novelty.
    vex_disable_prefix_in_prompt: bool = False

    def resolve_chunk_size(self, env_name: str | None) -> int:
        """Return the per-env chunk size if set, otherwise the global `chunk_size`."""
        if env_name and self.chunk_size_by_env:
            override = self.chunk_size_by_env.get(env_name)
            if override is not None:
                return max(1, int(override))
        return max(1, int(self.chunk_size))

    def resolve_proposer(self, env_name: str | None) -> str:
        """Return the per-env VEX proposer kind if set, otherwise the default."""
        if env_name and self.vex_proposer_by_env:
            override = self.vex_proposer_by_env.get(env_name)
            if override:
                return str(override)
        return str(self.vex_default_proposer or "json")


class Agent(ABC):
    """Base class for every reasoning method. Subclasses implement ``run(problem, env, client, config)``."""

    method: str

    @abstractmethod
    def run(self, problem: Problem, env: PuzzleEnv, client: ModelClient, config: AgentConfig) -> TraceRun:
        """Execute the method on `problem` using `env` and `client`, returning a `TraceRun`."""
        raise NotImplementedError


class FullSolutionAgent(Agent):
    """Single-prompt baseline: ask the model for the full move list, replay it once. CoT lives here."""

    method = "cot"
    prompt_style = "plain"

    def run(self, problem: Problem, env: PuzzleEnv, client: ModelClient, config: AgentConfig) -> TraceRun:
        """Issue one prompt for the full action list and replay it through the env."""
        t0 = time.perf_counter()
        prompt = self._prompt(problem)
        response = client.generate(
            GenerateRequest(
                messages=[
                    {"role": "system", "content": _system_prompt(problem)},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                metadata={
                    "problem_id": problem.problem_id,
                    "method": self.method,
                    "oracle_actions": problem.oracle_solution,
                },
            )
        )
        try:
            actions = extract_actions(response.text)
        except Exception as exc:
            error_type = _format_failure_type(response.finish_reason)
            step = TraceStep(
                problem_id=problem.problem_id,
                model=client.model,
                method=self.method,
                step=0,
                current_state=problem.initial_state,
                valid=False,
                error_type=error_type,
                message=str(exc),
                tokens_in=response.prompt_tokens,
                tokens_out=response.completion_tokens,
                wall_time=response.latency_s,
                raw_model_output=response.text,
                finish_reason=response.finish_reason,
            )
            return self._trace_run(problem, client.model, False, "output_format_error", [step], problem.initial_state, time.perf_counter() - t0)

        steps, final_state, success, stopped = apply_actions(
            problem=problem,
            env=env,
            actions=actions,
            model=client.model,
            method=self.method,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            wall_time=response.latency_s,
            raw_model_output=response.text,
            finish_reason=response.finish_reason,
        )
        return self._trace_run(problem, client.model, success, stopped, steps, final_state, time.perf_counter() - t0)

    def _prompt(self, problem: Problem) -> str:
        extra = ""
        if self.prompt_style == "long":
            extra = "\nUse a larger reasoning budget. Carefully check every intermediate state before returning the answer."
        elif self.prompt_style == "algorithm":
            extra = "\nUse the known optimal or standard algorithm for this puzzle, but still return every primitive move."
        elif self.prompt_style == "state_table":
            extra = "\nFor each move, maintain an explicit state table before the final answer."
        if problem.metadata.get("puzzlezoo"):
            return f"{problem.natural_language_prompt}{extra}"
        return f"{problem.natural_language_prompt}\nReturn JSON exactly as {{\"actions\": [...]}}.{extra}"

    def _trace_run(
        self,
        problem: Problem,
        model: str,
        success: bool,
        stopped: str,
        steps: list[TraceStep],
        final_state: State,
        wall_time: float,
    ) -> TraceRun:
        return TraceRun(
            problem_id=problem.problem_id,
            environment=problem.environment,
            complexity=problem.complexity,
            model=model,
            method=self.method,
            success=success,
            stopped_reason=stopped,
            steps=steps,
            final_state=final_state,
            tokens_in=sum(s.tokens_in for s in steps),
            tokens_out=sum(s.tokens_out for s in steps),
            wall_time=wall_time,
        )


def apply_actions(
    problem: Problem,
    env: PuzzleEnv,
    actions: list[Action],
    model: str,
    method: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    wall_time: float = 0.0,
    raw_model_output: str = "",
    finish_reason: str | None = None,
) -> tuple[list[TraceStep], State, bool, str]:
    """Replay ``actions`` through ``env``, building TraceSteps; stop at first invalid or at goal."""
    state = json.loads(json.dumps(problem.initial_state))
    steps: list[TraceStep] = []
    max_steps = problem.max_steps
    for idx, raw_action in enumerate(actions[:max_steps]):
        try:
            action = env.normalize_candidate_move(problem, raw_action, state)
        except Exception as exc:
            step = TraceStep(
                problem_id=problem.problem_id,
                model=model,
                method=method,
                step=idx,
                current_state=state,
                model_action=raw_action if isinstance(raw_action, dict) else {"raw": raw_action},
                valid=False,
                error_type=FailureType.OUTPUT_FORMAT_ERROR,
                message=f"Could not normalize move: {exc}",
                tokens_in=prompt_tokens if idx == 0 else 0,
                tokens_out=completion_tokens if idx == 0 else 0,
                wall_time=wall_time if idx == 0 else 0.0,
                raw_model_output=raw_model_output if idx == 0 else "",
                finish_reason=finish_reason if idx == 0 else None,
            )
            steps.append(step)
            return steps, state, False, "output_format_error"
        result = env.step(state, action)
        step = TraceStep(
            problem_id=problem.problem_id,
            model=model,
            method=method,
            step=idx,
            current_state=state,
            model_action=action,
            actual_next_state=result.next_state if result.valid else state,
            valid=result.valid,
            error_type=result.error_type,
            message=result.message,
            tokens_in=prompt_tokens if idx == 0 else 0,
            tokens_out=completion_tokens if idx == 0 else 0,
            wall_time=wall_time if idx == 0 else 0.0,
            raw_model_output=raw_model_output if idx == 0 else "",
            finish_reason=finish_reason if idx == 0 else None,
        )
        steps.append(step)
        if not result.valid:
            return steps, state, False, "invalid_transition"
        state = result.next_state
        if env.is_goal(state, problem.goal_state):
            return steps, state, True, "goal"
    success = env.is_goal(state, problem.goal_state)
    return steps, state, success, "goal" if success else "max_steps_or_premature_stop"


def parse_action_choice(text: str) -> tuple[str, State | None, list[str]]:
    """Parse a single-action JSON choice from model output: returns (action_id, predicted_state, reasoning_lines)."""
    obj = extract_json_object(text)
    action_id = obj.get("action_id")
    if not isinstance(action_id, str) or not action_id:
        raise ValueError("response must contain a non-empty action_id string")
    predicted = obj.get("predicted_next_state")
    ranked = obj.get("ranked_action_ids", [])
    ranked_ids = [item for item in ranked if isinstance(item, str)] if isinstance(ranked, list) else []
    return action_id, predicted if isinstance(predicted, dict) else None, ranked_ids


JSON_ONLY_SYSTEM_PROMPT = (
    "You must respond with exactly one valid JSON object and no markdown, no prose, "
    "no code fences, and no hidden reasoning tags. "
    "Use double quotes for every key and string."
)

PUZZLEZOO_SYSTEM_PROMPT = (
    "You are solving a verifiable puzzle. Show concise work if useful, but make the final answer "
    "a single line beginning with moves = and containing the complete primitive move list."
)


def _system_prompt(problem: Problem) -> str:
    if problem.metadata.get("puzzlezoo"):
        return PUZZLEZOO_SYSTEM_PROMPT
    return JSON_ONLY_SYSTEM_PROMPT


def _format_failure_type(finish_reason: str | None) -> FailureType:
    if finish_reason in {"length", "incomplete", "max_output_tokens", "max_tokens"}:
        return FailureType.TOKEN_TRUNCATION
    return FailureType.OUTPUT_FORMAT_ERROR


def make_agent(method: str) -> Agent:
    """Factory: instantiate the Agent class for a method name (lazy-imports the impl module)."""
    if method == "cot":
        from repot.agents.cot import CoTAgent

        return CoTAgent()
    if method == "self_consistency":
        from repot.agents.self_consistency import SelfConsistencyAgent

        return SelfConsistencyAgent()
    if method == "program_of_thought":
        from repot.agents.pot import ProgramOfThoughtAgent

        return ProgramOfThoughtAgent()
    if method == "pot_retry":
        from repot.agents.pot import PoTRetryAgent

        return PoTRetryAgent()
    if method == "vex":
        from repot.agents.vex import VEXAgent

        return VEXAgent()
    if method == "repot":
        from repot.agents.repot.agent import RePoTAgent

        return RePoTAgent()
    if method == "repot_adaptive":
        from repot.agents.repot.adaptive import AdaptiveRePoTAgent

        return AdaptiveRePoTAgent()
    # Recovery-harness-only agents (not user-facing methods, used by Derail
    # for the `stateguard_rollback` baseline condition).
    if method == "stateguard_rollback":
        from repot.agents.rollback import RollbackAgent

        return RollbackAgent()
    if method == "stateguard_search":
        from repot.agents.rollback import StateGuardSearchAgent

        return StateGuardSearchAgent()
    raise ValueError(f"Unknown method: {method}")
