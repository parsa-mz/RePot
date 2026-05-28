"""ChunkedPlanPolicy — VEX's JSON-emission proposer.

Used by `repot.agents.vex.VEXAgent` for state-tracking environments
(Blocksworld, River Crossing). The model is asked for up to `chunk_size`
primitive moves as a single JSON object `{"actions":[...]}`; the verifier
commits the valid prefix and rolls back any invalid suffix.

Only `ChunkedPlanPolicy`, `PlanProposal`, and `_extract_raw_action_list`
are public. Everything else in the legacy `policies.py` ancestor of this
file (RankedActionPolicy / ToolPlanPolicy / PolicySelector / Lookahead
/ HanoiMacro) was tied to the dropped legacy controllers and has been
removed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from repot.core.agent import AgentConfig
from repot.agents.repot.executor import format_blocked as _format_blocked
from repot.core.schemas import Action, Problem
from repot.core.env import PuzzleEnv
from repot.core.llm import GenerateRequest, ModelClient
from repot.core.parsing import extract_json_object


def _extract_raw_action_list(text: str) -> list:
    """Extract the actions array without filtering item shape.

    The chunked policy's _normalize_actions delegates per-item conversion to
    env.normalize_candidate_move, which accepts both dict and list/tuple
    encodings (e.g. Apple-array form ["A", from, to]). Filtering to dicts in
    extract_actions silently drops list-form items and yields an empty plan.
    """
    obj = extract_json_object(text)
    if "actions" in obj and isinstance(obj["actions"], list):
        return list(obj["actions"])
    if "action" in obj and isinstance(obj["action"], (dict, list, tuple)):
        return [obj["action"]]
    raise ValueError("response must contain action or actions")


@dataclass
class PlanProposal:
    """Bundle of normalized actions and the raw LLM call telemetry that produced them."""
    actions: list[Action]
    raw_model_output: str
    stdout: str
    stderr: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str | None = None
    error: str = ""
    program_executed: bool = False
    candidate_count: int = 0


class ChunkedPlanPolicy:
    """Proposer that asks the model for a JSON `{actions:[...]}` chunk and normalizes each move."""
    def propose_chunk(
        self,
        problem: Problem,
        env: PuzzleEnv,
        state: dict,
        client: ModelClient,
        config: AgentConfig,
        retry_reason: str | None = None,
        last_error: str = "",
        oracle_index: int = 0,
        chunk_size: int | None = None,
        blocked: list[tuple[Action, str]] | None = None,
    ) -> PlanProposal:
        """Prompt the LLM for up to `chunk_size` primitive moves and return them as a `PlanProposal`."""
        effective_chunk_size = max(1, int(chunk_size if chunk_size is not None else config.chunk_size))
        blocked = blocked or []
        env_name = getattr(problem, "environment", None)
        per_env_cap = None
        if config.chunk_max_tokens_by_env and env_name:
            per_env_cap = config.chunk_max_tokens_by_env.get(env_name)
        max_tokens = min(int(per_env_cap) if per_env_cap else config.chunk_max_tokens, config.max_tokens)
        response = client.generate(
            GenerateRequest(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You must respond with exactly one valid JSON object and no markdown. "
                            "Return only a short primitive action chunk."
                        ),
                    },
                    {
                        "role": "user",
                        "content": self._prompt(
                            problem,
                            env,
                            state,
                            effective_chunk_size,
                            retry_reason,
                            last_error,
                            blocked,
                        ),
                    },
                ],
                temperature=config.temperature,
                max_tokens=max_tokens,
                metadata={
                    "problem_id": problem.problem_id,
                    "method": "vex",
                    "step_policy": "chunked_state_plan",
                    "chunked_actions": True,
                    "chunk_size": effective_chunk_size,
                    "response_format": "action_list",
                    "oracle_actions": problem.oracle_solution,
                    "oracle_index": oracle_index,
                },
            )
        )
        try:
            raw_actions = _extract_raw_action_list(response.text)
            actions = self._normalize_actions(raw_actions, problem, env, state)
        except Exception as exc:
            return PlanProposal(
                actions=[],
                raw_model_output=response.text,
                stdout="",
                stderr="",
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                cached_tokens=int(getattr(response, "cached_tokens", 0) or 0),
                latency_s=response.latency_s,
                finish_reason=response.finish_reason,
                error=str(exc),
                program_executed=False,
            )
        return PlanProposal(
            actions=actions[:effective_chunk_size],
            raw_model_output=response.text,
            stdout="",
            stderr="",
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cached_tokens=int(getattr(response, "cached_tokens", 0) or 0),
            latency_s=response.latency_s,
            finish_reason=response.finish_reason,
            program_executed=False,
            candidate_count=1,
        )

    def _prompt(
        self,
        problem: Problem,
        env: PuzzleEnv,
        state: dict,
        chunk_size: int,
        retry_reason: str | None,
        last_error: str,
        blocked: list[tuple[Action, str]] | None = None,
    ) -> str:
        # Cache-friendly layout: every byte above the dashed marker is stable
        # within a single problem run. Providers (OpenAI, Anthropic, Gemini,
        # vLLM-prefix) can reuse the cached prefix across calls. Everything
        # below the marker depends on the verifier checkpoint and varies.
        stable_block = (
            f"{problem.natural_language_prompt}\n"
            f"Goal state: {json.dumps(problem.goal_state, sort_keys=True)}\n"
            f"Return up to {chunk_size} primitive moves from the current verified state as JSON exactly "
            "{\"actions\":[...]}. Do not include more than the requested number of moves. "
            "No markdown, no commentary.\n"
            "--- verifier checkpoint below ---\n"
        )
        legal_labels = self._legal_action_summary(env, state)
        blocked_block = _format_blocked(blocked or [])
        dynamic_block = (
            f"Current verified state: {json.dumps(state, sort_keys=True)}\n"
            f"Legal moves from this state: {legal_labels}\n"
            f"Blocked from this state (do not repeat): {blocked_block}\n"
            f"Retry reason: {retry_reason or 'none'}\n"
            f"Verifier message: {last_error or 'none'}\n"
            f"Strategy hint: {env.strategy_hint(problem, state) or 'none'}"
        )
        return stable_block + dynamic_block

    @staticmethod
    def _legal_action_summary(env: PuzzleEnv, state: dict, cap: int = 24) -> str:
        try:
            options = env.action_options(state)
        except Exception:
            return "unavailable"
        if not options:
            return "(none — terminal state)"
        labels = [f"{opt.id}={opt.label}" for opt in options[:cap]]
        more = "" if len(options) <= cap else f" (+{len(options) - cap} more)"
        return ", ".join(labels) + more

    def _normalize_actions(
        self,
        raw_actions: list[Action],
        problem: Problem,
        env: PuzzleEnv,
        state: dict,
    ) -> list[Action]:
        current = json.loads(json.dumps(state))
        normalized: list[Action] = []
        for raw in raw_actions:
            try:
                action = env.normalize_candidate_move(problem, raw, current)
            except Exception:
                if not isinstance(raw, dict):
                    raise
                action = dict(raw)
            normalized.append(action)
            result = env.step(current, action)
            if result.valid:
                current = result.next_state
        return normalized
