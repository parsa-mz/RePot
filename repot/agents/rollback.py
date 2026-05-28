from __future__ import annotations

import json
import time

from repot.core.agent import (
    JSON_ONLY_SYSTEM_PROMPT,
    Agent,
    AgentConfig,
    _format_failure_type,
    parse_action_choice,
)
from repot.core.schemas import ActionOption, FailureType, Problem, TraceRun, TraceStep
from repot.core.env import PuzzleEnv
from repot.core.llm import GenerateRequest, ModelClient


class RollbackAgent(Agent):
    """Step-by-step agent that proposes one action at a time and rolls back invalid moves."""
    method = "stateguard_rollback"
    use_strategy_hint = False
    use_search = False
    use_hanoi_policy = False

    def run(self, problem: Problem, env: PuzzleEnv, client: ModelClient, config: AgentConfig) -> TraceRun:
        """Loop one action per LLM call with per-state tabu blocking and loop-recovery retries."""
        t0 = time.perf_counter()
        state = json.loads(json.dumps(problem.initial_state))
        steps: list[TraceStep] = []
        retries = 0
        max_steps = problem.max_steps * config.max_steps_multiplier
        state_visits = {json.dumps(state, sort_keys=True): 1}
        blocked_by_state: dict[str, set[str]] = {}
        loop_recovery_used: set[str] = set()
        retry_reason: str | None = None
        last_error = ""

        while len(steps) < max_steps:
            if env.is_goal(state, problem.goal_state):
                return _trace_run(self.method, problem, client.model, True, "goal", steps, state, time.perf_counter() - t0)

            state_key = json.dumps(state, sort_keys=True)
            options = env.action_options(state)
            blocked = sorted(blocked_by_state.get(state_key, set()))
            controller_ranked_ids = self._controller_ranked_ids(problem, env, state, options)
            oracle_index = sum(1 for step in steps if step.valid)
            response = client.generate(
                GenerateRequest(
                    messages=[
                        {"role": "system", "content": JSON_ONLY_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": self._prompt(
                                problem=problem,
                                env=env,
                                state=state,
                                action_options=options,
                                blocked_action_ids=blocked,
                                retries=retries,
                                retry_reason=retry_reason,
                                last_error=last_error,
                            ),
                        },
                    ],
                    temperature=config.temperature,
                    max_tokens=min(config.action_choice_max_tokens, config.max_tokens),
                    metadata={
                        "problem_id": problem.problem_id,
                        "method": self.method,
                        "step": len(steps),
                        "one_action": True,
                        "oracle_actions": problem.oracle_solution,
                        "oracle_index": oracle_index,
                        "action_options": [option.model_dump(mode="json") for option in options],
                        "blocked_action_ids": blocked,
                        "strategy_hint": env.strategy_hint(problem, state) if self.use_strategy_hint else "",
                        "controller_ranked_action_ids": controller_ranked_ids,
                        "rank_actions": self.use_search,
                    },
                )
            )

            try:
                action_id, predicted, ranked_ids = parse_action_choice(response.text)
            except Exception as exc:
                steps.append(
                    self._error_step(
                        problem=problem,
                        client=client,
                        state=state,
                        options=options,
                        blocked=blocked,
                        retry_reason=retry_reason,
                        response=response,
                        error_type=_format_failure_type(response.finish_reason),
                        message=str(exc),
                    )
                )
                retries += 1
                retry_reason = "format_error"
                last_error = str(exc)
                if retries > config.max_retries:
                    return _trace_run(self.method, problem, client.model, False, "retry_budget_exhausted", steps, state, time.perf_counter() - t0)
                continue

            policy_choice = self._first_unblocked_controller_choice(controller_ranked_ids, blocked)
            known_action_ids = {item.id for item in options}
            if (
                self.use_hanoi_policy
                and policy_choice
                and action_id in known_action_ids
                and action_id not in set(blocked)
                and action_id != policy_choice
                and retries < config.max_retries
            ):
                message = (
                    f"Action id {action_id!r} does not match the Hanoi policy-ranked next action "
                    f"{policy_choice!r} from the current verified state."
                )
                steps.append(
                    self._error_step(
                        problem=problem,
                        client=client,
                        state=state,
                        options=options,
                        blocked=blocked,
                        retry_reason=retry_reason,
                        response=response,
                        error_type=FailureType.POLICY_DEVIATION,
                        message=message,
                        action_id=action_id,
                        predicted=predicted,
                    )
                )
                blocked_by_state.setdefault(state_key, set()).add(action_id)
                retries += 1
                retry_reason = "policy_deviation"
                last_error = message
                continue

            option = self._choose_option(
                action_id=action_id,
                ranked_ids=ranked_ids,
                options=options,
                blocked=blocked,
                state_visits=state_visits,
                state=state,
                env=env,
                controller_ranked_ids=controller_ranked_ids,
                force_controller_policy=self.use_hanoi_policy and policy_choice is not None and retries >= config.max_retries,
            )
            if option is None:
                known = {item.id for item in options}
                if action_id in blocked:
                    error_type = FailureType.REPETITION_LOOP
                    next_retry_reason = "loop_recovery"
                    message = f"Action id {action_id!r} is blocked because it repeats a failed or looping transition."
                else:
                    error_type = FailureType.UNKNOWN_ACTION_ID if action_id not in known else FailureType.OUTPUT_FORMAT_ERROR
                    next_retry_reason = "unknown_action_id"
                    message = f"Action id {action_id!r} is not selectable from the current legal action options."
                steps.append(
                    self._error_step(
                        problem=problem,
                        client=client,
                        state=state,
                        options=options,
                        blocked=blocked,
                        retry_reason=retry_reason,
                        response=response,
                        error_type=error_type,
                        message=message,
                        action_id=action_id,
                        predicted=predicted,
                    )
                )
                blocked_by_state.setdefault(state_key, set()).add(action_id)
                retries += 1
                retry_reason = next_retry_reason
                last_error = message
                if retries > config.max_retries:
                    stopped = "repetition_loop" if error_type == FailureType.REPETITION_LOOP else "retry_budget_exhausted"
                    return _trace_run(self.method, problem, client.model, False, stopped, steps, state, time.perf_counter() - t0)
                continue

            result = env.step(state, option.action)
            next_key = json.dumps(result.next_state, sort_keys=True) if result.valid else state_key
            would_loop = result.valid and next_key in state_visits and not env.is_goal(result.next_state, problem.goal_state)
            has_alternative = any(item.id not in set(blocked) | {option.id} for item in options)
            if would_loop and has_alternative and state_key not in loop_recovery_used:
                loop_recovery_used.add(state_key)
                blocked_by_state.setdefault(state_key, set()).add(option.id)
                step = self._step(
                    problem=problem,
                    client=client,
                    state=state,
                    option=option,
                    options=options,
                    blocked=sorted(blocked_by_state[state_key]),
                    retry_reason="loop_recovery",
                    response=response,
                    predicted=predicted,
                    actual_next_state=result.next_state,
                    valid=False,
                    error_type=FailureType.REPETITION_LOOP,
                    message="Action would return to a previously visited non-goal state; retrying from last verified state.",
                    loop_recovery_attempted=True,
                )
                steps.append(step)
                retries += 1
                retry_reason = "loop_recovery"
                last_error = step.message
                if retries > config.max_retries:
                    return _trace_run(self.method, problem, client.model, False, "repetition_loop", steps, state, time.perf_counter() - t0)
                continue

            steps.append(
                self._step(
                    problem=problem,
                    client=client,
                    state=state,
                    option=option,
                    options=options,
                    blocked=blocked,
                    retry_reason=retry_reason,
                    response=response,
                    predicted=predicted,
                    actual_next_state=result.next_state if result.valid else state,
                    valid=result.valid,
                    error_type=result.error_type,
                    message=result.message,
                )
            )
            if result.valid:
                state = result.next_state
                retries = 0
                retry_reason = None
                last_error = ""
                state_key = json.dumps(state, sort_keys=True)
                state_visits[state_key] = state_visits.get(state_key, 0) + 1
                if state_visits[state_key] >= config.max_state_visits and not env.is_goal(state, problem.goal_state):
                    return _trace_run(self.method, problem, client.model, False, "repetition_loop", steps, state, time.perf_counter() - t0)
            else:
                blocked_by_state.setdefault(state_key, set()).add(option.id)
                retries += 1
                retry_reason = "invalid_transition"
                last_error = result.message
                if retries > config.max_retries:
                    return _trace_run(self.method, problem, client.model, False, "retry_budget_exhausted", steps, state, time.perf_counter() - t0)

        success = env.is_goal(state, problem.goal_state)
        return _trace_run(self.method, problem, client.model, success, "goal" if success else "max_steps", steps, state, time.perf_counter() - t0)

    def _choose_option(
        self,
        action_id: str,
        ranked_ids: list[str],
        options: list[ActionOption],
        blocked: list[str],
        state_visits: dict[str, int],
        state: dict,
        env: PuzzleEnv,
        controller_ranked_ids: list[str] | None = None,
        force_controller_policy: bool = False,
    ) -> ActionOption | None:
        by_id = {option.id: option for option in options}
        blocked_ids = set(blocked)
        if force_controller_policy and controller_ranked_ids:
            for candidate_id in controller_ranked_ids:
                if candidate_id not in blocked_ids and candidate_id in by_id:
                    return by_id[candidate_id]
        if self.use_search and ranked_ids:
            for candidate_id in ranked_ids:
                option = by_id.get(candidate_id)
                if option is None or candidate_id in blocked_ids:
                    continue
                result = env.step(state, option.action)
                if not result.valid:
                    continue
                if json.dumps(result.next_state, sort_keys=True) not in state_visits:
                    return option
            for candidate_id in ranked_ids:
                option = by_id.get(candidate_id)
                if option is not None and candidate_id not in blocked_ids:
                    return option
        if action_id in blocked_ids:
            return None
        return by_id.get(action_id)

    def _controller_ranked_ids(
        self,
        problem: Problem,
        env: PuzzleEnv,
        state: dict,
        options: list[ActionOption],
    ) -> list[str]:
        # Hanoi-specific symbolic policy ranking was removed alongside
        # HanoiPolicyAgent. Stays as a no-op so the policy_deviation branch
        # in run() (guarded by `self.use_hanoi_policy=False` for all live
        # subclasses) remains unreachable but well-formed.
        return []

    def _first_unblocked_controller_choice(self, ranked_ids: list[str], blocked: list[str]) -> str | None:
        blocked_ids = set(blocked)
        for action_id in ranked_ids:
            if action_id not in blocked_ids:
                return action_id
        return None

    def _prompt(
        self,
        problem: Problem,
        env: PuzzleEnv,
        state: dict,
        action_options: list[ActionOption],
        blocked_action_ids: list[str],
        retries: int,
        retry_reason: str | None,
        last_error: str,
    ) -> str:
        repair = ""
        if retries:
            repair = (
                "\nThe transition checker did not accept the previous proposal. "
                "Continue only from the current valid state below and choose a different legal action id."
            )
        hint = env.strategy_hint(problem, state) if self.use_strategy_hint else ""
        controller_ranking = ""
        if self.use_hanoi_policy and problem.environment == "tower_of_hanoi":
            ranked_ids = self._controller_ranked_ids(problem, env, state, action_options)
            controller_ranking = (
                "\nHanoi policy-ranked action ids from best to worst: "
                f"{json.dumps(ranked_ids)}. Prefer the first available id."
            )
        ranking = (
            "Rank legal action ids from best to worst in ranked_action_ids, and put the selected best id in action_id. "
            if self.use_search
            else ""
        )
        return (
            f"{problem.natural_language_prompt}\n"
            f"Current valid state: {json.dumps(state, sort_keys=True)}\n"
            f"Goal state: {json.dumps(problem.goal_state, sort_keys=True)}\n"
            f"Legal action options: {json.dumps([item.model_dump(mode='json') for item in action_options], sort_keys=True)}\n"
            f"Blocked action ids for this state: {json.dumps(blocked_action_ids)}\n"
            f"Retry reason: {retry_reason or 'none'}\n"
            f"Verifier message: {last_error or 'none'}\n"
            f"Strategy hint: {hint or 'none'}\n"
            f"{controller_ranking}\n"
            "Choose exactly one action id from the legal action options. Do not invent ids. "
            f"{ranking}"
            "Return JSON exactly as {\"action_id\":\"a0\",\"predicted_next_state\":{...},\"rationale\":\"short\"}."
            f"{repair}"
        )

    def _error_step(
        self,
        problem: Problem,
        client: ModelClient,
        state: dict,
        options: list[ActionOption],
        blocked: list[str],
        retry_reason: str | None,
        response,
        error_type: FailureType,
        message: str,
        action_id: str | None = None,
        predicted: dict | None = None,
    ) -> TraceStep:
        return TraceStep(
            problem_id=problem.problem_id,
            model=client.model,
            method=self.method,
            step=0,
            current_state=state,
            action_id=action_id,
            legal_action_count=len(options),
            legal_action_ids=[item.id for item in options],
            blocked_action_ids=blocked,
            retry_reason=retry_reason,
            predicted_next_state=predicted,
            valid=False,
            error_type=error_type,
            message=message,
            tokens_in=response.prompt_tokens,
            tokens_out=response.completion_tokens,
            wall_time=response.latency_s,
            raw_model_output=response.text,
            finish_reason=response.finish_reason,
        )

    def _step(
        self,
        problem: Problem,
        client: ModelClient,
        state: dict,
        option: ActionOption,
        options: list[ActionOption],
        blocked: list[str],
        retry_reason: str | None,
        response,
        predicted: dict | None,
        actual_next_state: dict,
        valid: bool,
        error_type: FailureType | None,
        message: str,
        loop_recovery_attempted: bool = False,
    ) -> TraceStep:
        return TraceStep(
            problem_id=problem.problem_id,
            model=client.model,
            method=self.method,
            step=0,
            current_state=state,
            action_id=option.id,
            legal_action_count=len(options),
            legal_action_ids=[item.id for item in options],
            blocked_action_ids=blocked,
            retry_reason=retry_reason,
            model_action=option.action,
            predicted_next_state=predicted,
            actual_next_state=actual_next_state,
            valid=valid,
            error_type=error_type,
            message=message,
            tokens_in=response.prompt_tokens,
            tokens_out=response.completion_tokens,
            wall_time=response.latency_s,
            raw_model_output=response.text,
            finish_reason=response.finish_reason,
            loop_recovery_attempted=loop_recovery_attempted,
        )


class StateGuardSearchAgent(RollbackAgent):
    """RollbackAgent variant that asks the model to rank legal action ids and prefers fresh successors."""
    method = "stateguard_search"
    use_search = True


def _trace_run(
    method: str,
    problem: Problem,
    model: str,
    success: bool,
    stopped: str,
    steps: list[TraceStep],
    final_state: dict,
    wall_time: float,
) -> TraceRun:
    for idx, step in enumerate(steps):
        step.step = idx
    return TraceRun(
        problem_id=problem.problem_id,
        environment=problem.environment,
        complexity=problem.complexity,
        model=model,
        method=method,
        success=success,
        stopped_reason=stopped,
        steps=steps,
        final_state=final_state,
        tokens_in=sum(s.tokens_in for s in steps),
        tokens_out=sum(s.tokens_out for s in steps),
        wall_time=wall_time,
        metadata={
            "rollback_count": sum(1 for s in steps if not s.valid),
            "loop_recovery_attempts": sum(1 for s in steps if s.loop_recovery_attempted),
        },
    )
