from __future__ import annotations

import json
from abc import ABC, abstractmethod

from repot.core.schemas import (
    Action,
    ActionOption,
    EnvCapabilities,
    FailureType,
    Problem,
    State,
    StepResult,
    TraceEvaluation,
    TraceStep,
)


class Env(ABC):
    """Abstract puzzle environment: instance generator, legal actions, deterministic step, goal check."""

    name: str

    @abstractmethod
    def generate(self, seed: int, complexity: int, template_id: int = 0) -> Problem:
        """Generate a `Problem` instance for the given seed/complexity/template."""
        raise NotImplementedError

    @abstractmethod
    def legal_actions(self, state: State) -> list[Action]:
        """Return all legal primitive actions from `state`."""
        raise NotImplementedError

    @abstractmethod
    def step(self, state: State, action: Action) -> StepResult:
        """Apply `action` to `state` and return a `StepResult` (validity, next state, error)."""
        raise NotImplementedError

    @abstractmethod
    def is_goal(self, state: State, goal: State) -> bool:
        """Return whether `state` satisfies the puzzle's goal predicate."""
        raise NotImplementedError

    @abstractmethod
    def oracle_solution(self, problem: Problem) -> list[Action]:
        """Return a known-valid solution sequence for `problem` (used as the dummy-client oracle)."""
        raise NotImplementedError

    @abstractmethod
    def render_prompt(self, problem: Problem, template_id: int) -> str:
        """Render the natural-language puzzle prompt for `problem` with template `template_id`."""
        raise NotImplementedError

    def generate_puzzlezoo(self, seed: int, complexity: int, template_id: int = 0) -> Problem:
        """Generate a problem and tag its metadata as part of the PuzzleZoo benchmark family."""
        problem = self.generate(seed=seed, complexity=complexity, template_id=template_id)
        metadata = dict(problem.metadata)
        metadata.update(
            {
                "benchmark_family": "puzzlezoo_core",
                "generator_version": "puzzlezoo_v1",
                "puzzlezoo": True,
            }
        )
        return problem.model_copy(update={"metadata": metadata})

    def normalize_candidate_move(self, problem: Problem, raw_move, state: State | None = None) -> Action:
        """Default: return raw_move if it is already a dict, else raise; envs override per-format."""
        del problem, state
        if isinstance(raw_move, dict):
            return raw_move
        raise ValueError(f"Cannot normalize candidate move: {raw_move!r}")

    def action_options(self, state: State) -> list[ActionOption]:
        """Wrap each legal action in an `ActionOption` with a stable `aN` id and human label."""
        return [
            ActionOption(id=f"a{idx}", action=action, label=self.action_label(action))
            for idx, action in enumerate(self.legal_actions(state))
        ]

    def action_label(self, action: Action) -> str:
        """Default human-readable label for `action`: a stable JSON serialization."""
        return json.dumps(action, sort_keys=True, separators=(",", ":"))

    def capabilities(self, problem: Problem) -> EnvCapabilities:
        """Return the env-feature flags that the harness should respect for `problem` (default: empty)."""
        del problem
        return EnvCapabilities()

    def strategy_hint(self, problem: Problem, state: State) -> str:
        """Return a one-line strategy hint for the agent prompt (default: empty)."""
        del problem, state
        return ""

    def progress_score(self, state: State, goal: State) -> float:
        """Default progress: 1.0 if goal reached else 0.0; envs override with smoother metrics."""
        return 1.0 if self.is_goal(state, goal) else 0.0

    def verify_trace(self, problem: Problem, trace: list[TraceStep]) -> TraceEvaluation:
        """Replay `trace` and aggregate success, failure-type, drift, and adherence metrics."""
        state = json.loads(json.dumps(problem.initial_state))
        valid_count = 0
        drift_count = 0
        constraint_count = 0
        first_invalid: int | None = None
        first_failure: int | None = None
        failure_type: FailureType | None = None
        seen = {json.dumps(state, sort_keys=True)}
        repeated = 0

        for item in trace:
            result = self.step(state, item.model_action or {}) if item.valid else StepResult(
                valid=False,
                next_state=state,
                error_type=item.error_type or FailureType.INVALID_TRANSITION,
                message=item.message,
            )
            if item.valid and result.valid:
                valid_count += 1
                if item.predicted_next_state is not None and item.predicted_next_state != result.next_state:
                    drift_count += 1
                    if failure_type is None:
                        failure_type = FailureType.STATE_DRIFT
                        first_failure = item.step
                state = result.next_state
                state_key = json.dumps(state, sort_keys=True)
                if state_key in seen:
                    repeated += 1
                seen.add(state_key)
            else:
                if first_invalid is None:
                    first_invalid = item.step
                if failure_type is None:
                    failure_type = item.error_type or result.error_type or FailureType.INVALID_TRANSITION
                    first_failure = item.step
                if (item.error_type or result.error_type) == FailureType.CONSTRAINT_VIOLATION:
                    constraint_count += 1

        success = self.is_goal(state, problem.goal_state)
        if not success and failure_type is None:
            failure_type = FailureType.PREMATURE_STOP if trace else FailureType.OUTPUT_FORMAT_ERROR
            first_failure = trace[-1].step if trace else 0

        total = len(trace)
        tokens_total = sum(s.tokens_in + s.tokens_out for s in trace)
        return TraceEvaluation(
            problem_id=problem.problem_id,
            environment=problem.environment,
            complexity=problem.complexity,
            success=success,
            failure_type=None if success else failure_type,
            first_failure_step=None if success else first_failure,
            valid_move_rate=valid_count / total if total else 0.0,
            first_invalid_step=first_invalid,
            state_drift_rate=drift_count / total if total else 0.0,
            constraint_violation_rate=constraint_count / total if total else 0.0,
            repeated_state_count=repeated,
            premature_stop=not success and failure_type == FailureType.PREMATURE_STOP,
            legal_action_adherence_rate=_legal_action_adherence_rate(trace),
            unknown_action_id_rate=_unknown_action_id_rate(trace),
            invalid_after_resolution_rate=_invalid_after_resolution_rate(trace),
            loop_entry_rate=repeated / total if total else 0.0,
            loop_recovery_success_rate=_loop_recovery_success_rate(trace, success),
            state_drift_detected_rate=drift_count / total if total else 0.0,
            state_drift_corrected_rate=_state_drift_corrected_rate(trace, success),
            normalized_first_failure_step=(first_failure / problem.min_steps)
            if first_failure is not None and problem.min_steps
            else None,
            solution_length_ratio=(valid_count / problem.min_steps) if problem.min_steps else None,
            parse_schema_success_rate=_parse_schema_success_rate(trace),
            tokens_total=tokens_total,
            tokens_per_valid_step=tokens_total / valid_count if valid_count else None,
            explanation="" if success else f"Primary failure: {failure_type}",
        )


def _legal_action_adherence_rate(trace: list[TraceStep]) -> float:
    candidates = [s for s in trace if s.legal_action_ids]
    if not candidates:
        return 0.0
    return sum(1 for s in candidates if s.action_id in s.legal_action_ids) / len(candidates)


def _unknown_action_id_rate(trace: list[TraceStep]) -> float:
    candidates = [s for s in trace if s.action_id is not None or s.legal_action_ids]
    if not candidates:
        return 0.0
    return sum(1 for s in candidates if s.error_type == FailureType.UNKNOWN_ACTION_ID) / len(candidates)


def _invalid_after_resolution_rate(trace: list[TraceStep]) -> float:
    resolved = [
        s
        for s in trace
        if s.action_id is not None
        and s.legal_action_ids
        and s.action_id in s.legal_action_ids
        and s.error_type != FailureType.REPETITION_LOOP
    ]
    if not resolved:
        return 0.0
    return sum(1 for s in resolved if not s.valid) / len(resolved)


def _loop_recovery_success_rate(trace: list[TraceStep], success: bool) -> float:
    attempts = [idx for idx, s in enumerate(trace) if s.loop_recovery_attempted]
    if not attempts:
        return 0.0
    recovered = 0
    for idx in attempts:
        if success or any(s.valid and s.error_type is None for s in trace[idx + 1 :]):
            recovered += 1
    return recovered / len(attempts)


def _state_drift_corrected_rate(trace: list[TraceStep], success: bool) -> float:
    drift_indices = [
        idx
        for idx, s in enumerate(trace)
        if s.predicted_next_state is not None
        and s.actual_next_state is not None
        and s.predicted_next_state != s.actual_next_state
    ]
    if not drift_indices:
        return 0.0
    corrected = 0
    for idx in drift_indices:
        if success or any(s.valid for s in trace[idx + 1 :]):
            corrected += 1
    return corrected / len(drift_indices)


def _parse_schema_success_rate(trace: list[TraceStep]) -> float:
    if not trace:
        return 0.0
    bad = {
        FailureType.OUTPUT_FORMAT_ERROR,
        FailureType.TOKEN_TRUNCATION,
        FailureType.UNKNOWN_ACTION_ID,
    }
    return sum(1 for s in trace if s.error_type not in bad) / len(trace)


PuzzleEnv = Env
