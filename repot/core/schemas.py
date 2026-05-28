from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

State = dict[str, Any]
Action = dict[str, Any]


class FailureType(StrEnum):
    """Canonical taxonomy of trace failure modes used across envs, agents, and the judge."""

    INVALID_TRANSITION = "INVALID_TRANSITION"
    STATE_DRIFT = "STATE_DRIFT"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    REPETITION_LOOP = "REPETITION_LOOP"
    PREMATURE_STOP = "PREMATURE_STOP"
    WRONG_GOAL = "WRONG_GOAL"
    OUTPUT_FORMAT_ERROR = "OUTPUT_FORMAT_ERROR"
    TOKEN_TRUNCATION = "TOKEN_TRUNCATION"
    UNKNOWN_ACTION_ID = "UNKNOWN_ACTION_ID"
    POLICY_DEVIATION = "POLICY_DEVIATION"
    PROVIDER_REJECTION = "PROVIDER_REJECTION"
    API_ERROR = "API_ERROR"
    VERIFIER_DISAGREEMENT = "VERIFIER_DISAGREEMENT"
    NO_RECOVERY_AFTER_ERROR = "NO_RECOVERY_AFTER_ERROR"


class Problem(BaseModel):
    """One puzzle instance: env id, complexity, initial + goal state, oracle plan, prompt."""

    model_config = ConfigDict(extra="forbid")

    problem_id: str
    environment: str
    complexity: int
    initial_state: State
    goal_state: State
    natural_language_prompt: str
    oracle_solution: list[Action] = Field(default_factory=list)
    min_steps: int | None = None
    max_steps: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    """Outcome of ``Env.step(state, action)``: valid? next state? what went wrong?"""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    next_state: State
    error_type: FailureType | None = None
    message: str = ""


class ActionOption(BaseModel):
    """One enumerated legal action: stable id, the action dict, and a display label."""

    model_config = ConfigDict(extra="forbid")

    id: str
    action: Action
    label: str


class EnvCapabilities(BaseModel):
    """Per-env feature flags consumed by agents/runner (action-ids, progress score, PoT-style tool plan)."""

    model_config = ConfigDict(extra="forbid")

    supports_action_ids: bool = True
    supports_progress_score: bool = True
    supports_tool_plan: bool = True
    profile: str = "state_tracking"


class TraceStep(BaseModel):
    """One step of an agent's run: state, action, validity, error, raw model output, tokens."""

    model_config = ConfigDict(extra="forbid")

    problem_id: str
    model: str
    method: str
    step: int
    current_state: State
    action_id: str | None = None
    legal_action_count: int | None = None
    legal_action_ids: list[str] = Field(default_factory=list)
    blocked_action_ids: list[str] = Field(default_factory=list)
    retry_reason: str | None = None
    loop_recovery_attempted: bool = False
    model_action: Action | None = None
    predicted_next_state: State | None = None
    actual_next_state: State | None = None
    valid: bool = False
    error_type: FailureType | None = None
    message: str = ""
    raw_model_output: str = ""
    finish_reason: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    wall_time: float = 0.0


class TraceRun(BaseModel):
    """One complete agent rollout on one problem: ordered TraceSteps, success flag, totals."""

    model_config = ConfigDict(extra="forbid")

    problem_id: str
    environment: str
    complexity: int
    model: str
    method: str
    success: bool
    stopped_reason: str
    steps: list[TraceStep] = Field(default_factory=list)
    final_state: State | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    wall_time: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateSolution(BaseModel):
    """One move-list candidate extracted from raw model output + its replay outcome."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: int
    source: str = "unknown"
    raw_moves: list[Any] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    token_start: int = 0
    token_end: int = 0
    normalized_token_position: float | None = None
    valid_format: bool = True
    replay_success: bool = False
    first_failure_move: int | None = None
    first_failure_fraction: float | None = None
    failure_type: FailureType | None = None
    dedupe_key: str = ""


class TraceEvaluation(BaseModel):
    """Verified-trace evaluation: success, failure mode, derived rates, paper-headline metrics."""

    model_config = ConfigDict(extra="forbid")

    problem_id: str
    environment: str
    complexity: int
    method: str = ""
    model: str = ""
    success: bool
    failure_type: FailureType | None = None
    first_failure_step: int | None = None
    valid_move_rate: float = 0.0
    first_invalid_step: int | None = None
    state_drift_rate: float = 0.0
    constraint_violation_rate: float = 0.0
    recovery_success_rate: float = 0.0
    rollback_count: int = 0
    repeated_state_count: int = 0
    premature_stop: bool = False
    legal_action_adherence_rate: float = 0.0
    unknown_action_id_rate: float = 0.0
    invalid_after_resolution_rate: float = 0.0
    loop_entry_rate: float = 0.0
    loop_recovery_success_rate: float = 0.0
    state_drift_detected_rate: float = 0.0
    state_drift_corrected_rate: float = 0.0
    normalized_first_failure_step: float | None = None
    solution_length_ratio: float | None = None
    parse_schema_success_rate: float = 0.0
    final_success: bool = False
    thought_contains_success: bool = False
    correct_in_thought_wrong_final: bool = False
    first_solution_token_position: float | None = None
    first_failure_move_fraction: float | None = None
    num_unique_candidate_solutions: int = 0
    format_filter_pass: bool = False
    candidate_solutions: list[CandidateSolution] = Field(default_factory=list)
    pot_success: bool = False
    pot_plan_valid_prefix_fraction: float | None = None
    pot_first_failure_step: int | None = None
    pot_repaired_success: bool = False
    auto_start_policy: str | None = None
    auto_policy_route: list[str] = Field(default_factory=list)
    auto_switch_reasons: list[str] = Field(default_factory=list)
    auto_tool_attempted: bool = False
    auto_tool_valid_prefix_fraction: float | None = None
    auto_lookahead_steps: int = 0
    auto_model_rank_steps: int = 0
    auto_policy_switch_count: int = 0
    llm_calls: int = 0
    verified_steps_per_llm_call: float | None = None
    valid_prefix_reuse_rate: float = 0.0
    repair_call_rate: float = 0.0
    call_budget_exhaustion_rate: float = 0.0
    success_under_call_budget: bool = False
    provider_rejection: bool = False
    auto_v2_chunk_size_history: list[int] = Field(default_factory=list)
    tokens_total: int = 0
    tokens_per_valid_step: float | None = None
    explanation: str = ""
