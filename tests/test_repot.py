"""Tests for ``repot.agents.repot.agent.RePoTAgent`` (Algorithm 1).

Cover:
1. PoT-success path — initial PoT plan is correct; replay reaches goal; zero repair calls.
2. Format-error path — model never emits parseable moves; repair budget exhausts.
3. Partial-plan + recovery path — first plan is partly valid; suffix repair completes.
4. Trace schema — metadata keys present, token accounting on first step of each chunk.
"""

from __future__ import annotations

import json

from repot.core.agent import AgentConfig, make_agent
from repot.agents.repot.agent import RePoTAgent
from repot.core.schemas import Problem
from repot.core.tools.benchmarks import make_env
from repot.core.llm import GenerateRequest, GenerateResult
from repot.core.clients.local_dummy import LocalDummyClient


def _hanoi3() -> tuple:
    env = make_env("tower_of_hanoi")
    problem = env.generate(seed=0, complexity=3, template_id=0)
    return env, problem


def test_pot_success_path_zero_repair_calls() -> None:
    env, problem = _hanoi3()
    run = make_agent("repot").run(problem, env, LocalDummyClient(), AgentConfig(max_repair_calls=1))
    assert run.success
    assert run.stopped_reason == "goal"
    assert run.metadata["repot_initial_pot_success"] is True
    assert run.metadata["repot_repair_calls"] == 0
    assert run.metadata["repot_first_failure_step"] is None
    assert run.metadata["repot_committed_actions"] == len(problem.oracle_solution)
    assert all(step.valid for step in run.steps)


def test_format_error_exhausts_repair_budget() -> None:
    env, problem = _hanoi3()
    run = RePoTAgent().run(
        problem, env, LocalDummyClient(mode="format_error"),
        AgentConfig(max_repair_calls=2),
    )
    assert not run.success
    assert run.stopped_reason in {"repair_budget_exhausted", "repair_format_error"}
    assert run.metadata["repot_repair_calls"] == 2
    assert run.metadata["repot_committed_actions"] == 0
    assert run.metadata["repot_initial_pot_success"] is False


class _PartialThenSuffixClient:
    """First call emits a *partial* PoT plan (oracle_actions[:3] only).
    Second call emits the *suffix* (oracle_actions[3:]) — simulating a
    successful suffix-repair recovery."""

    model: str = "partial_then_suffix"

    def __init__(self, oracle: list) -> None:
        self.oracle = oracle
        self.calls = 0

    def generate(self, request: GenerateRequest) -> GenerateResult:
        self.calls += 1
        if self.calls == 1:
            text = "moves = " + repr(list(self.oracle[:3])) + "\nprint('moves = ' + repr(moves))"
        else:
            text = "moves = " + repr(list(self.oracle[3:])) + "\nprint('moves = ' + repr(moves))"
        return GenerateResult(
            text=text,
            prompt_tokens=10 * self.calls,
            completion_tokens=20 * self.calls,
            latency_s=0.001,
            finish_reason="stop",
        )


def test_partial_plan_then_suffix_repair_succeeds() -> None:
    env, problem = _hanoi3()
    client = _PartialThenSuffixClient(problem.oracle_solution)
    run = RePoTAgent().run(problem, env, client, AgentConfig(max_repair_calls=2))
    assert run.success
    assert run.stopped_reason == "goal"
    # PoT alone did NOT solve — only the partial prefix replayed.
    assert run.metadata["repot_initial_pot_success"] is False
    # Suffix repair fired exactly once.
    assert run.metadata["repot_repair_calls"] == 1
    assert run.metadata["repot_committed_actions"] == len(problem.oracle_solution)
    # The verified-prefix-fraction on the initial plan reflects "all 3 emitted moves were valid"
    # (1.0 because the plan was *short*, not because it reached the goal).
    assert run.metadata["repot_initial_valid_prefix_fraction"] == 1.0
    assert run.metadata["repot_first_failure_step"] is None
    assert client.calls == 2


def test_disable_prefix_in_prompt_flag_is_recorded() -> None:
    env, problem = _hanoi3()
    cfg = AgentConfig(max_repair_calls=1, vex_disable_prefix_in_prompt=True)
    run = RePoTAgent().run(problem, env, LocalDummyClient(), cfg)
    assert run.metadata["repot_disable_prefix_in_prompt"] is True


def test_trace_step_schema_token_accounting() -> None:
    """First TraceStep of each chunk gets the proposal's tokens; subsequent
    steps in the same chunk carry zero. Token totals on the run reconcile."""
    env, problem = _hanoi3()
    run = make_agent("repot").run(problem, env, LocalDummyClient(), AgentConfig())
    # Steps with non-zero tokens_in mark chunk boundaries.
    chunk_boundaries = [i for i, s in enumerate(run.steps) if s.tokens_in > 0]
    assert chunk_boundaries == [0]  # only the initial PoT chunk in the success path
    assert run.tokens_in == sum(s.tokens_in for s in run.steps)
    assert run.tokens_out == sum(s.tokens_out for s in run.steps)


def test_repot_dispatches_via_make_agent() -> None:
    agent = make_agent("repot")
    assert agent.method == "repot"
    assert isinstance(agent, RePoTAgent)


def test_repot_handles_other_envs() -> None:
    """Smoke: RePoT should dispatch and complete on each env at low complexity."""
    for env_name, complexity in [
        ("checker_jumping", 2),
        ("river_crossing", 1),
        ("blocksworld", 3),
    ]:
        env = make_env(env_name)
        problem = env.generate(seed=0, complexity=complexity, template_id=0)
        run = make_agent("repot").run(problem, env, LocalDummyClient(), AgentConfig(max_repair_calls=1))
        assert run.method == "repot"
        # The oracle dummy should solve every environment on first PoT call.
        assert run.success, f"{env_name}/{complexity} did not succeed: stopped={run.stopped_reason}"
        # Final state must be the goal.
        assert env.is_goal(run.final_state, problem.goal_state)


def test_problem_round_trip() -> None:
    """Sanity: Problem schema survives JSON round-trip (used for trace storage)."""
    _, problem = _hanoi3()
    blob = problem.model_dump_json()
    Problem.model_validate(json.loads(blob))
