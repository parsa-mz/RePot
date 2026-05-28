from __future__ import annotations

import pytest

from repot.core.schemas import FailureType, TraceStep
from repot.core.tools.benchmarks import make_env


@pytest.mark.parametrize(
    ("env_name", "complexity"),
    [
        ("tower_of_hanoi", 3),
        ("checker_jumping", 3),
        ("river_crossing", 1),
        ("blocksworld", 4),
    ],
)
def test_oracle_reaches_goal(env_name: str, complexity: int) -> None:
    env = make_env(env_name)
    problem = env.generate(seed=1, complexity=complexity, template_id=0)
    state = problem.initial_state
    for action in problem.oracle_solution:
        result = env.step(state, action)
        assert result.valid, result.message
        state = result.next_state
    assert env.is_goal(state, problem.goal_state)


@pytest.mark.parametrize(
    ("env_name", "complexity"),
    [
        ("tower_of_hanoi", 3),
        ("checker_jumping", 2),
        ("river_crossing", 1),
        ("blocksworld", 3),
    ],
)
def test_verify_trace_accepts_oracle(env_name: str, complexity: int) -> None:
    env = make_env(env_name)
    problem = env.generate(seed=0, complexity=complexity, template_id=0)
    state = problem.initial_state
    steps = []
    for idx, action in enumerate(problem.oracle_solution):
        result = env.step(state, action)
        steps.append(
            TraceStep(
                problem_id=problem.problem_id,
                model="oracle",
                method="oracle",
                step=idx,
                current_state=state,
                model_action=action,
                actual_next_state=result.next_state,
                valid=result.valid,
            )
        )
        state = result.next_state
    evaluation = env.verify_trace(problem, steps)
    assert evaluation.success
    assert evaluation.failure_type is None


def test_hanoi_rejects_larger_disk_on_smaller() -> None:
    env = make_env("tower_of_hanoi")
    problem = env.generate(seed=0, complexity=3, template_id=0)
    state = problem.initial_state
    state = env.step(state, {"type": "move", "disk": 1, "from": "A", "to": "B"}).next_state
    result = env.step(state, {"type": "move", "disk": 2, "from": "A", "to": "B"})
    assert not result.valid
    assert result.error_type == FailureType.CONSTRAINT_VIOLATION


def test_checker_rejects_backward_move() -> None:
    env = make_env("checker_jumping")
    problem = env.generate(seed=0, complexity=2, template_id=0)
    result = env.step(problem.initial_state, {"type": "slide", "from": 3, "to": 2, "piece": "Y"})
    assert result.valid
    result = env.step(result.next_state, {"type": "slide", "from": 2, "to": 3, "piece": "Y"})
    assert not result.valid


def test_river_rejects_unsafe_bank() -> None:
    env = make_env("river_crossing")
    problem = env.generate(seed=0, complexity=1, template_id=0)
    result = env.step(problem.initial_state, {"type": "cross", "cargo": ["cabbage1"], "from": "left", "to": "right"})
    assert not result.valid
    assert result.error_type == FailureType.CONSTRAINT_VIOLATION


def test_blocksworld_rejects_non_clear_block() -> None:
    env = make_env("blocksworld")
    state = {"stacks": [["A", "B"], [], []], "blocks": ["A", "B"]}
    result = env.step(state, {"type": "move", "block": "A", "from": 0, "to": 1})
    assert not result.valid


def test_action_options_are_deterministic_and_resolve() -> None:
    env = make_env("checker_jumping")
    problem = env.generate(seed=0, complexity=2, template_id=2)
    first = env.action_options(problem.initial_state)
    second = env.action_options(problem.initial_state)
    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]
    assert [item.id for item in first] == [f"a{i}" for i in range(len(first))]
    result = env.step(problem.initial_state, first[0].action)
    assert result.valid
