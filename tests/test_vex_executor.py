"""Unit tests for the VEX core executor and the VEXAgent wrapper."""

from __future__ import annotations

import json

import pytest

from repot.core.agent import AgentConfig, make_agent
from repot.agents.repot.executor import ChunkProposal, VEXExecutor
from repot.core.verifier import StepOutcome, Verifier
from repot.core.tools.benchmarks import make_env
from repot.core.llm import GenerateRequest, GenerateResult


# ---------- Toy verifier (counter env) ------------------------------------


class _CounterVerifier(Verifier):
    """state = {"i": int}; only legal action is {"op": "inc"}; goal at i=3."""

    def __init__(self, target: int = 3) -> None:
        self.target = target

    def legal_actions(self, state):
        if state["i"] >= self.target:
            return []
        return [{"op": "inc"}]

    def step(self, state, action):
        if action.get("op") != "inc":
            return StepOutcome(
                valid=False,
                next_state=state,
                error_type="INVALID_TRANSITION",
                message=f"unknown op {action.get('op')!r}",
            )
        return StepOutcome(valid=True, next_state={"i": state["i"] + 1})

    def is_goal(self, state, goal):
        return state.get("i") == goal.get("i")


def test_executor_reaches_goal_in_one_chunk():
    chunks = [{"op": "inc"} for _ in range(3)]

    def proposer(state, verified_prefix, chunk_size, last_error, blocked=()):
        return ChunkProposal(actions=chunks, prompt_tokens=10, completion_tokens=4)

    executor = VEXExecutor(verifier=_CounterVerifier(3), proposer=proposer, chunk_size=4, max_llm_calls=4)
    result = executor.run({"i": 0}, {"i": 3})
    assert result.success
    assert result.stopped_reason == "goal"
    assert len(result.committed) == 3
    assert result.rollback_count == 0
    assert result.llm_calls == 1


def test_executor_rolls_back_invalid_suffix_and_continues():
    """First chunk: 2 valid + 1 invalid. Second chunk: 1 valid → goal."""
    seq = [
        [{"op": "inc"}, {"op": "inc"}, {"op": "BAD"}],
        [{"op": "inc"}],
    ]
    idx = {"n": 0}

    def proposer(state, verified_prefix, chunk_size, last_error, blocked=()):
        chunk = seq[idx["n"]]
        idx["n"] += 1
        return ChunkProposal(actions=chunk, prompt_tokens=10, completion_tokens=4)

    executor = VEXExecutor(verifier=_CounterVerifier(3), proposer=proposer, chunk_size=4, max_llm_calls=4)
    result = executor.run({"i": 0}, {"i": 3})
    assert result.success
    assert len(result.committed) == 3
    assert result.rollback_count == 1
    assert result.llm_calls == 2


def test_executor_exhausts_budget_when_proposer_keeps_failing():
    def proposer(state, verified_prefix, chunk_size, last_error, blocked=()):
        return ChunkProposal(actions=[{"op": "BAD"}], prompt_tokens=1, completion_tokens=1)

    executor = VEXExecutor(verifier=_CounterVerifier(3), proposer=proposer, chunk_size=4, max_llm_calls=3)
    result = executor.run({"i": 0}, {"i": 3})
    assert not result.success
    assert result.stopped_reason == "llm_call_budget_exhausted"
    assert result.llm_calls == 3
    # First BAD action gets recorded as a rollback and added to tabu; subsequent
    # calls hit the local skip path (no new rollback). So rollback_count is 1,
    # not 3 like before the tabu fix.
    assert result.rollback_count == 1


def test_executor_handles_empty_proposal_without_crashing():
    seq = [
        ChunkProposal(actions=[], error="empty"),
        ChunkProposal(actions=[{"op": "inc"} for _ in range(3)]),
    ]
    idx = {"n": 0}

    def proposer(state, verified_prefix, chunk_size, last_error, blocked=()):
        out = seq[idx["n"]]
        idx["n"] += 1
        return out

    executor = VEXExecutor(verifier=_CounterVerifier(3), proposer=proposer, chunk_size=4, max_llm_calls=4)
    result = executor.run({"i": 0}, {"i": 3})
    assert result.success
    assert result.llm_calls == 2


def test_executor_tabu_skips_known_blocked_action_inside_chunk():
    """Chunk contains [BAD, inc, inc, inc]. After 1st chunk: BAD blocked at i=0.
    2nd chunk: same [BAD, inc, inc, inc] — BAD is now locally skipped, the
    three incs commit through goal, no extra rollback recorded."""

    def proposer(state, verified_prefix, chunk_size, last_error, blocked=()):
        return ChunkProposal(actions=[{"op": "BAD"}, {"op": "inc"}, {"op": "inc"}, {"op": "inc"}], prompt_tokens=1, completion_tokens=1)

    executor = VEXExecutor(verifier=_CounterVerifier(3), proposer=proposer, chunk_size=4, max_llm_calls=4)
    result = executor.run({"i": 0}, {"i": 3})
    assert result.success
    # 1st chunk: BAD rollback (1 rollback). 2nd chunk: BAD skipped locally, 3 incs commit.
    assert result.rollback_count == 1
    assert result.llm_calls == 2


# ---------- Real-env integration via VEXAgent + dummy LLM client ----------


class _OracleClient:
    """Returns the next K oracle moves on every chunked call."""

    model = "vex_oracle_dummy"

    def generate(self, request: GenerateRequest) -> GenerateResult:
        meta = request.metadata
        if meta.get("program_of_thought"):
            return GenerateResult(text="print('moves = []')", prompt_tokens=5, completion_tokens=5)
        if meta.get("chunked_actions"):
            oracle = meta.get("oracle_actions", [])
            idx = int(meta.get("oracle_index", 0))
            chunk_size = int(meta.get("chunk_size", 4))
            chunk = oracle[idx : idx + chunk_size]
            return GenerateResult(
                text=json.dumps({"actions": chunk}), prompt_tokens=12, completion_tokens=8
            )
        return GenerateResult(text='{"action_id":"a0"}', prompt_tokens=3, completion_tokens=2)


@pytest.mark.parametrize(
    "env_name,complexity",
    [("blocksworld", 3), ("tower_of_hanoi", 3), ("river_crossing", 2), ("checker_jumping", 2)],
)
def test_vex_agent_solves_oracle_problems_end_to_end(env_name, complexity):
    env = make_env(env_name)
    problem = env.generate(seed=0, complexity=complexity, template_id=0)
    cfg = AgentConfig(chunk_size=4, max_llm_calls=10)
    run = make_agent("vex").run(problem, env, _OracleClient(), cfg)
    assert run.success, f"VEX failed on {env_name} c={complexity}: {run.stopped_reason}"
    assert run.metadata["vex_chunk_size"] in {2, 3, 4, 6}
    assert run.metadata["vex_llm_calls"] >= 1


def test_vex_agent_records_rollback_on_invalid_suffix():
    """Returns 1 valid move + 1 invalid every chunk; should partially commit."""

    class _PartialClient:
        model = "vex_partial_dummy"

        def generate(self, request):
            meta = request.metadata
            if meta.get("program_of_thought"):
                return GenerateResult(text="print('moves = []')", prompt_tokens=5, completion_tokens=5)
            if meta.get("chunked_actions"):
                oracle = meta.get("oracle_actions", [])
                idx = int(meta.get("oracle_index", 0))
                head = oracle[idx : idx + 1]
                actions = list(head) + [{"type": "invalid"}]
                return GenerateResult(text=json.dumps({"actions": actions}), prompt_tokens=10, completion_tokens=4)
            return GenerateResult(text='{"action_id":"a0"}', prompt_tokens=3, completion_tokens=2)

    env = make_env("blocksworld")
    problem = env.generate(seed=0, complexity=3, template_id=0)
    cfg = AgentConfig(chunk_size=4, max_llm_calls=10)
    run = make_agent("vex").run(problem, env, _PartialClient(), cfg)
    # Should still solve eventually because each chunk commits one good move.
    assert run.success
    assert run.metadata["vex_rollback_count"] >= 1


# ---------- Code proposer routing -----------------------------------------


class _CodeProposerOracleClient:
    """Emits Python that prints `moves = [<oracle suffix>]` keyed off oracle_index.

    Tier-6 novelty: code conditioned on verified prefix. Verifier-grounded
    metadata (oracle_index = how many moves already committed) drives which
    suffix the program prints.
    """

    model = "vex_code_oracle_dummy"

    def generate(self, request: GenerateRequest) -> GenerateResult:
        meta = request.metadata
        if not meta.get("vex_code_proposer"):
            # Fallback for safety; shouldn't fire when proposer=code is selected.
            return GenerateResult(text='{"actions":[]}', prompt_tokens=2, completion_tokens=2)
        oracle = meta.get("oracle_actions", [])
        idx = int(meta.get("oracle_index", 0))
        suffix = oracle[idx:]
        code = "moves = " + repr(list(suffix)) + "\nprint('moves =', moves)"
        return GenerateResult(text=code, prompt_tokens=12, completion_tokens=10)


@pytest.mark.parametrize(
    "env_name,complexity",
    [("blocksworld", 4), ("tower_of_hanoi", 4), ("river_crossing", 2), ("checker_jumping", 2)],
)
def test_vex_code_proposer_solves_via_python(env_name, complexity):
    env = make_env(env_name)
    problem = env.generate(seed=0, complexity=complexity, template_id=0)
    cfg = AgentConfig(
        chunk_size=8,
        max_llm_calls=4,
        vex_default_proposer="code",
    )
    run = make_agent("vex").run(problem, env, _CodeProposerOracleClient(), cfg)
    assert run.success, f"code proposer failed on {env_name}: {run.stopped_reason}"
    assert run.metadata["vex_proposer"] == "code"


def test_vex_proposer_by_env_routing_picks_correctly():
    """Per-env override must beat the default."""
    env = make_env("blocksworld")
    problem = env.generate(seed=0, complexity=3, template_id=0)
    cfg = AgentConfig(
        chunk_size=8,
        max_llm_calls=4,
        vex_default_proposer="json",
        vex_proposer_by_env={"blocksworld": "code"},
    )
    run = make_agent("vex").run(problem, env, _CodeProposerOracleClient(), cfg)
    assert run.metadata["vex_proposer"] == "code"
    assert run.success
