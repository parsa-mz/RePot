from __future__ import annotations

from repot.core.agent import AgentConfig, FullSolutionAgent
from repot.core.schemas import Problem, TraceRun
from repot.core.env import PuzzleEnv
from repot.core.llm import ModelClient


class SelfConsistencyAgent(FullSolutionAgent):
    """FullSolutionAgent that runs `self_consistency_k` samples and keeps the first success or longest trace."""
    method = "self_consistency"
    prompt_style = "plain"

    def run(self, problem: Problem, env: PuzzleEnv, client: ModelClient, config: AgentConfig) -> TraceRun:
        """Sample up to k full-solution runs, returning the first success or the longest attempt."""
        best: TraceRun | None = None
        for _ in range(config.self_consistency_k):
            run = super().run(problem, env, client, config)
            if run.success:
                return run
            if best is None or len(run.steps) > len(best.steps):
                best = run
        assert best is not None
        return best
