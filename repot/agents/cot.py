from __future__ import annotations

from repot.core.agent import FullSolutionAgent


class CoTAgent(FullSolutionAgent):
    """FullSolutionAgent variant that prompts the model for a chain-of-thought single-shot solution."""
    method = "cot"
    prompt_style = "plain"
