"""Verifier ABC for the public VEX API.

Any object exposing ``legal_actions / step / is_goal / progress_score / normalize``
can be a Verifier. The existing project envs (Hanoi, Blocksworld, River, Checker)
already satisfy this protocol; the ABC here documents the contract for external
adopters of the pip package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

State = dict[str, Any]
Action = dict[str, Any]


class StepOutcome:
    """Result of executing a single action against a verifier."""

    __slots__ = ("valid", "next_state", "error_type", "message")

    def __init__(
        self,
        valid: bool,
        next_state: State,
        error_type: str | None = None,
        message: str = "",
    ) -> None:
        self.valid = valid
        self.next_state = next_state
        self.error_type = error_type
        self.message = message


class Verifier(ABC):
    """Single source of truth for external state.

    Implementations OWN the state. The LLM never mutates state directly; it
    proposes actions, and the Verifier accepts or rejects each one.
    """

    @abstractmethod
    def legal_actions(self, state: State) -> list[Action]:
        """Enumerate every action that would be accepted from ``state``."""

    @abstractmethod
    def step(self, state: State, action: Action) -> StepOutcome:
        """Apply ``action``; return new state + validity."""

    @abstractmethod
    def is_goal(self, state: State, goal: State) -> bool:
        """Has the run reached its terminal goal?"""

    def progress_score(self, state: State, goal: State) -> float:
        """Optional [0,1] heuristic; defaults to 1.0 at goal, 0.0 elsewhere."""
        return 1.0 if self.is_goal(state, goal) else 0.0

    def normalize(self, raw: Any, state: State | None = None) -> Action:
        """Convert a model-emitted move (dict / list / string) into a canonical action.

        Default implementation accepts only dicts; per-domain Verifiers should
        override to handle list-form (e.g. ["A", from, to]) and other shapes.
        """
        del state
        if isinstance(raw, dict):
            return raw
        raise ValueError(f"Cannot normalize candidate move: {raw!r}")
