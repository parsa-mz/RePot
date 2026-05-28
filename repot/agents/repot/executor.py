"""VEX executor — the one-loop algorithm.

```
state = initial_state
while not goal(state) and calls < budget:
    chunk = propose(state, K, verified_prefix)        # one LLM call
    for action in chunk:
        action = verifier.normalize(action, state)
        outcome = verifier.step(state, action)
        if not outcome.valid:
            break                                     # rollback suffix
        state = outcome.next_state                    # commit
```

Two knobs only: ``K`` (per env) and ``max_llm_calls``. Everything else
(prefix caching, schema decoding, verifier-grounded prompting) is the
caller's responsibility — supplied through the ``Proposer`` callable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from repot.core.verifier import Action, State, Verifier


# Per-state tabu cap. Small on purpose — we want the prompt block to stay
# tiny and we want to bail out of pathological repeat-rollback loops fast.
DEFAULT_MAX_BLOCKED_PER_STATE = 5


@dataclass
class ChunkProposal:
    """One LLM call's worth of suggested actions plus accounting metadata."""

    actions: list[Any]
    raw_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str | None = None
    error: str = ""


class Proposer(Protocol):
    """Anything that maps (state, K, history, blocked) → ``ChunkProposal``.

    ``blocked`` is the per-state tabu list — a list of ``(action, message)`` the
    verifier has already rejected from the *current* state. Proposers should
    surface it to the model so it does not re-propose the same dead-ends.
    """

    def __call__(
        self,
        state: State,
        verified_prefix: list[Action],
        chunk_size: int,
        last_error: str,
        blocked: list[tuple[Action, str]] = ...,
    ) -> ChunkProposal:
        ...


@dataclass
class CommittedAction:
    """Action that the verifier accepted, with the states before and after it."""
    action: Action
    pre_state: State
    post_state: State


@dataclass
class RolledBackAction:
    """Action the verifier rejected, with the failure category and message."""
    action: Action
    pre_state: State
    error_type: str | None
    message: str


@dataclass
class VEXResult:
    """Outcome of a VEX executor run: success flag, final state, and per-call accounting."""
    success: bool
    final_state: State
    stopped_reason: str
    committed: list[CommittedAction] = field(default_factory=list)
    rollbacks: list[RolledBackAction] = field(default_factory=list)
    proposals: list[ChunkProposal] = field(default_factory=list)
    llm_calls: int = 0
    chunk_attempts: int = 0
    rollback_count: int = 0


@dataclass
class VEXExecutor:
    """The whole mechanism, in one class."""

    verifier: Verifier
    proposer: Proposer
    chunk_size: int = 4
    max_llm_calls: int = 8
    max_blocked_per_state: int = DEFAULT_MAX_BLOCKED_PER_STATE

    def run(self, initial_state: State, goal_state: State) -> VEXResult:
        """Loop propose-then-verify chunks until goal is reached or the LLM-call budget runs out."""
        state = _deep_copy(initial_state)
        result = VEXResult(success=False, final_state=state, stopped_reason="")
        last_error = ""
        # Tabu memory: keyed by state-key, value is a list of (action, msg)
        # that the verifier has already rejected at this exact state. Bounded
        # per-state by ``max_blocked_per_state`` so the prompt block stays small.
        blocked: dict[str, list[tuple[Any, str]]] = {}

        while not self.verifier.is_goal(state, goal_state):
            if result.llm_calls >= self.max_llm_calls:
                result.stopped_reason = "llm_call_budget_exhausted"
                result.final_state = state
                return result

            verified_prefix = [c.action for c in result.committed]
            current_key = _state_key(state)
            blocked_here = blocked.get(current_key, [])
            proposal = self.proposer(
                state=state,
                verified_prefix=verified_prefix,
                chunk_size=self.chunk_size,
                last_error=last_error,
                blocked=blocked_here,
            )
            result.llm_calls += 1
            result.chunk_attempts += 1
            result.proposals.append(proposal)

            if not proposal.actions:
                last_error = proposal.error or "empty proposal"
                continue

            committed_this_round = 0
            for raw in proposal.actions:
                try:
                    action = self.verifier.normalize(raw, state)
                except Exception as exc:
                    last_error = f"normalize_failed: {exc}"
                    result.rollbacks.append(
                        RolledBackAction(
                            action=raw if isinstance(raw, dict) else {"raw": raw},
                            pre_state=_deep_copy(state),
                            error_type="OUTPUT_FORMAT_ERROR",
                            message=str(exc),
                        )
                    )
                    result.rollback_count += 1
                    break
                # Local tabu skip: if the verifier already rejected this exact
                # action from this exact state, don't waste a verifier call (and
                # don't re-trigger a rollback that would break the chunk early).
                # The model has been told about it in the prompt; treat it as a
                # silent skip and try the next slot.
                if _action_is_blocked(action, blocked_here):
                    continue
                outcome = self.verifier.step(state, action)
                if not outcome.valid:
                    last_error = outcome.message or "invalid_transition"
                    bucket = blocked.setdefault(current_key, [])
                    if len(bucket) < self.max_blocked_per_state and not _action_is_blocked(action, bucket):
                        bucket.append((action, outcome.message or outcome.error_type or "blocked"))
                    result.rollbacks.append(
                        RolledBackAction(
                            action=action,
                            pre_state=_deep_copy(state),
                            error_type=outcome.error_type,
                            message=outcome.message,
                        )
                    )
                    result.rollback_count += 1
                    break
                pre = _deep_copy(state)
                state = outcome.next_state
                result.committed.append(
                    CommittedAction(action=action, pre_state=pre, post_state=_deep_copy(state))
                )
                committed_this_round += 1
                # State changed — refresh tabu pointer for any further slots in
                # this chunk (different state ⇒ different blocked list).
                current_key = _state_key(state)
                blocked_here = blocked.get(current_key, [])
                if self.verifier.is_goal(state, goal_state):
                    break

            if committed_this_round > 0:
                # Made forward progress — clear the residual "last error" so
                # the next prompt doesn't drag stale context.
                last_error = ""

        result.success = self.verifier.is_goal(state, goal_state)
        result.final_state = state
        result.stopped_reason = "goal" if result.success else "max_steps"
        return result


def _deep_copy(state: State) -> State:
    """Cheap deep copy good enough for the state shapes in the project envs."""
    return json.loads(json.dumps(state))


def _state_key(state: State) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"))


def _action_sig(action: Action) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"), default=str)


def _action_is_blocked(action: Action, bucket: list[tuple[Any, str]]) -> bool:
    sig = _action_sig(action)
    return any(_action_sig(prev) == sig for prev, _msg in bucket)


def format_blocked(blocked: list[tuple[Any, str]], cap: int = 5) -> str:
    """Render the per-state tabu list for a prompt block. One short line each.

    Public so the JSON and code proposers in ``repot.agents.vex`` /
    ``repot.agents._chunked_plan`` render the block identically.
    """
    if not blocked:
        return "(none)"
    items = []
    for action, msg in blocked[:cap]:
        items.append(f"{_action_sig(action)} → {(msg or 'blocked').strip()[:120]}")
    more = "" if len(blocked) <= cap else f" (+{len(blocked) - cap} more)"
    return "; ".join(items) + more
