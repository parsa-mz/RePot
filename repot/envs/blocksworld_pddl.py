"""PlanBench Blocksworld env (PDDL semantics).

Adapter for PlanBench's Blocksworld benchmark (Valmeekam et al., NeurIPS 2023).
Source instances: github.com/karthikv792/LLMs-Planning/plan-bench/instances/blocksworld

Why a separate env from `blocksworld.py`:
- PlanBench uses predicate-set state, not stack-list state.
- PlanBench's 4-op action vocabulary is gripper-mediated: pick-up, put-down,
  stack, unstack. Our existing env uses stack-to-stack `move` actions.
- PlanBench goals are PARTIAL (subset of `(on x y)` predicates must hold);
  the existing env requires full-state equality.

We keep the same `Env` ABC. Plans are represented as lists of actions of
shape ``{"type": "pick-up"|"put-down"|"stack"|"unstack", ...}``.
"""

from __future__ import annotations

from collections import deque

from repot.core.schemas import Action, EnvCapabilities, FailureType, Problem, State, StepResult
from repot.core.env import PuzzleEnv


class BlocksWorldPDDLEnv(PuzzleEnv):
    """PuzzleEnv adapter for PlanBench's predicate-state Blocksworld with the 4-op (pick-up/put-down/stack/unstack) action vocabulary."""
    name = "blocksworld_pddl"

    def capabilities(self, problem: Problem) -> EnvCapabilities:
        """Report the state-tracking capability profile for predicate Blocksworld."""
        del problem
        return EnvCapabilities(profile="state_tracking")

    # --- generation is not used (instances are loaded from PDDL files) ---
    def generate(self, seed: int, complexity: int, template_id: int = 0) -> Problem:
        """Not implemented: PDDL Blocksworld instances are loaded from PlanBench files, not procedurally generated."""
        raise NotImplementedError(
            "BlocksWorldPDDLEnv loads instances from PlanBench PDDL files. "
        )

    # --- core semantics ---
    def legal_actions(self, state: State) -> list[Action]:
        """Enumerate the 4-op gripper-mediated actions: pick-up/unstack when handempty, put-down/stack when holding a block."""
        clear = set(state.get("clear", []))
        ontable = set(state.get("ontable", []))
        on = dict(state.get("on", {}))  # x -> y means x is on y
        holding = state.get("holding")
        actions: list[Action] = []

        if holding is None:
            # pick-up: clear AND ontable
            for blk in sorted(clear & ontable):
                actions.append({"type": "pick-up", "block": blk})
            # unstack: on(x, y) AND clear(x)
            for blk in sorted(clear):
                if blk in on:
                    actions.append({"type": "unstack", "block": blk, "under": on[blk]})
        else:
            # put-down
            actions.append({"type": "put-down", "block": holding})
            # stack onto a clear block
            for under in sorted(clear):
                if under == holding:
                    continue
                actions.append({"type": "stack", "block": holding, "under": under})
        return actions

    def step(self, state: State, action: Action) -> StepResult:
        """Apply a 4-op action; return INVALID_TRANSITION on precondition failure."""
        try:
            new_state = self._apply(state, action)
        except _PDDLError as exc:
            return StepResult(
                valid=False,
                next_state=state,
                error_type=FailureType.INVALID_TRANSITION,
                message=str(exc),
            )
        return StepResult(valid=True, next_state=new_state)

    def _apply(self, state: State, action: Action) -> State:
        a = action or {}
        atype = a.get("type")
        blk = a.get("block")
        under = a.get("under")
        clear = list(state.get("clear", []))
        ontable = list(state.get("ontable", []))
        on = dict(state.get("on", {}))
        holding = state.get("holding")
        blocks = list(state.get("blocks", []))

        if atype == "pick-up":
            if holding is not None:
                raise _PDDLError(f"pick-up: hand not empty (holding {holding}).")
            if blk not in clear:
                raise _PDDLError(f"pick-up: {blk} is not clear.")
            if blk not in ontable:
                raise _PDDLError(f"pick-up: {blk} is not ontable.")
            ontable.remove(blk)
            clear.remove(blk)
            holding = blk
        elif atype == "put-down":
            if holding != blk:
                raise _PDDLError(f"put-down: not holding {blk} (holding {holding}).")
            ontable.append(blk)
            clear.append(blk)
            holding = None
        elif atype == "stack":
            if holding != blk:
                raise _PDDLError(f"stack: not holding {blk} (holding {holding}).")
            if under not in clear:
                raise _PDDLError(f"stack: target {under} is not clear.")
            if under == blk:
                raise _PDDLError("stack: cannot stack a block on itself.")
            clear.remove(under)
            on[blk] = under
            clear.append(blk)
            holding = None
        elif atype == "unstack":
            if holding is not None:
                raise _PDDLError(f"unstack: hand not empty (holding {holding}).")
            if blk not in clear:
                raise _PDDLError(f"unstack: {blk} is not clear.")
            if on.get(blk) != under:
                raise _PDDLError(f"unstack: {blk} is not on {under}.")
            del on[blk]
            clear.remove(blk)
            clear.append(under)
            holding = blk
        else:
            raise _PDDLError(f"Unknown action type: {atype!r}. Use pick-up/put-down/stack/unstack.")

        return {
            "blocks": sorted(blocks),
            "ontable": sorted(set(ontable)),
            "on": on,
            "clear": sorted(set(clear)),
            "holding": holding,
        }

    def is_goal(self, state: State, goal: State) -> bool:
        """Partial-goal check: every `on(x,y)` and `ontable(x)` predicate in `goal` must hold in `state`."""
        # PlanBench partial-goal semantics: every goal `on(x,y)` predicate
        # must hold in state. handempty / clear / holding are not asserted in
        # the goal block, so we don't check them.
        goal_on = dict(goal.get("on", {}))
        state_on = dict(state.get("on", {}))
        for x, y in goal_on.items():
            if state_on.get(x) != y:
                return False
        # Goals can also include `ontable(x)` predicates, though rare in PlanBench
        for blk in goal.get("ontable", []):
            if blk not in state.get("ontable", []):
                return False
        return True

    # --- oracle: BFS over predicate states. Capped to keep generation cheap. ---
    def oracle_solution(self, problem: Problem) -> list[Action]:
        """Breadth-first search over predicate states up to `max_steps`; returns an empty list if unsolved within the budget."""
        max_depth = problem.max_steps
        seen = {self._state_key(problem.initial_state)}
        queue: deque[tuple[State, list[Action]]] = deque([(problem.initial_state, [])])
        max_seen = 200_000
        while queue and len(seen) < max_seen:
            state, path = queue.popleft()
            if self.is_goal(state, problem.goal_state):
                return path
            if len(path) >= max_depth:
                continue
            for action in self.legal_actions(state):
                result = self.step(state, action)
                if not result.valid:
                    continue
                key = self._state_key(result.next_state)
                if key in seen:
                    continue
                seen.add(key)
                queue.append((result.next_state, path + [action]))
        return []  # unsolved within budget; OK to return empty (we still measure attempt)

    def _state_key(self, state: State) -> tuple:
        return (
            tuple(sorted(state.get("ontable", []))),
            tuple(sorted((b, u) for b, u in state.get("on", {}).items())),
            tuple(sorted(state.get("clear", []))),
            state.get("holding"),
        )

    # --- prompting (PlanBench-style; tells model to emit code with the 4-op vocabulary) ---
    def render_prompt(self, problem: Problem, template_id: int) -> str:
        """Build a PlanBench-style prompt listing initial/goal predicates and the 4-op action vocabulary."""
        del template_id
        init = problem.initial_state
        goal = problem.goal_state
        return (
            f"Solve a Blocksworld planning problem with {problem.complexity} blocks.\n"
            f"Blocks: {init.get('blocks')}\n"
            f"Initial state predicates:\n"
            f"{_render_predicates(init)}\n"
            f"Goal predicates (must all hold; partial — other relations may end up arbitrary):\n"
            f"{_render_predicates(goal, goal_only=True)}\n\n"
            f"Action vocabulary (PDDL Blocksworld 4-ops):\n"
            f"  pick-up(b)        — preconds: clear(b), ontable(b), handempty\n"
            f"  put-down(b)       — precond:  holding(b)\n"
            f"  stack(b, under)   — preconds: holding(b), clear(under)\n"
            f"  unstack(b, under) — preconds: on(b, under), clear(b), handempty\n\n"
            f"Return the answer exactly as:\n"
            f"  moves = [[\"pick-up\", \"a\"], [\"stack\", \"a\", \"b\"], [\"unstack\", \"c\", \"d\"], ...]\n"
            f"Each move is a list whose first element is the action name; "
            f"pick-up/put-down take 1 block; stack/unstack take 2."
        )

    def action_label(self, action: Action) -> str:
        """Return a human-readable label like `stack a on/from b` or `pick-up a`."""
        a = action or {}
        atype = a.get("type", "?")
        if atype in {"stack", "unstack"}:
            return f"{atype} {a.get('block')} on/from {a.get('under')}"
        return f"{atype} {a.get('block')}"

    def normalize_candidate_move(self, problem: Problem, raw_move, state: State | None = None) -> Action:
        """Coerce a list `[op, block, ...]` or partial dict into a normalized 4-op action dict."""
        del problem, state
        # Already-normalized dict
        if isinstance(raw_move, dict):
            d = dict(raw_move)
            if "type" not in d:
                # Some models emit {"action": "...", "block": "..."} — accept it
                if "action" in d:
                    d["type"] = d.pop("action")
            return d
        # List form: ["pick-up", "a"] or ["stack", "a", "b"]
        if isinstance(raw_move, (list, tuple)):
            parts = list(raw_move)
            if not parts:
                raise ValueError("Empty move.")
            atype = str(parts[0]).strip().lower().replace("_", "-")
            if atype in {"pick-up", "pickup", "pick"}:
                atype = "pick-up"
            elif atype in {"put-down", "putdown", "put"}:
                atype = "put-down"
            elif atype not in {"stack", "unstack"}:
                raise ValueError(f"Unknown action: {parts[0]!r}")
            if atype in {"pick-up", "put-down"}:
                if len(parts) < 2:
                    raise ValueError(f"{atype} requires 1 block argument.")
                return {"type": atype, "block": str(parts[1])}
            if len(parts) < 3:
                raise ValueError(f"{atype} requires 2 block arguments.")
            return {"type": atype, "block": str(parts[1]), "under": str(parts[2])}
        raise ValueError(f"Cannot normalize move: {raw_move!r}")

    def progress_score(self, state: State, goal: State) -> float:
        """Fraction of goal `on(x,y)` predicates already satisfied in `state`; 1.0 when the partial goal holds."""
        if self.is_goal(state, goal):
            return 1.0
        goal_on = dict(goal.get("on", {}))
        if not goal_on:
            return 0.0
        state_on = dict(state.get("on", {}))
        hits = sum(1 for x, y in goal_on.items() if state_on.get(x) == y)
        return hits / len(goal_on)


class _PDDLError(ValueError):
    """Verifier rejection inside `_apply`."""


def _render_predicates(s: State, goal_only: bool = False) -> str:
    lines: list[str] = []
    if not goal_only:
        if s.get("holding"):
            lines.append(f"  holding({s['holding']})")
        else:
            lines.append("  handempty")
        for b in s.get("ontable", []):
            lines.append(f"  ontable({b})")
        for b in s.get("clear", []):
            lines.append(f"  clear({b})")
    for b, u in sorted(s.get("on", {}).items()):
        lines.append(f"  on({b}, {u})")
    if goal_only:
        for b in s.get("ontable", []):
            lines.append(f"  ontable({b})")
    return "\n".join(lines) if lines else "  (none)"
