from __future__ import annotations

import copy

from repot.core.schemas import Action, EnvCapabilities, FailureType, Problem, State, StepResult
from repot.core.env import PuzzleEnv


class TowerOfHanoiEnv(PuzzleEnv):
    """PuzzleEnv for Tower of Hanoi with three pegs A/B/C; long-horizon move sequences governed by the smaller-on-larger constraint."""
    name = "tower_of_hanoi"

    def capabilities(self, problem: Problem) -> EnvCapabilities:
        """Report the long-horizon capability profile for Tower of Hanoi."""
        del problem
        return EnvCapabilities(profile="long_horizon")

    def generate(self, seed: int, complexity: int, template_id: int = 0) -> Problem:
        """Build a Hanoi instance with `complexity` disks and seed-rotated source/target pegs; attaches the recursive oracle solution."""
        pegs = ["A", "B", "C"]
        source = pegs[seed % 3]
        target = pegs[(seed + 1) % 3]
        spare = next(p for p in pegs if p not in {source, target})
        initial = {"pegs": {p: [] for p in pegs}, "peg_names": pegs}
        initial["pegs"][source] = list(range(complexity, 0, -1))
        goal = {"pegs": {p: [] for p in pegs}, "peg_names": pegs}
        goal["pegs"][target] = list(range(complexity, 0, -1))
        metadata = {"seed": seed, "template_id": template_id, "source": source, "target": target, "spare": spare}
        problem = Problem(
            problem_id=f"hanoi_n{complexity:02d}_seed{seed:04d}_template{template_id:02d}",
            environment=self.name,
            complexity=complexity,
            initial_state=initial,
            goal_state=goal,
            natural_language_prompt="",
            min_steps=(2**complexity) - 1,
            max_steps=max(1, 2 * ((2**complexity) - 1)),
            metadata=metadata,
        )
        actions = self.oracle_solution(problem)
        return problem.model_copy(update={"natural_language_prompt": self.render_prompt(problem, template_id), "oracle_solution": actions})

    def generate_puzzlezoo(self, seed: int, complexity: int, template_id: int = 0) -> Problem:
        """Build the PuzzleZoo Hanoi variant fixed to A->C with `[disk, from_peg, to_peg]` output format."""
        pegs = ["A", "B", "C"]
        source, target, spare = "A", "C", "B"
        initial = {"pegs": {p: [] for p in pegs}, "peg_names": pegs}
        initial["pegs"][source] = list(range(complexity, 0, -1))
        goal = {"pegs": {p: [] for p in pegs}, "peg_names": pegs}
        goal["pegs"][target] = list(range(complexity, 0, -1))
        metadata = {
            "seed": seed,
            "template_id": template_id,
            "source": source,
            "target": target,
            "spare": spare,
            "peg_index_map": {"0": "A", "1": "B", "2": "C"},
            "benchmark_family": "puzzlezoo_core",
            "generator_version": "puzzlezoo_v1",
            "puzzlezoo": True,
            "puzzlezoo_output_format": "moves = [[disk, from_peg, to_peg], ...]",
        }
        problem = Problem(
            problem_id=f"apple_hanoi_n{complexity:02d}_seed{seed:04d}_template{template_id:02d}",
            environment=self.name,
            complexity=complexity,
            initial_state=initial,
            goal_state=goal,
            natural_language_prompt="",
            min_steps=(2**complexity) - 1,
            max_steps=max(1, 2 * ((2**complexity) - 1)),
            metadata=metadata,
        )
        actions = self.oracle_solution(problem)
        return problem.model_copy(
            update={"natural_language_prompt": self.render_prompt(problem, template_id), "oracle_solution": actions}
        )

    def legal_actions(self, state: State) -> list[Action]:
        """Legal disk moves from `state`: pop top of any non-empty peg onto a peg whose top is larger."""
        pegs = state["pegs"]
        actions: list[Action] = []
        for src, disks in pegs.items():
            if not disks:
                continue
            disk = disks[-1]
            for dst, dst_disks in pegs.items():
                if src == dst:
                    continue
                if not dst_disks or dst_disks[-1] > disk:
                    actions.append({"type": "move", "disk": disk, "from": src, "to": dst})
        return actions

    def step(self, state: State, action: Action) -> StepResult:
        """Move a top disk between pegs; rejects malformed moves, empty sources, mismatched disks, or larger-on-smaller placements."""
        pegs = copy.deepcopy(state["pegs"])
        src = action.get("from")
        dst = action.get("to")
        disk = action.get("disk")
        if action.get("type") != "move" or src not in pegs or dst not in pegs or src == dst:
            return StepResult(valid=False, next_state=state, error_type=FailureType.INVALID_TRANSITION, message="Malformed Hanoi move.")
        if not pegs[src]:
            return StepResult(valid=False, next_state=state, error_type=FailureType.INVALID_TRANSITION, message="Source peg is empty.")
        if pegs[src][-1] != disk:
            return StepResult(valid=False, next_state=state, error_type=FailureType.INVALID_TRANSITION, message="Disk is not on top of source peg.")
        if pegs[dst] and pegs[dst][-1] < disk:
            return StepResult(valid=False, next_state=state, error_type=FailureType.CONSTRAINT_VIOLATION, message="Cannot place a larger disk on a smaller disk.")
        pegs[src].pop()
        pegs[dst].append(disk)
        return StepResult(valid=True, next_state={"pegs": pegs, "peg_names": state.get("peg_names", list(pegs))})

    def is_goal(self, state: State, goal: State) -> bool:
        """True when the peg stacks exactly match the goal stacks."""
        return state.get("pegs") == goal.get("pegs")

    def oracle_solution(self, problem: Problem) -> list[Action]:
        """Return the optimal `2**n - 1` move sequence using the standard recursive Hanoi algorithm."""
        source = problem.metadata["source"]
        target = problem.metadata["target"]
        spare = problem.metadata["spare"]
        actions: list[Action] = []

        def solve(n: int, src: str, dst: str, aux: str) -> None:
            """Recursively move `n` disks from `src` to `dst` using `aux`, appending each move to the outer list."""
            if n == 0:
                return
            solve(n - 1, src, aux, dst)
            actions.append({"type": "move", "disk": n, "from": src, "to": dst})
            solve(n - 1, aux, dst, src)

        solve(problem.complexity, source, target, spare)
        return actions

    def render_prompt(self, problem: Problem, template_id: int) -> str:
        """Render the Hanoi prompt for the chosen template (PuzzleZoo, single-step, or default JSON-list form)."""
        source = problem.metadata["source"]
        target = problem.metadata["target"]
        n = problem.complexity
        if problem.metadata.get("puzzlezoo"):
            return (
                f"We need to solve the Tower of Hanoi problem with {n} disks. "
                "There are three pegs numbered 0, 1, and 2. Initially all disks are on peg 0, "
                "with disk 1 smallest and larger disks below smaller disks. Move all disks to peg 2. "
                "Only one disk may move at a time, and a larger disk may never be placed on a smaller disk. "
                "Return the answer exactly as moves = [[disk, from_peg, to_peg], ...]."
            )
        if template_id == 1:
            return f"Solve Tower of Hanoi with {n} disks from peg {source} to peg {target}. After each move, list the resulting peg state and whether the rules are satisfied."
        if template_id == 2:
            return f"You are solving Tower of Hanoi one move at a time. Current state: {problem.initial_state}. Goal: {problem.goal_state}. Return exactly one legal next move as JSON."
        return f"Solve Tower of Hanoi with {n} disks. Move all disks from peg {source} to peg {target}. Return the sequence of moves as JSON."

    def action_label(self, action: Action) -> str:
        """Human-readable label like `move disk 3 from peg A to peg C`."""
        return f"move disk {action.get('disk')} from peg {action.get('from')} to peg {action.get('to')}"

    def strategy_hint(self, problem: Problem, state: State) -> str:
        """Reminder of the no-larger-on-smaller rule plus the current smallest movable disk and target peg."""
        legal = self.legal_actions(state)
        smallest = min((action["disk"] for action in legal), default=None)
        target = problem.metadata.get("target")
        return (
            f"Move only a top disk; never place a larger disk on a smaller disk. "
            f"The smallest currently movable disk is {smallest}. Target peg is {target}."
        )

    def progress_score(self, state: State, goal: State) -> float:
        """Fraction of disks already on the target peg (1.0 when all `n` disks are stacked there)."""
        target_pegs = [peg for peg, disks in goal["pegs"].items() if disks]
        if not target_pegs:
            return 1.0 if self.is_goal(state, goal) else 0.0
        target = target_pegs[0]
        total = sum(len(disks) for disks in goal["pegs"].values())
        on_target = len(state["pegs"].get(target, []))
        return on_target / max(1, total)

    def normalize_candidate_move(self, problem: Problem, raw_move, state: State | None = None) -> Action:
        """Coerce a `[disk, from, to]` list or partial dict into a `{type:"move", disk, from, to}` action with peg names mapped from indices."""
        del state
        if isinstance(raw_move, dict):
            action = dict(raw_move)
            if "from_peg" in action and "from" not in action:
                action["from"] = action["from_peg"]
            if "to_peg" in action and "to" not in action:
                action["to"] = action["to_peg"]
            if "from" in action:
                action["from"] = self._normalize_peg(problem, action["from"])
            if "to" in action:
                action["to"] = self._normalize_peg(problem, action["to"])
            action.setdefault("type", "move")
            if "disk" in action:
                action["disk"] = int(action["disk"])
            return action
        if not isinstance(raw_move, (list, tuple)) or len(raw_move) != 3:
            raise ValueError("Hanoi move must be [disk, from, to].")
        disk, src, dst = raw_move
        return {
            "type": "move",
            "disk": int(disk),
            "from": self._normalize_peg(problem, src),
            "to": self._normalize_peg(problem, dst),
        }

    def _normalize_peg(self, problem: Problem, value) -> str:
        mapping = problem.metadata.get("peg_index_map", {"0": "A", "1": "B", "2": "C"})
        key = str(value)
        if key in mapping:
            return mapping[key]
        if key in {"A", "B", "C"}:
            return key
        raise ValueError(f"Unknown peg: {value!r}")
