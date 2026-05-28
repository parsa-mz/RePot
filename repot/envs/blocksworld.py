from __future__ import annotations

import random
from collections import deque
from itertools import zip_longest

from repot.core.schemas import Action, EnvCapabilities, FailureType, Problem, State, StepResult
from repot.core.env import PuzzleEnv


class BlocksWorldEnv(PuzzleEnv):
    """BlocksWorld puzzle: rearrange labeled blocks across stacks by moving one clear block at a time."""
    name = "blocksworld"

    def capabilities(self, problem: Problem) -> EnvCapabilities:
        """BlocksWorld uses the state-tracking capability profile."""
        del problem
        return EnvCapabilities(profile="state_tracking")

    def generate(self, seed: int, complexity: int, template_id: int = 0) -> Problem:
        """Generate a random BlocksWorld instance with `complexity` blocks across 3-4 stacks."""
        rng = random.Random(seed)
        blocks = [chr(ord("A") + i) for i in range(complexity)]
        stack_count = 3 if complexity <= 8 else 4
        initial_stacks = [[] for _ in range(stack_count)]
        goal_stacks = [[] for _ in range(stack_count)]
        for block in blocks:
            initial_stacks[rng.randrange(stack_count)].append(block)
            goal_stacks[rng.randrange(stack_count)].append(block)
        initial = {"stacks": initial_stacks, "blocks": blocks}
        goal = {"stacks": goal_stacks, "blocks": blocks}
        problem = Problem(
            problem_id=f"blocks_n{complexity:02d}_seed{seed:04d}_template{template_id:02d}",
            environment=self.name,
            complexity=complexity,
            initial_state=initial,
            goal_state=goal,
            natural_language_prompt="",
            max_steps=max(16, complexity * 8),
            metadata={"seed": seed, "template_id": template_id, "stacks": stack_count},
        )
        actions = self.oracle_solution(problem)
        return problem.model_copy(update={"natural_language_prompt": self.render_prompt(problem, template_id), "oracle_solution": actions, "min_steps": len(actions)})

    def generate_puzzlezoo(self, seed: int, complexity: int, template_id: int = 0) -> Problem:
        """Generate the PuzzleZoo BlocksWorld interleave instance: split-then-interleave the two halves into stack 2."""
        blocks = [chr(ord("A") + i) for i in range(complexity)]
        split = (complexity + 1) // 2
        left = blocks[:split]
        right = blocks[split:]
        initial_stacks = [left, right, []]
        interleaved = [block for pair in zip_longest(left, right) for block in pair if block is not None]
        goal_stacks = [[], [], interleaved]
        metadata = {
            "seed": seed,
            "template_id": template_id,
            "stacks": 3,
            "benchmark_family": "puzzlezoo_core",
            "generator_version": "puzzlezoo_v1",
            "puzzlezoo": True,
            "puzzlezoo_output_format": "moves = [[block, from_stack, to_stack], ...]",
        }
        problem = Problem(
            problem_id=f"apple_blocks_n{complexity:02d}_seed{seed:04d}_template{template_id:02d}",
            environment=self.name,
            complexity=complexity,
            initial_state={"stacks": initial_stacks, "blocks": blocks},
            goal_state={"stacks": goal_stacks, "blocks": blocks},
            natural_language_prompt="",
            max_steps=max(16, complexity * 10),
            metadata=metadata,
        )
        actions = self.oracle_solution(problem)
        return problem.model_copy(
            update={
                "natural_language_prompt": self.render_prompt(problem, template_id),
                "oracle_solution": actions,
                "min_steps": len(actions),
            }
        )

    def legal_actions(self, state: State) -> list[Action]:
        """Legal moves: take the top block of any non-empty stack and place it on any other stack."""
        stacks = state["stacks"]
        actions: list[Action] = []
        for i, stack in enumerate(stacks):
            if not stack:
                continue
            block = stack[-1]
            for j in range(len(stacks)):
                if i != j:
                    actions.append({"type": "move", "block": block, "from": i, "to": j})
        return actions

    def step(self, state: State, action: Action) -> StepResult:
        """Pop `block` from stack `from` and push it onto stack `to`; reject if the block is not the top of `from`."""
        stacks = [list(stack) for stack in state["stacks"]]
        src = action.get("from")
        dst = action.get("to")
        block = action.get("block") or action.get("move")
        if not isinstance(src, int) or not isinstance(dst, int) or src == dst or not (0 <= src < len(stacks)) or not (0 <= dst < len(stacks)):
            return StepResult(valid=False, next_state=state, error_type=FailureType.INVALID_TRANSITION, message="Malformed BlocksWorld move.")
        if not stacks[src] or stacks[src][-1] != block:
            return StepResult(valid=False, next_state=state, error_type=FailureType.INVALID_TRANSITION, message="Block is not clear/on top of source stack.")
        stacks[src].pop()
        stacks[dst].append(block)
        return StepResult(valid=True, next_state={"stacks": stacks, "blocks": state.get("blocks", [])})

    def is_goal(self, state: State, goal: State) -> bool:
        """Goal: every stack matches the goal stacks list-for-list."""
        return state.get("stacks") == goal.get("stacks")

    def oracle_solution(self, problem: Problem) -> list[Action]:
        """Return BFS solution for small `complexity`, otherwise a constructive bottom-up build."""
        if problem.complexity <= 7:
            solution = self._bfs_solution(problem)
            if solution is not None:
                return solution
        return self._constructive_solution(problem)

    def _bfs_solution(self, problem: Problem) -> list[Action] | None:
        queue = deque([(problem.initial_state, [])])
        seen = {self._state_key(problem.initial_state)}
        max_seen = 200_000
        while queue and len(seen) < max_seen:
            state, path = queue.popleft()
            if self.is_goal(state, problem.goal_state):
                return path
            for action in self.legal_actions(state):
                result = self.step(state, action)
                key = self._state_key(result.next_state)
                if key not in seen:
                    seen.add(key)
                    queue.append((result.next_state, path + [action]))
        return None

    def _constructive_solution(self, problem: Problem) -> list[Action]:
        state = {"stacks": [list(s) for s in problem.initial_state["stacks"]], "blocks": problem.initial_state.get("blocks", [])}
        goal = problem.goal_state["stacks"]
        actions: list[Action] = []
        protected = [0 for _ in state["stacks"]]
        for goal_idx, goal_stack in enumerate(goal):
            for block in goal_stack:
                loc = self._find_block(state["stacks"], block)
                if loc[0] == goal_idx and loc[1] < protected[goal_idx]:
                    continue
                while state["stacks"][loc[0]][-1] != block:
                    if len(state["stacks"][loc[0]]) <= protected[loc[0]]:
                        raise RuntimeError(f"Cannot clear protected stack while solving {problem.problem_id}")
                    top = state["stacks"][loc[0]][-1]
                    dst = self._buffer_stack(loc[0], goal_idx, len(state["stacks"]))
                    action = {"type": "move", "block": top, "from": loc[0], "to": dst}
                    state = self.step(state, action).next_state
                    actions.append(action)
                    loc = self._find_block(state["stacks"], block)
                if loc[0] != goal_idx:
                    action = {"type": "move", "block": block, "from": loc[0], "to": goal_idx}
                    state = self.step(state, action).next_state
                    actions.append(action)
                protected[goal_idx] += 1
        return actions

    def _buffer_stack(self, src: int, goal_idx: int, stack_count: int) -> int:
        for candidate in range(stack_count):
            if candidate not in {src, goal_idx}:
                return candidate
        raise RuntimeError("BlocksWorld constructive solver needs at least three stacks.")

    def _find_block(self, stacks: list[list[str]], block: str) -> tuple[int, int]:
        for i, stack in enumerate(stacks):
            if block in stack:
                return i, stack.index(block)
        raise ValueError(f"Missing block: {block}")

    def _state_key(self, state: State) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(stack) for stack in state["stacks"])

    def render_prompt(self, problem: Problem, template_id: int) -> str:
        """Render the BlocksWorld natural-language prompt; PuzzleZoo and template variants differ in tone."""
        if problem.metadata.get("puzzlezoo"):
            return (
                f"Solve BlocksWorld with {problem.complexity} blocks. "
                f"Initial stacks: {problem.initial_state['stacks']}. Goal stacks: {problem.goal_state['stacks']}. "
                "Only the top block of any stack may be moved, and one block moves at a time. "
                "Stacks are numbered 0, 1, and 2. "
                "Return the answer exactly as moves = [[block, from_stack, to_stack], ...]."
            )
        if template_id == 1:
            return f"Solve BlocksWorld. Initial stacks: {problem.initial_state['stacks']}. Goal stacks: {problem.goal_state['stacks']}. After each move, list resulting stacks and whether the moved block was clear."
        if template_id == 2:
            return f"You are solving BlocksWorld one move at a time. Current state: {problem.initial_state}. Goal: {problem.goal_state}. Return exactly one legal block move as JSON."
        return f"Solve BlocksWorld from initial stacks {problem.initial_state['stacks']} to goal stacks {problem.goal_state['stacks']}. Return moves as JSON."

    def action_label(self, action: Action) -> str:
        """Render a BlocksWorld move as `move clear block X from stack i to stack j`."""
        return f"move clear block {action.get('block')} from stack {action.get('from')} to stack {action.get('to')}"

    def strategy_hint(self, problem: Problem, state: State) -> str:
        """Hint listing currently clear (top-of-stack) blocks and the goal layout."""
        clear = [stack[-1] for stack in state["stacks"] if stack]
        return (
            f"Only clear/top blocks can move. Clear blocks now: {clear}. "
            f"Build toward goal stacks {problem.goal_state['stacks']} without burying blocks needed next."
        )

    def progress_score(self, state: State, goal: State) -> float:
        """Fraction of goal-stack positions correct from the bottom up; 1.0 at goal."""
        if self.is_goal(state, goal):
            return 1.0
        correct = 0
        total = sum(len(stack) for stack in goal["stacks"])
        for current_stack, goal_stack in zip(state["stacks"], goal["stacks"], strict=False):
            for current, expected in zip(current_stack, goal_stack, strict=False):
                if current != expected:
                    break
                correct += 1
        return correct / max(1, total)

    def normalize_candidate_move(self, problem: Problem, raw_move, state: State | None = None) -> Action:
        """Coerce a model-emitted move into a `{type:"move", block, from, to}` action with integer stack indices."""
        del problem, state
        if isinstance(raw_move, dict):
            action = dict(raw_move)
            action.setdefault("type", "move")
            if "move" in action and "block" not in action:
                action["block"] = action["move"]
            if "from_stack" in action and "from" not in action:
                action["from"] = action["from_stack"]
            if "to_stack" in action and "to" not in action:
                action["to"] = action["to_stack"]
            action["from"] = self._normalize_stack(action["from"])
            action["to"] = self._normalize_stack(action["to"])
            return action
        if not isinstance(raw_move, (list, tuple)) or len(raw_move) != 3:
            raise ValueError("BlocksWorld move must be [block, from_stack, to_stack].")
        block, src, dst = raw_move
        return {
            "type": "move",
            "block": str(block),
            "from": self._normalize_stack(src),
            "to": self._normalize_stack(dst),
        }

    def _normalize_stack(self, value) -> int:
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if text.startswith("stack_"):
            text = text.removeprefix("stack_")
        return int(text)
