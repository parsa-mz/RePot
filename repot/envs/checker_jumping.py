from __future__ import annotations

from collections import deque

from repot.core.schemas import Action, EnvCapabilities, FailureType, Problem, State, StepResult
from repot.core.env import PuzzleEnv


class CheckerJumpingEnv(PuzzleEnv):
    """PuzzleEnv for the linear checker-jumping puzzle: swap two colored runs across a single empty cell using slides and jumps."""
    name = "checker_jumping"

    def capabilities(self, problem: Problem) -> EnvCapabilities:
        """Report the puzzle-grid capability profile for checker jumping."""
        del problem
        return EnvCapabilities(profile="puzzle_grid")

    def generate(self, seed: int, complexity: int, template_id: int = 0) -> Problem:
        """Build a default B/Y instance with `complexity` checkers per side and attach the BFS oracle solution."""
        initial = {"board": "B" * complexity + "_" + "Y" * complexity}
        goal = {"board": "Y" * complexity + "_" + "B" * complexity}
        problem = Problem(
            problem_id=f"checker_n{complexity:02d}_seed{seed:04d}_template{template_id:02d}",
            environment=self.name,
            complexity=complexity,
            initial_state=initial,
            goal_state=goal,
            natural_language_prompt="",
            min_steps=None,
            max_steps=complexity * complexity + 4 * complexity + 8,
            metadata={"seed": seed, "template_id": template_id},
        )
        actions = self.oracle_solution(problem)
        return problem.model_copy(update={"natural_language_prompt": self.render_prompt(problem, template_id), "oracle_solution": actions, "min_steps": len(actions)})

    def generate_puzzlezoo(self, seed: int, complexity: int, template_id: int = 0) -> Problem:
        """Build the PuzzleZoo R/B variant with `[color, from, to]` output format and attach the oracle solution."""
        initial = {"board": "R" * complexity + "_" + "B" * complexity, "left_piece": "R", "right_piece": "B"}
        goal = {"board": "B" * complexity + "_" + "R" * complexity, "left_piece": "R", "right_piece": "B"}
        metadata = {
            "seed": seed,
            "template_id": template_id,
            "benchmark_family": "puzzlezoo_core",
            "generator_version": "puzzlezoo_v1",
            "puzzlezoo": True,
            "puzzlezoo_output_format": "moves = [[color, from_position, to_position], ...]",
        }
        problem = Problem(
            problem_id=f"apple_checker_n{complexity:02d}_seed{seed:04d}_template{template_id:02d}",
            environment=self.name,
            complexity=complexity,
            initial_state=initial,
            goal_state=goal,
            natural_language_prompt="",
            min_steps=None,
            max_steps=complexity * complexity + 4 * complexity + 8,
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
        """Slides and jumps into the empty cell that respect each color's allowed direction."""
        board = state["board"]
        empty = board.index("_")
        left_piece, right_piece = self._pieces(state)
        actions: list[Action] = []
        for idx, piece in enumerate(board):
            if piece not in {left_piece, right_piece}:
                continue
            direction = 1 if piece == left_piece else -1
            slide = idx + direction
            jump = idx + 2 * direction
            if slide == empty:
                actions.append({"type": "slide", "from": idx, "to": empty, "piece": piece})
            if jump == empty and 0 <= idx + direction < len(board) and board[idx + direction] not in {piece, "_"}:
                actions.append({"type": "jump", "from": idx, "to": empty, "piece": piece})
        return actions

    def step(self, state: State, action: Action) -> StepResult:
        """Apply a one-step slide or two-step jump-over-opposite-color move into the empty cell, validating direction and contents."""
        board = list(state["board"])
        src = action.get("from")
        dst = action.get("to")
        if not isinstance(src, int) or not isinstance(dst, int) or not (0 <= src < len(board)) or not (0 <= dst < len(board)):
            return StepResult(valid=False, next_state=state, error_type=FailureType.INVALID_TRANSITION, message="Malformed checker move.")
        piece = board[src]
        left_piece, right_piece = self._pieces(state)
        if piece not in {left_piece, right_piece} or board[dst] != "_":
            return StepResult(valid=False, next_state=state, error_type=FailureType.INVALID_TRANSITION, message="Move must move a checker into the empty cell.")
        direction = 1 if piece == left_piece else -1
        dist = dst - src
        move_type = action.get("type")
        if dist == direction and move_type in {"slide", "move", None}:
            pass
        elif dist == 2 * direction and move_type in {"jump", "move", None} and board[src + direction] not in {piece, "_"}:
            pass
        else:
            return StepResult(valid=False, next_state=state, error_type=FailureType.INVALID_TRANSITION, message="Illegal slide or jump.")
        board[dst], board[src] = board[src], "_"
        next_state = {"board": "".join(board)}
        if "left_piece" in state:
            next_state["left_piece"] = state["left_piece"]
            next_state["right_piece"] = state["right_piece"]
        if next_state["board"].count("_") != 1:
            return StepResult(valid=False, next_state=state, error_type=FailureType.VERIFIER_DISAGREEMENT, message="Checker count invariant failed.")
        return StepResult(valid=True, next_state=next_state)

    def is_goal(self, state: State, goal: State) -> bool:
        """True when the board string equals the goal board (colors fully swapped)."""
        return state.get("board") == goal.get("board")

    def oracle_solution(self, problem: Problem) -> list[Action]:
        """Solve via BFS over board strings; raises if no path exists."""
        start = problem.initial_state["board"]
        target = problem.goal_state["board"]
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            board, path = queue.popleft()
            if board == target:
                return path
            current = self._state_like(problem.initial_state, board)
            for action in self.legal_actions(current):
                result = self.step(current, action)
                if not result.valid:
                    continue
                nxt = result.next_state["board"]
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, path + [action]))
        raise RuntimeError(f"No checker solution found for {problem.problem_id}")

    def render_prompt(self, problem: Problem, template_id: int) -> str:
        """Render the task description (PuzzleZoo or default-template variant) including rules and output format."""
        n = problem.complexity
        if problem.metadata.get("puzzlezoo"):
            return (
                f"We need to solve the Checkers Jumping puzzle with {n} red checkers and {n} blue checkers. "
                f"The start board is {problem.initial_state['board']} and the target board is {problem.goal_state['board']}. "
                "Positions are 0-based. R pieces move only right; B pieces move only left. "
                "A checker may slide one step into the empty square or jump over exactly one opposite-color checker "
                "into the empty square. No checker may move backward. "
                "Return the answer exactly as moves = [[color, from_position, to_position], ...]."
            )
        rules = (
            "The board is a linear string with 0-based positions. "
            "B pieces move only right; Y pieces move only left; _ is the single empty cell. "
            "A slide moves one position into _. A jump moves two positions over exactly one opposite-colored checker into _. "
            "Return actions as objects like {\"type\":\"slide\",\"from\":1,\"to\":2,\"piece\":\"B\"}."
        )
        if template_id == 1:
            return f"Solve Checker Jumping for board {problem.initial_state['board']} to {problem.goal_state['board']}. {rules} After each move, output the resulting board and whether the slide/jump rule is satisfied."
        if template_id == 2:
            return f"You are solving Checker Jumping one move at a time. {rules} Current state: {problem.initial_state}. Goal: {problem.goal_state}. Return exactly one legal next move as JSON."
        return f"Solve Checker Jumping with {n} checkers per side. Start board: {problem.initial_state['board']}. Goal board: {problem.goal_state['board']}. {rules} Return moves as JSON."

    def action_label(self, action: Action) -> str:
        """Human-readable label like `B slide from position 1 to empty position 2`."""
        return (
            f"{action.get('piece')} {action.get('type')} "
            f"from position {action.get('from')} to empty position {action.get('to')}"
        )

    def strategy_hint(self, problem: Problem, state: State) -> str:
        """Hint reminding the solver of the empty-cell location and per-color forward-progress rule."""
        board = state["board"]
        empty = board.index("_")
        return (
            f"The empty cell is at position {empty}. B pieces must progress right and Y pieces must progress left. "
            "Prefer moves that increase swapped order and avoid immediately undoing the previous board unless forced."
        )

    def progress_score(self, state: State, goal: State) -> float:
        """Average per-piece normalized closeness to its goal position, in [0, 1]."""
        board = state["board"]
        goal_board = goal["board"]
        if board == goal_board:
            return 1.0
        left_piece, right_piece = self._pieces(state)
        max_distance = max(1, len(board) - 1)
        components: list[float] = []
        for piece in (left_piece, right_piece):
            current_positions = sorted(i for i, p in enumerate(board) if p == piece)
            goal_positions = sorted(i for i, p in enumerate(goal_board) if p == piece)
            for cur, goal_pos in zip(current_positions, goal_positions):
                components.append((max_distance - abs(cur - goal_pos)) / max_distance)
        if not components:
            return 0.0
        return sum(components) / len(components)

    def normalize_candidate_move(self, problem: Problem, raw_move, state: State | None = None) -> Action:
        """Coerce a `[color, from, to]` list or partial dict into a `{type, from, to, piece}` action with integer indices."""
        del problem, state
        if isinstance(raw_move, dict):
            action = dict(raw_move)
            if "from_position" in action and "from" not in action:
                action["from"] = action["from_position"]
            if "to_position" in action and "to" not in action:
                action["to"] = action["to_position"]
            action["from"] = int(action["from"])
            action["to"] = int(action["to"])
            action.setdefault("piece", action.get("color"))
            action.setdefault("type", self._move_type(action["from"], action["to"]))
            return action
        if not isinstance(raw_move, (list, tuple)) or len(raw_move) != 3:
            raise ValueError("Checker move must be [color, from, to].")
        piece, src, dst = raw_move
        src_i = int(src)
        dst_i = int(dst)
        return {"type": self._move_type(src_i, dst_i), "from": src_i, "to": dst_i, "piece": str(piece)}

    def _pieces(self, state: State) -> tuple[str, str]:
        return state.get("left_piece", "B"), state.get("right_piece", "Y")

    def _state_like(self, template: State, board: str) -> State:
        state = {"board": board}
        if "left_piece" in template:
            state["left_piece"] = template["left_piece"]
            state["right_piece"] = template["right_piece"]
        return state

    def _move_type(self, src: int, dst: int) -> str:
        return "jump" if abs(dst - src) == 2 else "slide"
