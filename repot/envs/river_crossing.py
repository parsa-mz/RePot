from __future__ import annotations

import itertools
from collections import deque

from repot.core.schemas import Action, EnvCapabilities, FailureType, Problem, State, StepResult
from repot.core.env import PuzzleEnv


class RiverCrossingEnv(PuzzleEnv):
    """PuzzleEnv for river-crossing puzzles (farmer/wolf/goat/cabbage scaling and the PuzzleZoo actor/agent variant)."""

    name = "river_crossing"

    def capabilities(self, problem: Problem) -> EnvCapabilities:
        """Report the constraint-maintenance capability profile for river crossing."""
        del problem
        return EnvCapabilities(profile="constraint_maintenance")

    def generate(self, seed: int, complexity: int, template_id: int = 0) -> Problem:
        """Build a default farmer/wolf/goat/cabbage instance with `complexity` triples and BFS oracle solution."""
        actors = ["farmer"]
        unsafe_pairs: list[list[str]] = []
        for i in range(1, complexity + 1):
            wolf = f"wolf{i}"
            goat = f"goat{i}"
            cabbage = f"cabbage{i}"
            actors.extend([wolf, goat, cabbage])
            unsafe_pairs.extend([[wolf, goat], [goat, cabbage]])
        initial = {
            "left": sorted(actors),
            "right": [],
            "boat_side": "left",
            "actors": sorted(actors),
            "unsafe_pairs": unsafe_pairs,
            "boat_capacity": 1 if complexity == 1 else 2,
        }
        goal = {**initial, "left": [], "right": sorted(actors), "boat_side": "right"}
        problem = Problem(
            problem_id=f"river_n{complexity:02d}_seed{seed:04d}_template{template_id:02d}",
            environment=self.name,
            complexity=complexity,
            initial_state=initial,
            goal_state=goal,
            natural_language_prompt="",
            max_steps=40 * complexity,
            metadata={"seed": seed, "template_id": template_id},
        )
        actions = self.oracle_solution(problem)
        return problem.model_copy(update={"natural_language_prompt": self.render_prompt(problem, template_id), "oracle_solution": actions, "min_steps": len(actions)})

    def generate_puzzlezoo(self, seed: int, complexity: int, template_id: int = 0) -> Problem:
        """Build the PuzzleZoo actor/agent variant where each actor must be guarded from non-matching agents by its own agent."""
        people: list[str] = []
        unsafe_pairs: list[list[str]] = []
        for idx in range(1, complexity + 1):
            actor = f"actor{idx}"
            agent = f"agent{idx}"
            people.extend([actor, agent])
        for idx in range(1, complexity + 1):
            actor = f"actor{idx}"
            for other in range(1, complexity + 1):
                if other != idx:
                    unsafe_pairs.append([actor, f"agent{other}"])
        capacity = 2 if complexity == 1 else 3
        initial = {
            "left": sorted(people),
            "right": [],
            "boat_side": "left",
            "actors": sorted(people),
            "unsafe_pairs": unsafe_pairs,
            "boat_capacity": capacity,
            "variant": "apple_agents",
        }
        goal = {**initial, "left": [], "right": sorted(people), "boat_side": "right"}
        metadata = {
            "seed": seed,
            "template_id": template_id,
            "benchmark_family": "puzzlezoo_core",
            "generator_version": "puzzlezoo_v1",
            "puzzlezoo": True,
            "puzzlezoo_output_format": "moves = [[person1, person2, ...], ...]",
        }
        problem = Problem(
            problem_id=f"apple_river_n{complexity:02d}_seed{seed:04d}_template{template_id:02d}",
            environment=self.name,
            complexity=complexity,
            initial_state=initial,
            goal_state=goal,
            natural_language_prompt="",
            max_steps=40 * complexity,
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
        """Legal boat crossings (one or two passengers/cargo) under capacity + bank-safety constraints."""
        if state.get("variant") == "apple_agents":
            return self._puzzlezoo_legal_actions(state)
        side = state["boat_side"]
        source = state[side]
        movable = [a for a in source if a != "farmer"]
        capacity = int(state["boat_capacity"])
        actions: list[Action] = []
        for count in range(0, capacity + 1):
            for cargo in itertools.combinations(movable, count):
                action = {"type": "cross", "cargo": list(cargo), "from": side, "to": "right" if side == "left" else "left"}
                if self.step(state, action).valid:
                    actions.append(action)
        return actions

    def step(self, state: State, action: Action) -> StepResult:
        """Cross the boat with the given cargo; rejects capacity, source, or bank-safety violations."""
        if state.get("variant") == "apple_agents":
            return self._puzzlezoo_step(state, action)
        side = state["boat_side"]
        dst = "right" if side == "left" else "left"
        if action.get("type") != "cross" or action.get("from") != side or action.get("to") != dst:
            return StepResult(valid=False, next_state=state, error_type=FailureType.INVALID_TRANSITION, message="Boat must cross from its current side.")
        cargo = action.get("cargo", [])
        if not isinstance(cargo, list) or len(cargo) > int(state["boat_capacity"]):
            return StepResult(valid=False, next_state=state, error_type=FailureType.INVALID_TRANSITION, message="Cargo exceeds boat capacity.")
        if "farmer" in cargo:
            return StepResult(valid=False, next_state=state, error_type=FailureType.INVALID_TRANSITION, message="Cargo should omit farmer; farmer always operates boat.")
        source = set(state[side])
        if "farmer" not in source or any(item not in source for item in cargo):
            return StepResult(valid=False, next_state=state, error_type=FailureType.INVALID_TRANSITION, message="Cargo is not on source bank.")
        next_state = {
            "left": set(state["left"]),
            "right": set(state["right"]),
            "boat_side": dst,
            "actors": state["actors"],
            "unsafe_pairs": state["unsafe_pairs"],
            "boat_capacity": state["boat_capacity"],
        }
        for item in ["farmer", *cargo]:
            next_state[side].remove(item)
            next_state[dst].add(item)
        normalized = {
            "left": sorted(next_state["left"]),
            "right": sorted(next_state["right"]),
            "boat_side": next_state["boat_side"],
            "actors": state["actors"],
            "unsafe_pairs": state["unsafe_pairs"],
            "boat_capacity": state["boat_capacity"],
        }
        if not self._banks_safe(normalized):
            return StepResult(valid=False, next_state=state, error_type=FailureType.CONSTRAINT_VIOLATION, message="Unsafe predator/prey bank state.")
        return StepResult(valid=True, next_state=normalized)

    def _banks_safe(self, state: State) -> bool:
        for bank in ("left", "right"):
            occupants = set(state[bank])
            if "farmer" in occupants:
                continue
            for a, b in state["unsafe_pairs"]:
                if a in occupants and b in occupants:
                    return False
        return True

    def _puzzlezoo_legal_actions(self, state: State) -> list[Action]:
        side = state["boat_side"]
        source = state[side]
        capacity = int(state["boat_capacity"])
        actions: list[Action] = []
        for count in range(1, capacity + 1):
            for passengers in itertools.combinations(source, count):
                action = {
                    "type": "cross",
                    "passengers": list(passengers),
                    "from": side,
                    "to": "right" if side == "left" else "left",
                }
                if self._puzzlezoo_step(state, action).valid:
                    actions.append(action)
        return actions

    def _puzzlezoo_step(self, state: State, action: Action) -> StepResult:
        side = state["boat_side"]
        dst = "right" if side == "left" else "left"
        if action.get("type") != "cross" or action.get("from") != side or action.get("to") != dst:
            return StepResult(valid=False, next_state=state, error_type=FailureType.INVALID_TRANSITION, message="Boat must cross from its current side.")
        passengers = action.get("passengers", action.get("cargo", []))
        if not isinstance(passengers, list) or not passengers or len(passengers) > int(state["boat_capacity"]):
            return StepResult(valid=False, next_state=state, error_type=FailureType.INVALID_TRANSITION, message="Passenger count violates boat capacity.")
        source = set(state[side])
        if any(item not in source for item in passengers):
            return StepResult(valid=False, next_state=state, error_type=FailureType.INVALID_TRANSITION, message="Passenger is not on source bank.")
        next_state = {
            "left": set(state["left"]),
            "right": set(state["right"]),
            "boat_side": dst,
            "actors": state["actors"],
            "unsafe_pairs": state["unsafe_pairs"],
            "boat_capacity": state["boat_capacity"],
            "variant": "apple_agents",
        }
        for item in passengers:
            next_state[side].remove(item)
            next_state[dst].add(item)
        normalized = {
            "left": sorted(next_state["left"]),
            "right": sorted(next_state["right"]),
            "boat_side": next_state["boat_side"],
            "actors": state["actors"],
            "unsafe_pairs": state["unsafe_pairs"],
            "boat_capacity": state["boat_capacity"],
            "variant": "apple_agents",
        }
        if not self._puzzlezoo_banks_safe(normalized):
            return StepResult(valid=False, next_state=state, error_type=FailureType.CONSTRAINT_VIOLATION, message="Unsafe actor/agent bank state.")
        return StepResult(valid=True, next_state=normalized)

    def _puzzlezoo_banks_safe(self, state: State) -> bool:
        for bank in ("left", "right"):
            occupants = set(state[bank])
            for person in occupants:
                if not person.startswith("actor"):
                    continue
                idx = person.removeprefix("actor")
                own_agent = f"agent{idx}"
                other_agents = {item for item in occupants if item.startswith("agent") and item != own_agent}
                if other_agents and own_agent not in occupants:
                    return False
        return True

    def is_goal(self, state: State, goal: State) -> bool:
        """True when the right-bank occupants match the goal's right-bank occupants."""
        return sorted(state.get("right", [])) == sorted(goal.get("right", []))

    def oracle_solution(self, problem: Problem) -> list[Action]:
        """Solve via BFS over (left, right, boat_side) states; raises if no path exists."""
        queue = deque([(problem.initial_state, [])])
        seen = {self._state_key(problem.initial_state)}
        while queue:
            state, path = queue.popleft()
            if self.is_goal(state, problem.goal_state):
                return path
            for action in self.legal_actions(state):
                result = self.step(state, action)
                key = self._state_key(result.next_state)
                if key not in seen:
                    seen.add(key)
                    queue.append((result.next_state, path + [action]))
        raise RuntimeError(f"No river solution found for {problem.problem_id}")

    def _state_key(self, state: State) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        return (tuple(sorted(state["left"])), tuple(sorted(state["right"])), state["boat_side"])

    def render_prompt(self, problem: Problem, template_id: int) -> str:
        """Render the river-crossing prompt for the chosen template (PuzzleZoo, single-step, or default JSON-list form)."""
        if problem.metadata.get("puzzlezoo"):
            return (
                "Solve this river crossing puzzle. Everyone starts on the left bank and must end on the right bank. "
                f"Initial state: {problem.initial_state}. The boat can carry at most "
                f"{problem.initial_state['boat_capacity']} people and cannot cross empty. "
                "An actor cannot be left with a non-matching agent unless its matching agent is also present. "
                "Return the answer exactly as moves = [[person1, person2, ...], ...], where each inner list is one crossing."
            )
        if template_id == 1:
            return f"Solve this river crossing. State: {problem.initial_state}. After each crossing, list both banks and whether constraints are satisfied."
        if template_id == 2:
            return f"You are solving a river crossing one boat move at a time. Current state: {problem.initial_state}. Goal: all actors on the right bank. Return exactly one legal crossing as JSON."
        return f"Solve this river crossing puzzle. Initial state: {problem.initial_state}. Return crossings as JSON."

    def action_label(self, action: Action) -> str:
        """Human-readable label like ``cross farmer with goat from left to right``."""
        cargo = action.get("cargo", [])
        cargo_text = "farmer alone" if not cargo else "farmer with " + ", ".join(cargo)
        return f"cross {cargo_text} from {action.get('from')} to {action.get('to')}"

    def strategy_hint(self, problem: Problem, state: State) -> str:
        """Reminder of boat capacity, the unsafe pair list, and that every crossing moves the farmer."""
        del problem
        return (
            f"Boat capacity is {state['boat_capacity']} cargo item(s) plus farmer. "
            f"Unsafe pairs without farmer are {state['unsafe_pairs']}. "
            "Every crossing moves the farmer and must leave both banks safe."
        )

    def progress_score(self, state: State, goal: State) -> float:
        """Fraction of goal right-bank occupants already on the right bank."""
        if self.is_goal(state, goal):
            return 1.0
        target = set(goal.get("right", []))
        right = set(state.get("right", []))
        return len(right & target) / max(1, len(target))

    def normalize_candidate_move(self, problem: Problem, raw_move, state: State | None = None) -> Action:
        """Coerce a passenger/cargo list (or partial dict) into a `{type:"cross", from, to, cargo|passengers}` action."""
        current_state = state or problem.initial_state
        side = current_state["boat_side"]
        dst = "right" if side == "left" else "left"
        if isinstance(raw_move, dict):
            action = dict(raw_move)
            action.setdefault("type", "cross")
            action.setdefault("from", side)
            action.setdefault("to", dst)
            return action
        if not isinstance(raw_move, (list, tuple)):
            raise ValueError("River crossing move must be a passenger/cargo list.")
        if problem.initial_state.get("variant") == "apple_agents":
            return {"type": "cross", "passengers": [str(item) for item in raw_move], "from": side, "to": dst}
        return {"type": "cross", "cargo": [str(item) for item in raw_move], "from": side, "to": dst}
