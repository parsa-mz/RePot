from __future__ import annotations

import json
from typing import Any

from repot.core.schemas import Problem
from repot.core.tools.benchmarks import make_env
from repot.core.env import PuzzleEnv


def generate_problems(config: dict[str, Any]) -> list[Problem]:
    """Materialize all `Problem` instances declared in the dataset config across envs/complexities/seeds."""
    dataset = config["dataset"]
    templates = dataset.get("templates", [0, 1, 2])
    instances_per_level = int(dataset.get("instances_per_level", 1))
    puzzlezoo = bool(dataset.get("puzzlezoo", False))
    benchmark_family = dataset.get(
        "benchmark_family",
        "puzzlezoo_core" if puzzlezoo else "statetracebench_core",
    )
    generator_version = dataset.get(
        "generator_version",
        "puzzlezoo_v1" if puzzlezoo else "statetracebench_v1",
    )
    problems: list[Problem] = []
    skipped: list[str] = []
    for env_name, env_cfg in dataset["environments"].items():
        env = make_env(env_name)
        for complexity in env_cfg["complexities"]:
            for seed in range(instances_per_level):
                for template_id in templates:
                    try:
                        if puzzlezoo:
                            problem = env.generate_puzzlezoo(
                                seed=seed,
                                complexity=int(complexity),
                                template_id=int(template_id),
                            )
                        else:
                            problem = env.generate(seed=seed, complexity=int(complexity), template_id=int(template_id))
                        problems.append(
                            _with_dataset_metadata(
                                env=env,
                                problem=problem,
                                benchmark_family=benchmark_family,
                                generator_version=generator_version,
                                puzzlezoo=puzzlezoo,
                            )
                        )
                    except RuntimeError as exc:
                        skipped.append(f"{env_name}/n={complexity}/seed={seed}/template={template_id}: {exc}")
    if skipped and not problems:
        raise RuntimeError("No problems generated. First skipped case: " + skipped[0])
    config.setdefault("_generation_report", {})["skipped"] = skipped
    return problems


def _with_dataset_metadata(
    env: PuzzleEnv,
    problem: Problem,
    benchmark_family: str,
    generator_version: str,
    puzzlezoo: bool,
) -> Problem:
    metadata = dict(problem.metadata)
    metadata.update(
        {
            "benchmark_family": metadata.get("benchmark_family", benchmark_family),
            "generator_version": metadata.get("generator_version", generator_version),
            "solvable": bool(problem.oracle_solution),
            "oracle_min_steps": len(problem.oracle_solution),
            "branching_stats": _branching_stats(env, problem),
            "puzzlezoo": bool(metadata.get("puzzlezoo", puzzlezoo)),
        }
    )
    return problem.model_copy(
        update={
            "metadata": metadata,
            "min_steps": problem.min_steps if problem.min_steps is not None else len(problem.oracle_solution),
        }
    )


def _branching_stats(env: PuzzleEnv, problem: Problem) -> dict[str, float | int | None]:
    state = json.loads(json.dumps(problem.initial_state))
    counts: list[int] = []
    counts.append(len(env.legal_actions(state)))
    for action in problem.oracle_solution:
        result = env.step(state, action)
        if not result.valid:
            break
        state = result.next_state
        counts.append(len(env.legal_actions(state)))
    if not counts:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(counts), "max": max(counts), "mean": sum(counts) / len(counts)}
