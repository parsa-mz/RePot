"""``repot run`` — inference for PoT, RePoT, RePoT-A, VEX, CoT, SC.

Reads an experiment YAML, samples problems from the dataset, runs the
selected methods (optionally in parallel), and writes a trace JSONL.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from repot.cli._common import (
    build_agent_config,
    default_trace_path,
    filter_problems_by_config,
    load_model_config,
    resolve_workers,
    sample_problems,
)
from repot.core.schemas import FailureType, Problem, TraceRun, TraceStep
from repot.core.tools.benchmarks import make_env
from repot.core.llm import create_model_client
from repot.core.agent import AgentConfig, make_agent
from repot.core.utils.config import load_yaml
from repot.core.utils.jsonl import read_jsonl

_CONSOLE = Console()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `run` subcommand."""
    parser = subparsers.add_parser(
        "run",
        help="Run inference (PoT, RePoT, RePoT-A, VEX, CoT, SC) on a problem set.",
        description="Run inference methods on a problem set; emit a trace JSONL.",
    )
    parser.add_argument(
        "--config", default="configs/repot_main.yaml", help="Path to experiment YAML."
    )
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap on problems.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model alias from configs/models.yaml. Use 'local_dummy' for offline smoke.",
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Concurrent problem/method runs."
    )
    parser.add_argument("--traces", default=None, help="Trace JSONL output path.")
    parser.add_argument(
        "--sample",
        default="shuffle",
        choices=["ordered", "shuffle", "stratified"],
        help="Sampling mode.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--methods-only",
        default=None,
        help="Comma-separated subset of the config's methods to run (e.g. 'repot' or 'repot,vex').",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append new (problem_id, method) pairs to existing trace; otherwise overwrite.",
    )
    parser.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)
    model_cfg, model_label = load_model_config(cfg, args.model)
    worker_count = resolve_workers(args.workers, cfg, model_cfg)
    client = create_model_client(model_cfg)
    agent_cfg = build_agent_config(cfg.get("run", {}), model_cfg)

    dataset_path = Path(cfg["dataset"]["output"])
    if not dataset_path.exists():
        _CONSOLE.print(
            f"[red]Dataset file does not exist:[/red] {dataset_path}",
            style="bold",
        )
        return 2

    problems = [Problem.model_validate(row) for row in read_jsonl(dataset_path)]
    problems = filter_problems_by_config(problems, cfg)
    problems = sample_problems(problems, max_items=args.max_items, mode=args.sample, seed=args.seed)

    methods = list(cfg["methods"])
    if args.methods_only:
        wanted = {m.strip() for m in args.methods_only.split(",") if m.strip()}
        unknown = wanted - set(methods)
        if unknown:
            _CONSOLE.print(
                f"[red]--methods-only includes methods not in {args.config}:[/red] {sorted(unknown)}\n"
                f"Config methods: {methods}"
            )
            return 2
        methods = [m for m in methods if m in wanted]
        _CONSOLE.print(f"[dim]--methods-only: running[/dim] {methods}")

    tasks = [(problem, method) for problem in problems for method in methods]
    out = Path(args.traces or default_trace_path(cfg, model_label))
    out.parent.mkdir(parents=True, exist_ok=True)

    open_mode = "w"
    if args.resume and out.exists():
        done_keys = _load_done_keys(out)
        before = len(tasks)
        tasks = [(p, m) for (p, m) in tasks if (p.problem_id, m) not in done_keys]
        skipped = before - len(tasks)
        _CONSOLE.print(
            f"[dim]--resume: {skipped}/{before} tasks already in {out}, "
            f"running {len(tasks)} new.[/dim]"
        )
        open_mode = "a"

    if not tasks:
        _CONSOLE.print(f"Nothing to do — all (problem, method) pairs already in {out}.")
        return 0

    completed = 0
    t_start = time.perf_counter()

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=_CONSOLE,
    )
    with out.open(open_mode) as f, progress:
        progress_task = progress.add_task(f"Running {model_label}", total=len(tasks))
        if worker_count == 1:
            for problem, method in tasks:
                _write_run(f, _run_one_safe(problem, method, client, agent_cfg))
                completed += 1
                progress.advance(progress_task)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(_run_one_safe, problem, method, client, agent_cfg)
                    for problem, method in tasks
                ]
                for future in as_completed(futures):
                    _write_run(f, future.result())
                    completed += 1
                    progress.advance(progress_task)

    elapsed = time.perf_counter() - t_start
    _CONSOLE.print(
        f"[green]Wrote {completed} trace runs to {out}[/green]  "
        f"[dim]({elapsed:.0f}s, {completed / max(elapsed, 1e-6):.2f}/s)[/dim]"
    )
    return 0 if completed == len(tasks) else 1


def _run_one(problem: Problem, method: str, client, agent_cfg: AgentConfig) -> TraceRun:
    env = make_env(problem.environment)
    return make_agent(method).run(problem, env, client, agent_cfg)


def _run_one_safe(problem: Problem, method: str, client, agent_cfg: AgentConfig) -> TraceRun:
    try:
        return _run_one(problem, method, client, agent_cfg)
    except Exception as exc:  # noqa: BLE001
        message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        lowered = message.lower()
        error_type = (
            FailureType.PROVIDER_REJECTION
            if "invalid_prompt" in lowered or "flagged as potentially violating" in lowered
            else FailureType.API_ERROR
        )
        return TraceRun(
            problem_id=problem.problem_id,
            environment=problem.environment,
            complexity=problem.complexity,
            model=getattr(client, "model", "unknown"),
            method=method,
            success=False,
            stopped_reason="runner_exception",
            steps=[
                TraceStep(
                    problem_id=problem.problem_id,
                    model=getattr(client, "model", "unknown"),
                    method=method,
                    step=0,
                    current_state=problem.initial_state,
                    valid=False,
                    error_type=error_type,
                    message=message,
                    raw_model_output=traceback.format_exc(),
                )
            ],
            final_state=problem.initial_state,
            metadata={"runner_exception": True},
        )


def _write_run(file, run: TraceRun) -> None:
    file.write(json.dumps(run.model_dump(mode="json"), sort_keys=True) + "\n")
    file.flush()


def _load_done_keys(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = row.get("problem_id")
            method = row.get("method")
            if pid and method:
                done.add((pid, method))
    return done
