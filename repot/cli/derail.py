"""``repot derail`` — controlled mid-rollout recovery experiment (Derail-550).

For each source problem, replays the oracle plan to ~1/3 of the way, injects
one wrong action, then asks each recovery condition to take over from the
post-injection state. Writes a record JSONL plus a summary JSON.
"""
from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter, defaultdict
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
    load_model_config,
    resolve_workers,
    sample_problems,
)
from repot.core.schemas import FailureType, Problem
from repot.core.tools.benchmarks import make_env
from repot.core.evaluation import (
    RECOVERY_CONDITIONS,
    RecoveryRecord,
    make_recovery_case,
    run_recovery_condition,
)
from repot.core.llm import create_model_client
from repot.core.agent import AgentConfig
from repot.core.utils.config import load_yaml
from repot.core.utils.jsonl import read_jsonl

_CONSOLE = Console()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `derail` subcommand."""
    parser = subparsers.add_parser(
        "derail",
        help="Run the Derail-550 mid-rollout recovery experiment.",
        description=(
            "Inject one wrong action ~1/3 through the oracle plan, then ask each "
            "recovery condition to take over from the post-injection state."
        ),
    )
    parser.add_argument(
        "--config", default="configs/derail_experiment.yaml", help="Path to recovery YAML."
    )
    parser.add_argument(
        "--max-items", type=int, default=None, help="Optional cap on source problems."
    )
    parser.add_argument("--model", default=None, help="Model alias from configs/models.yaml.")
    parser.add_argument("--workers", type=int, default=None, help="Concurrent recovery calls.")
    parser.add_argument(
        "--sample",
        default="stratified",
        choices=["ordered", "shuffle", "stratified"],
        help="Sampling mode.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--output", default=None, help="Recovery JSONL output path.")
    parser.add_argument("--summary", default=None, help="Summary JSON output path.")
    parser.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)
    model_cfg, model_label = load_model_config(cfg, args.model)
    worker_count = resolve_workers(args.workers, cfg, model_cfg)
    client = create_model_client(model_cfg)
    agent_cfg = build_agent_config(cfg.get("run", {}), model_cfg)

    problems_path = Path(cfg["dataset"]["input"])
    if not problems_path.exists():
        _CONSOLE.print(f"[red]Dataset file does not exist:[/red] {problems_path}")
        return 2

    problems = [Problem.model_validate(row) for row in read_jsonl(problems_path)]
    env_filter = set(cfg.get("dataset", {}).get("environments", []))
    if env_filter:
        problems = [p for p in problems if p.environment in env_filter]
    problems = sample_problems(problems, max_items=args.max_items, mode=args.sample, seed=args.seed)

    conditions = cfg.get("recovery", {}).get("conditions", RECOVERY_CONDITIONS)

    cases = []
    for problem in problems:
        env = make_env(problem.environment)
        case = make_recovery_case(problem, env)
        if case is not None:
            cases.append(case)

    tasks = [(case, condition) for case in cases for condition in conditions]
    out = Path(args.output or _default_output_path(cfg, model_label))
    out.parent.mkdir(parents=True, exist_ok=True)

    records: list[RecoveryRecord] = []
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=_CONSOLE,
    )
    with out.open("w") as f, progress:
        progress_task = progress.add_task(f"Derail {model_label}", total=len(tasks))
        if worker_count == 1:
            for case, condition in tasks:
                record = _run_one_safe(case, condition, client, agent_cfg)
                _write_record(f, record)
                records.append(record)
                progress.advance(progress_task)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(_run_one_safe, case, condition, client, agent_cfg)
                    for case, condition in tasks
                ]
                for future in as_completed(futures):
                    record = future.result()
                    _write_record(f, record)
                    records.append(record)
                    progress.advance(progress_task)

    summary_path = Path(args.summary or _default_summary_path(out))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_summarize(records), indent=2, sort_keys=True))

    _CONSOLE.print(f"[green]Wrote {len(records)} recovery records to {out}[/green]")
    _CONSOLE.print(f"[green]Wrote recovery summary to {summary_path}[/green]")
    return 0


def _run_one_safe(case, condition, client, agent_cfg: AgentConfig) -> RecoveryRecord:
    try:
        env = make_env(case.problem.environment)
        return run_recovery_condition(case, env, client, condition, agent_cfg)
    except Exception as exc:  # noqa: BLE001
        return RecoveryRecord(
            problem_id=case.problem.problem_id,
            environment=case.problem.environment,
            complexity=case.problem.complexity,
            model=getattr(client, "model", "unknown"),
            condition=condition,
            success=False,
            failure_type=FailureType.API_ERROR,
            injection_step=case.injection_step,
            injection_type=case.injection_type,
            last_valid_state=case.last_valid_state,
            injected_action=case.injected_action,
            injected_state=case.injected_state,
            raw_model_output=traceback.format_exc(),
            stopped_reason="runner_exception",
            metadata={
                "exception": "".join(traceback.format_exception_only(type(exc), exc)).strip()
            },
        )


def _write_record(file, record: RecoveryRecord) -> None:
    file.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
    file.flush()


def _summarize(records: list[RecoveryRecord]) -> dict:
    groups: dict[tuple[str, str, int], list[RecoveryRecord]] = defaultdict(list)
    for record in records:
        groups[(record.environment, record.condition, record.complexity)].append(record)
    by_group = []
    for (env, condition, complexity), rows in sorted(groups.items()):
        successes = sum(row.success for row in rows)
        success_records = [row for row in rows if row.success]
        tokens_per_recovered_success_mean = (
            sum(row.tokens_in + row.tokens_out for row in success_records) / len(success_records)
            if success_records
            else None
        )
        tokens_per_success_amortised = (
            (sum(row.tokens_in + row.tokens_out for row in rows) / successes)
            if successes
            else None
        )
        by_group.append(
            {
                "environment": env,
                "condition": condition,
                "complexity": complexity,
                "n": len(rows),
                "n_successes": successes,
                "success_rate": successes / len(rows),
                "tokens_per_recovered_success_mean": tokens_per_recovered_success_mean,
                "tokens_per_success_amortised": tokens_per_success_amortised,
                "tokens_per_success": tokens_per_recovered_success_mean,
            }
        )
    condition_counts = Counter(record.condition for record in records)
    condition_success = Counter(record.condition for record in records if record.success)
    condition_tokens_success: dict[str, list[int]] = defaultdict(list)
    for record in records:
        if record.success:
            condition_tokens_success[record.condition].append(record.tokens_in + record.tokens_out)
    conditions_summary = {}
    for condition in sorted(condition_counts):
        success_tokens = condition_tokens_success.get(condition, [])
        conditions_summary[condition] = {
            "n": condition_counts[condition],
            "success": condition_success[condition],
            "success_rate": condition_success[condition] / condition_counts[condition],
            "tokens_per_recovered_success_mean": (
                sum(success_tokens) / len(success_tokens) if success_tokens else None
            ),
        }
    return {
        "groups": by_group,
        "conditions": conditions_summary,
        "failure_counts": dict(
            Counter(
                record.failure_type.value for record in records if record.failure_type is not None
            )
        ),
    }


def _default_output_path(cfg: dict, model_label: str) -> str:
    configured = cfg.get("run", {}).get("output", "data/derail/derail.jsonl")
    path = Path(configured)
    return str(path.with_name(f"{model_label}_{path.name}"))


def _default_summary_path(output: Path) -> str:
    return str(output.with_suffix(".summary.json"))
