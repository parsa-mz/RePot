"""Shared helpers for the repot CLI subcommands."""
from __future__ import annotations

import random
from pathlib import Path

from repot.core.schemas import Problem
from repot.core.agent import AgentConfig
from repot.core.utils.config import load_yaml


def build_agent_config(run_cfg: dict, model_cfg: dict) -> AgentConfig:
    """Translate a config-file ``run`` block into an ``AgentConfig`` instance."""
    chunk_size_by_env_raw = run_cfg.get("chunk_size_by_env")
    chunk_size_by_env = (
        {str(name): int(size) for name, size in chunk_size_by_env_raw.items()}
        if isinstance(chunk_size_by_env_raw, dict) and chunk_size_by_env_raw
        else None
    )
    chunk_max_tokens_by_env_raw = run_cfg.get("chunk_max_tokens_by_env")
    chunk_max_tokens_by_env = (
        {str(name): int(size) for name, size in chunk_max_tokens_by_env_raw.items()}
        if isinstance(chunk_max_tokens_by_env_raw, dict) and chunk_max_tokens_by_env_raw
        else None
    )
    proposer_by_env_raw = run_cfg.get("vex_proposer_by_env")
    vex_proposer_by_env = (
        {str(name): str(kind) for name, kind in proposer_by_env_raw.items()}
        if isinstance(proposer_by_env_raw, dict) and proposer_by_env_raw
        else None
    )
    return AgentConfig(
        max_steps_multiplier=int(run_cfg.get("max_steps_multiplier", 2)),
        max_retries=int(run_cfg.get("max_retries", 3)),
        self_consistency_k=int(run_cfg.get("self_consistency_k", 8)),
        max_state_visits=int(run_cfg.get("max_state_visits", 4)),
        chunk_size=int(run_cfg.get("chunk_size", 4)),
        chunk_size_by_env=chunk_size_by_env,
        max_llm_calls=int(run_cfg.get("max_llm_calls", 4)),
        max_repair_calls=int(run_cfg.get("max_repair_calls", 2)),
        chunk_max_tokens=int(run_cfg.get("chunk_max_tokens", 2048)),
        chunk_max_tokens_by_env=chunk_max_tokens_by_env,
        action_choice_max_tokens=int(run_cfg.get("action_choice_max_tokens", 256)),
        temperature=float(model_cfg.get("temperature", 0.0)),
        max_tokens=int(model_cfg.get("max_tokens", 16384)),
        vex_default_proposer=str(run_cfg.get("vex_default_proposer", "json")),
        vex_proposer_by_env=vex_proposer_by_env,
    )


def load_model_config(cfg: dict, model_override: str | None) -> tuple[dict, str]:
    """Resolve the model config from the experiment YAML + optional override."""
    models_path = cfg.get("models", {}).get("config", "configs/models.yaml")
    models = load_yaml(models_path)
    active = cfg.get("models", {}).get("active", "local_dummy")
    model_cfg = dict(models.get(active, models.get("local_dummy", {"provider": "local_dummy"})))
    label = active
    if model_override:
        if model_override in models:
            model_cfg = dict(models[model_override])
            label = model_override
        elif model_override == "local_dummy":
            model_cfg = dict(
                models.get("local_dummy", {"provider": "local_dummy", "model": "local_dummy"})
            )
            label = "local_dummy"
        else:
            model_cfg = dict(models.get("default", {"provider": "openai_compat"}))
            model_cfg.update({"provider": "openai_compat", "model": model_override})
            label = _slug(model_override)
    return model_cfg, label


def resolve_workers(workers: int | None, cfg: dict, model_cfg: dict) -> int:
    """Pick worker count from CLI > config > provider default."""
    if workers is not None:
        return max(1, int(workers))
    configured = cfg.get("run", {}).get("workers")
    if configured is not None:
        return max(1, int(configured))
    provider = model_cfg.get("provider")
    if provider in {"openai_responses", "anthropic_messages", "google_genai"}:
        return 10
    return 1


def sample_problems(
    problems: list[Problem],
    max_items: int | None,
    mode: str,
    seed: int,
) -> list[Problem]:
    """Sample / cap problems. Modes: ordered | shuffle | stratified."""
    if max_items is None:
        return problems
    if mode == "ordered":
        return problems[:max_items]
    rng = random.Random(seed)
    if mode == "shuffle":
        sampled = list(problems)
        rng.shuffle(sampled)
        return sampled[:max_items]
    if mode == "stratified":
        groups: dict[tuple[str, int], list[Problem]] = {}
        for problem in problems:
            groups.setdefault((problem.environment, problem.complexity), []).append(problem)
        for rows in groups.values():
            rng.shuffle(rows)
        selected: list[Problem] = []
        while len(selected) < max_items and any(groups.values()):
            for key in sorted(groups):
                if groups[key]:
                    selected.append(groups[key].pop())
                    if len(selected) >= max_items:
                        break
        return selected
    raise ValueError(f"sample must be one of: ordered, shuffle, stratified (got {mode!r})")


def filter_problems_by_config(problems: list[Problem], cfg: dict) -> list[Problem]:
    """Apply environment+complexity filter from the experiment YAML."""
    env_cfg = cfg.get("dataset", {}).get("environments")
    if not isinstance(env_cfg, dict) or not env_cfg:
        return problems
    allowed: dict[str, set[int] | None] = {}
    for env_name, spec in env_cfg.items():
        if isinstance(spec, dict) and "complexities" in spec:
            allowed[env_name] = {int(item) for item in spec["complexities"]}
        else:
            allowed[env_name] = None
    return [
        problem
        for problem in problems
        if problem.environment in allowed
        and (allowed[problem.environment] is None or problem.complexity in allowed[problem.environment])
    ]


def default_trace_path(cfg: dict, model_label: str) -> Path:
    """Compose a model-prefixed trace JSONL path."""
    configured = cfg.get("run", {}).get("traces_path", "data/traces/repot_main.jsonl")
    path = Path(configured)
    suffix = path.suffix or ".jsonl"
    stem = path.stem
    if stem.endswith("_dummy"):
        stem = stem[: -len("_dummy")]
    return path.with_name(f"{model_label}_{stem}{suffix}")


def _slug(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace("-", "_")
        .lower()
    )
