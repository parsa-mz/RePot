from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML from `path` and expand `${VAR}` / `${VAR:-default}` env-var references in strings."""
    with Path(path).open() as f:
        return _expand(yaml.safe_load(f) or {})


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            """Substitute `${VAR}` / `${VAR:-default}` from the environment."""
            name, default = match.group(1), match.group(2) or ""
            return os.environ.get(name, default)

        return _ENV_PATTERN.sub(repl, value)
    return value
