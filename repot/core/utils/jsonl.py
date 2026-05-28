from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load every non-empty line of `path` as a JSON object and return the list."""
    records: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any] | BaseModel]) -> None:
    """Write each record as one JSON line to `path`, creating parent dirs and dumping `BaseModel`s."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for record in records:
            if isinstance(record, BaseModel):
                data = record.model_dump(mode="json")
            else:
                data = record
            f.write(json.dumps(data, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, record: dict[str, Any] | BaseModel) -> None:
    """Append `record` as one JSON line to `path`, creating parent dirs."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
    with out.open("a") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")
