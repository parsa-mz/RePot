from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(value: Any, length: int = 12) -> str:
    """Return a deterministic hex sha256 prefix of length `length` for any JSON-serializable value."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
