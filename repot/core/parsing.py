from __future__ import annotations

import ast
import json
import re
from typing import Any


_MOVES_RE = re.compile(r"\bmoves\s*=\s*(\[.*?\])\s*(?:\n|$)", re.DOTALL)


def parse_moves_from_stdout(stdout: str) -> list:
    """Extract the last ``moves = [...]`` literal from sandbox stdout.

    Handles both JSON and Python repr forms (single or double quotes).
    Returns an empty list if no moves block is found or the literal cannot
    be parsed as a list.
    """
    matches = list(_MOVES_RE.finditer(stdout))
    if not matches:
        return []
    raw = matches[-1].group(1)
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(raw)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, list):
            return value
    return []


def parse_moves_with_fallback(stdout: str, code: str) -> list:
    """Like ``parse_moves_from_stdout`` but falls back to scanning the
    model's emitted code when stdout is empty.

    Some models (notably Gemini Flash 3.5) emit ``moves = [...]`` directly
    in their response without wrapping it in ``print(...)``. The sandbox
    runs that as a Python assignment (no error, no stdout) and the
    stdout-only parser returns []. The fallback rescues these by parsing
    the same regex against the code text itself.
    """
    actions = parse_moves_from_stdout(stdout)
    if actions:
        return actions
    return parse_moves_from_stdout(code or "")


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first valid JSON object out of `text`, tolerating fences and surrounding prose."""
    text = _clean_model_text(text)
    if not text:
        raise ValueError("empty model response")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        candidates = _json_object_candidates(text)
        if not candidates:
            raise ValueError("no JSON object found in model response") from None
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                value = json.loads(candidate)
                break
            except json.JSONDecodeError as exc:
                last_error = exc
        else:
            raise ValueError(str(last_error or "invalid JSON object")) from None
    if not isinstance(value, dict):
        raise ValueError("model response JSON is not an object")
    return value


def extract_actions(text: str) -> list[Any]:
    """Return the raw items of the model's action array.

    Items may be dicts or sequences (e.g. Apple-array form ["A", from, to]).
    Per-env normalisation happens at the caller via env.normalize_candidate_move,
    so filtering by shape here would silently drop list-form responses.

    Accepts several shapes (Gemini varies more than OpenAI):
      * ``{"actions": [...]}``  (canonical)
      * ``{"action": ...}``     (single-action)
      * ``{"moves": [...]}``    (Gemini variant)
      * top-level array ``[...]`` (Gemini sometimes drops the wrapper)
      * raw text containing ``moves = [...]`` (fallback, also used for
        Gemini's JSON-string-wrapped responses)
    """
    # Top-level JSON array — accept it directly without going through the
    # object extractor (which would reject it as "not an object").
    stripped = _clean_model_text(text)
    if stripped.startswith("["):
        try:
            decoded = json.loads(stripped)
        except (ValueError, TypeError):
            decoded = None
        if isinstance(decoded, list):
            return list(decoded)

    try:
        obj = extract_json_object(text)
    except ValueError:
        fallback = _fallback_action_list(text)
        if fallback is not None:
            return fallback
        raise
    if "actions" in obj and isinstance(obj["actions"], list):
        return list(obj["actions"])
    if "action" in obj and isinstance(obj["action"], (dict, list, tuple)):
        return [obj["action"]]
    # Gemini sometimes returns `{"moves": [...]}` instead of `{"actions": [...]}`.
    # Accept it as a synonym.
    if "moves" in obj and isinstance(obj["moves"], list):
        return list(obj["moves"])
    fallback = _fallback_action_list(text)
    if fallback is not None:
        return fallback
    raise ValueError("response must contain action or actions")


def _fallback_action_list(text: str) -> list[Any] | None:
    """Recover an action list from non-object JSON responses.

    Handles two cases:
      1. The whole response is a JSON string like ``"moves = [...]"`` — we
         unquote it and re-parse.
      2. The response (or its decoded string form) contains a bare
         ``moves = [...]`` literal anywhere in the text.
    """
    stripped = text.strip()
    if not stripped:
        return None
    try:
        decoded = json.loads(stripped)
    except (ValueError, TypeError):
        decoded = None
    if isinstance(decoded, str):
        actions = parse_moves_from_stdout(decoded)
        if actions:
            return actions
    actions = parse_moves_from_stdout(stripped)
    if actions:
        return actions
    return None


def _clean_model_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    return text


def _json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    stack = 0
    start: int | None = None
    in_string = False
    escape = False
    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if stack == 0:
                start = idx
            stack += 1
        elif char == "}":
            if stack:
                stack -= 1
                if stack == 0 and start is not None:
                    candidates.append(text[start : idx + 1])
                    start = None
    return candidates
