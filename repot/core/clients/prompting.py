from __future__ import annotations

import json
from typing import Any


def messages_with_metadata(
    messages: list[dict[str, str]],
    metadata: dict[str, Any],
) -> list[dict[str, str]]:
    """Append legal-action / blocked-id / strategy-hint scaffolding to the last user message."""
    enriched = [dict(message) for message in messages]
    additions: list[str] = []
    action_options = metadata.get("action_options")
    if action_options:
        additions.append(
            "action_options_json: "
            + json.dumps(action_options, sort_keys=True, separators=(",", ":"))
        )
        additions.append(
            "Choose exactly one action id from action_options_json and return it as action_id. Do not invent ids."
        )
        if metadata.get("rank_actions"):
            additions.append("Also return ranked_action_ids as a list of action ids from best to worst.")
        blocked = metadata.get("blocked_action_ids")
        if blocked:
            additions.append("Do not choose blocked_action_ids: " + json.dumps(blocked, separators=(",", ":")))
        hint = metadata.get("strategy_hint")
        if hint:
            additions.append("strategy_hint: " + str(hint))
    legal_actions = metadata.get("legal_actions")
    if legal_actions and not action_options:
        additions.append(
            "legal_actions_json: "
            + json.dumps(legal_actions, sort_keys=True, separators=(",", ":"))
        )
        additions.append(
            "Choose one action exactly from legal_actions_json. Do not invent a different action."
        )
    if additions:
        enriched[-1]["content"] = enriched[-1]["content"] + "\n" + "\n".join(additions)
    return enriched
