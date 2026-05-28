from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from repot.core.llm import GenerateRequest, GenerateResult


@dataclass
class LocalDummyClient:
    """Offline test client that emits oracle-derived responses for the harness without any network call."""
    model: str = "local_dummy"
    mode: str = "oracle"
    _invalid_once_used: set[str] = field(default_factory=set)

    def generate(self, request: GenerateRequest) -> GenerateResult:
        """Synthesize a deterministic response from `request.metadata` (oracle/format_error/loop/etc.)."""
        t0 = time.perf_counter()
        metadata = request.metadata
        key = f"{metadata.get('problem_id')}:{metadata.get('method', '')}"

        if self.mode == "format_error":
            text = "not json"
        elif self.mode == "loop":
            options = metadata.get("action_options") or []
            if options:
                text = json.dumps({"action_id": options[0]["id"], "rationale": "repeat first legal action"})
            else:
                action = metadata.get("legal_actions", [{}])[0] if metadata.get("legal_actions") else {}
                text = json.dumps({"thought": "repeat first legal action", "action": action})
        elif self.mode == "invalid_once" and key not in self._invalid_once_used:
            self._invalid_once_used.add(key)
            if metadata.get("action_options"):
                text = json.dumps({"action_id": "bad_id", "rationale": "intentional bad id"})
            else:
                text = json.dumps({"thought": "intentional bad move", "action": {"type": "invalid"}})
        elif metadata.get("program_of_thought"):
            actions = metadata.get("oracle_actions", [])
            text = "moves = " + repr(actions) + "\nprint('moves = ' + repr(moves))"
        elif metadata.get("chunked_actions"):
            actions = metadata.get("oracle_actions", [])
            idx = int(metadata.get("oracle_index", 0))
            chunk_size = int(metadata.get("chunk_size", 4))
            text = json.dumps({"actions": actions[idx : idx + chunk_size], "rationale": "oracle action chunk"})
        elif metadata.get("action_options"):
            options = metadata["action_options"]
            actions = metadata.get("oracle_actions", [])
            idx = int(metadata.get("oracle_index", 0))
            option = _oracle_option(options, actions, idx)
            if metadata.get("rank_actions"):
                ranked = [option["id"]] + [item["id"] for item in options if item["id"] != option["id"]]
                text = json.dumps({"action_id": option["id"], "ranked_action_ids": ranked, "rationale": "oracle ranked action"})
            else:
                text = json.dumps({"action_id": option["id"], "rationale": "oracle action id"})
        elif "oracle_actions" in metadata:
            actions = metadata["oracle_actions"]
            if metadata.get("one_action"):
                idx = int(metadata.get("oracle_index", 0))
                action = actions[idx] if idx < len(actions) else {}
                text = json.dumps({"thought": "oracle next action", "action": action, "predicted_next_state": metadata.get("predicted_next_state")})
            else:
                text = json.dumps({"thought": "oracle full solution", "actions": actions})
        else:
            action = metadata.get("legal_actions", [{}])[0] if metadata.get("legal_actions") else {}
            text = json.dumps({"thought": "first legal action", "action": action})

        approx_in = sum(len(m.get("content", "")) for m in request.messages) // 4
        approx_out = len(text) // 4
        return GenerateResult(
            text=text,
            prompt_tokens=approx_in,
            completion_tokens=approx_out,
            latency_s=time.perf_counter() - t0,
        )


def _oracle_option(options: list[dict], oracle_actions: list[dict], oracle_index: int) -> dict:
    if not options:
        return {"id": "", "action": {}}
    if oracle_index < len(oracle_actions):
        oracle_action = oracle_actions[oracle_index]
        for option in options:
            if option.get("action") == oracle_action:
                return option
    return options[0]
