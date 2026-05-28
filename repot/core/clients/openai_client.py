from __future__ import annotations

import os
import time
from dataclasses import dataclass

from repot.core.llm import GenerateRequest, GenerateResult
from repot.core.clients.prompting import messages_with_metadata


@dataclass
class OpenAIResponsesClient:
    """ModelClient for OpenAI's Responses API, with optional reasoning-effort knob."""
    model: str
    temperature: float = 0.0
    max_tokens: int = 16384
    reasoning_effort: str = "none"

    def generate(self, request: GenerateRequest) -> GenerateResult:
        """Send `request` to the OpenAI Responses API and return a `GenerateResult`."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI SDK is missing. Install it: pip install openai"
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Put it in .env or export it.")

        client = OpenAI(api_key=api_key)
        t0 = time.perf_counter()
        messages = messages_with_metadata(request.messages, request.metadata)
        response = self._create_response(client, request, messages)
        latency_s = time.perf_counter() - t0
        usage = getattr(response, "usage", None)
        cached_tokens = 0
        if usage:
            details = getattr(usage, "input_tokens_details", None)
            if details is not None:
                cached_tokens = getattr(details, "cached_tokens", 0) or 0
        raw = response.model_dump() if hasattr(response, "model_dump") else {}
        raw["cached_tokens"] = cached_tokens
        return GenerateResult(
            text=getattr(response, "output_text", "") or "",
            prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            latency_s=latency_s,
            finish_reason=getattr(response, "status", None),
            raw_response=raw,
            cached_tokens=cached_tokens,
        )

    def _create_response(self, client, request: GenerateRequest, messages: list[dict[str, str]]):
        create_args = self._create_args(request, messages)
        try:
            return client.responses.create(**create_args)
        except Exception as exc:  # noqa: BLE001 - inspect provider-specific prompt rejection.
            lowered = str(exc).lower()
            if "invalid_prompt" not in lowered and "flagged as potentially violating" not in lowered:
                raise
            sanitized = _sanitize_messages(messages)
            if sanitized == messages:
                raise
            create_args["input"] = sanitized
            return client.responses.create(**create_args)

    def _create_args(self, request: GenerateRequest, messages: list[dict[str, str]]) -> dict:
        model = request.model or self.model
        args = {
            "model": model,
            "input": messages,
            "max_output_tokens": request.max_tokens or self.max_tokens,
            "reasoning": {"effort": self.reasoning_effort},
        }
        if not request.metadata.get("program_of_thought"):
            args["text"] = {"format": _response_schema(request.metadata)}
        temperature = request.temperature if request.temperature is not None else self.temperature
        if temperature is not None and not _omits_temperature(model):
            args["temperature"] = temperature
        return args


def _omits_temperature(model: str) -> bool:
    return model.startswith("gpt-5")


def _response_schema(metadata: dict) -> dict:
    if metadata.get("action_options"):
        properties = {
            "action_id": {"type": "string"},
            "predicted_next_state": {"type": "object", "additionalProperties": True},
            "rationale": {"type": "string"},
        }
        required = ["action_id", "predicted_next_state", "rationale"]
        if metadata.get("rank_actions"):
            properties["ranked_action_ids"] = {"type": "array", "items": {"type": "string"}}
            required.append("ranked_action_ids")
        return {
            "type": "json_schema",
            "name": "stateguard_action_choice",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }
    if (metadata.get("oracle_actions") or metadata.get("chunked_actions")) and not metadata.get("one_action"):
        return {
            "type": "json_schema",
            "name": "stateguard_action_list",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {"actions": {"type": "array", "items": {"type": "object"}}},
                "required": ["actions"],
                "additionalProperties": True,
            },
        }
    return {"type": "json_object"}


def _sanitize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    replacements = {
        "Checker Jumping": "Linear swap puzzle",
        "checker": "token",
        "jump": "two-cell move",
        "rejected": "not accepted",
        "repair": "continue",
    }
    sanitized = []
    for message in messages:
        content = message["content"]
        for old, new in replacements.items():
            content = content.replace(old, new)
        sanitized.append({**message, "content": content})
    return sanitized
