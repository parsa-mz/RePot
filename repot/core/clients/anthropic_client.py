"""Anthropic Claude client using the public Anthropic Python SDK.

Mirrors the shape of OpenAIResponsesClient and GoogleGenAIClient so the rest
of the harness can switch providers via configs/models.yaml only.

Auth: reads ``ANTHROPIC_API_KEY`` from the environment (or accepts an
``api_key`` field in the model config).

Models: pass the public Anthropic model id, e.g. ``claude-sonnet-4-6``,
``claude-opus-4-7``, ``claude-haiku-4-5-20251001``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from repot.core.llm import GenerateRequest, GenerateResult
from repot.core.clients.prompting import messages_with_metadata


@dataclass
class AnthropicMessagesClient:
    """ModelClient for Anthropic's Messages API; supports extended-thinking budget."""
    model: str
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 16384
    thinking_budget: int | None = None  # extended-thinking budget tokens; None = off

    def generate(self, request: GenerateRequest) -> GenerateResult:
        """Send `request` to Anthropic Messages and return a `GenerateResult`."""
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic SDK not installed. Install with:\n"
                "    pip install anthropic"
            ) from exc

        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it in your environment "
                "or pass api_key in configs/models.yaml."
            )

        client = Anthropic(api_key=api_key)

        messages = messages_with_metadata(request.messages, request.metadata)
        system_text, user_messages = _split_system(messages)

        kwargs: dict = {
            "model": request.model or self.model,
            "messages": user_messages,
            "max_tokens": int(request.max_tokens or self.max_tokens),
        }
        if system_text:
            kwargs["system"] = system_text
        # Newer Claude models reject `temperature`. Drop it for them.
        temperature = request.temperature if request.temperature is not None else self.temperature
        if temperature is not None and _supports_temperature(kwargs["model"]):
            kwargs["temperature"] = float(temperature)
        if self.thinking_budget and self.thinking_budget > 0:
            # Anthropic extended-thinking; budget_tokens must be < max_tokens.
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": min(self.thinking_budget, int(kwargs["max_tokens"]) - 256),
            }

        t0 = time.perf_counter()
        response = client.messages.create(**kwargs)
        latency_s = time.perf_counter() - t0

        text = _extract_text(response)
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        cached_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0
        finish_reason = getattr(response, "stop_reason", None)
        raw = response.model_dump() if hasattr(response, "model_dump") else {}
        raw["cached_tokens"] = cached_tokens
        return GenerateResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency_s,
            finish_reason=finish_reason,
            raw_response=raw,
            cached_tokens=cached_tokens,
        )


def _split_system(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Anthropic takes ``system`` separately from the message list. Pull
    every system message out and concatenate, leaving user/assistant turns."""
    system_chunks: list[str] = []
    user_messages: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            if content:
                system_chunks.append(content)
            continue
        anth_role = "assistant" if role == "assistant" else "user"
        user_messages.append({"role": anth_role, "content": content})
    return "\n\n".join(system_chunks), user_messages


def _extract_text(response) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", "") or ""
            if text:
                parts.append(text)
    return "".join(parts)


def _supports_temperature(model: str) -> bool:
    """Newer Claude models reject the `temperature` field."""
    m = model.lower()
    deprecated_markers = ("opus-4-7", "opus-4-6", "sonnet-4-6")
    return not any(marker in m for marker in deprecated_markers)
