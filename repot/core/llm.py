from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    """One LLM request: chat-style messages, sampling config, optional metadata for tracing."""

    model_config = ConfigDict(extra="forbid")

    messages: list[dict[str, str]]
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 16384
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerateResult(BaseModel):
    """One LLM response: extracted text, token counts, latency, finish reason, raw payload."""

    model_config = ConfigDict(extra="allow")

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)


class ModelClient(Protocol):
    """Provider-agnostic LLM interface implemented by every client in ``repot.core.clients``."""

    model: str

    def generate(self, request: GenerateRequest) -> GenerateResult:
        """Send `request` to the provider and return a `GenerateResult`."""
        ...


def create_model_client(config: dict[str, Any]) -> ModelClient:
    """Construct a ``ModelClient`` from a YAML provider config (provider + model + params)."""
    provider = config.get("provider", "local_dummy")
    if provider == "local_dummy":
        from repot.core.clients.local_dummy import LocalDummyClient

        return LocalDummyClient(model=config.get("model", "local_dummy"), mode=config.get("mode", "oracle"))
    if provider == "openai_compat":
        from repot.core.clients.oss_client import OpenAICompatClient

        return OpenAICompatClient(
            model=config["model"],
            base_url=config.get("base_url", "http://localhost:8000/v1"),
            api_key=config.get("api_key", "EMPTY"),
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 16384)),
        )
    if provider == "openai_responses":
        from repot.core.clients.openai_client import OpenAIResponsesClient

        return OpenAIResponsesClient(
            model=config["model"],
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 16384)),
            reasoning_effort=str(config.get("reasoning_effort", "none")),
        )
    if provider == "google_genai":
        from repot.core.clients.gemini_client import GoogleGenAIClient

        return GoogleGenAIClient(
            model=config["model"],
            project=str(config.get("project", "")),
            location=str(config.get("location", "global")),
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 16384)),
            thinking_level=str(config.get("thinking_level", "MEDIUM")),
        )
    if provider == "anthropic_messages":
        from repot.core.clients.anthropic_client import AnthropicMessagesClient

        thinking_budget_raw = config.get("thinking_budget")
        thinking_budget: int | None = None
        if thinking_budget_raw is not None:
            try:
                thinking_budget = int(thinking_budget_raw)
            except (TypeError, ValueError):
                thinking_budget = None
        return AnthropicMessagesClient(
            model=config["model"],
            api_key=config.get("api_key"),
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 16384)),
            thinking_budget=thinking_budget,
        )
    raise ValueError(f"Unknown model provider: {provider}")
