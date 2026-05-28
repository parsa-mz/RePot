from __future__ import annotations

import os
import time
from dataclasses import dataclass

from repot.core.llm import GenerateRequest, GenerateResult
from repot.core.clients.prompting import messages_with_metadata


def _expand_env(value: str) -> str:
    if value.startswith("${") and value.endswith("}"):
        expr = value[2:-1]
        if ":-" in expr:
            name, default = expr.split(":-", 1)
            return os.environ.get(name, default)
        return os.environ.get(expr, "")
    return os.path.expandvars(value)


@dataclass
class OpenAICompatClient:
    """ModelClient for OpenAI-compatible OSS endpoints (vLLM, SGLang, etc.) via Chat Completions."""
    model: str
    base_url: str = "http://localhost:9011/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 16384

    def __post_init__(self) -> None:
        self.base_url = _expand_env(self.base_url)
        self.api_key = _expand_env(self.api_key)

    def generate(self, request: GenerateRequest) -> GenerateResult:
        """Send `request` to the OpenAI-compatible chat endpoint and return a `GenerateResult`."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install API support with: uv sync --extra api") from exc

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        t0 = time.perf_counter()
        messages = messages_with_metadata(request.messages, request.metadata)
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = "/no_think\n" + messages[0]["content"]
        # Disable extended thinking explicitly via the Qwen-style chat_template
        # kwarg. vLLM ignores chat_template_kwargs that the template doesn't
        # reference, so this is safe for non-Qwen models as well. Belt-and-
        # suspenders with the `/no_think` system-prompt prefix above.
        response = client.chat.completions.create(
            model=request.model or self.model,
            messages=messages,
            temperature=request.temperature if request.temperature is not None else self.temperature,
            max_tokens=request.max_tokens or self.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        latency_s = time.perf_counter() - t0
        choice = response.choices[0]
        usage = response.usage
        return GenerateResult(
            text=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_s=latency_s,
            finish_reason=getattr(choice, "finish_reason", None),
            raw_response=response.model_dump() if hasattr(response, "model_dump") else {},
        )
