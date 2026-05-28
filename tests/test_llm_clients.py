"""Unit tests for the four real ModelClient implementations.

All HTTP/SDK calls are mocked at the transport boundary so these tests are
fully offline and deterministic. They guard the request shape (right model,
messages, max_tokens) and response parsing (text + token counts).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from repot.core.clients.anthropic_client import AnthropicMessagesClient
from repot.core.clients.gemini_client import GoogleGenAIClient
from repot.core.clients.openai_client import OpenAIResponsesClient
from repot.core.clients.oss_client import OpenAICompatClient
from repot.core.llm import GenerateRequest

_OPENAI_SDK_PATH = "openai.OpenAI"
_ANTHROPIC_SDK_PATH = "anthropic.Anthropic"
_GEMINI_SDK_PATH = "google.genai"
_OSS_SDK_PATH = "openai.OpenAI"


# ---------------------------------------------------------------------------
# OpenAIResponsesClient
# ---------------------------------------------------------------------------


def test_openai_responses_client_shapes_request_and_parses_response(monkeypatch):
    """Mock openai.OpenAI; verify Responses-API call args and result parsing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    fake_response = SimpleNamespace(
        output_text="hello world",
        status="completed",
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=5,
            input_tokens_details=SimpleNamespace(cached_tokens=3),
        ),
        model_dump=lambda: {"id": "resp_abc"},
    )
    fake_client = MagicMock()
    fake_client.responses.create.return_value = fake_response

    with patch(_OPENAI_SDK_PATH, return_value=fake_client) as openai_ctor:
        client = OpenAIResponsesClient(model="gpt-5-mini", max_tokens=2048)
        req = GenerateRequest(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=128,
        )
        result = client.generate(req)

    openai_ctor.assert_called_once()
    fake_client.responses.create.assert_called_once()
    call_kwargs = fake_client.responses.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-5-mini"
    assert call_kwargs["input"][0]["role"] == "user"
    assert call_kwargs["max_output_tokens"] == 128
    assert result.text == "hello world"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 5
    assert result.finish_reason == "completed"


# ---------------------------------------------------------------------------
# AnthropicMessagesClient
# ---------------------------------------------------------------------------


def test_anthropic_messages_client_shapes_request_and_parses_response(monkeypatch):
    """Mock anthropic.Anthropic; verify Messages-API call args and parsing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    fake_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="claude reply")],
        usage=SimpleNamespace(
            input_tokens=8,
            output_tokens=4,
            cache_read_input_tokens=0,
        ),
        stop_reason="end_turn",
        model_dump=lambda: {"id": "msg_abc"},
    )
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    with patch(_ANTHROPIC_SDK_PATH, return_value=fake_client) as anthropic_ctor:
        client = AnthropicMessagesClient(model="claude-sonnet-4-6", max_tokens=1024)
        req = GenerateRequest(
            messages=[
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
            max_tokens=256,
        )
        result = client.generate(req)

    anthropic_ctor.assert_called_once_with(api_key="test-key")
    fake_client.messages.create.assert_called_once()
    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["max_tokens"] == 256
    # System is split out separately from messages in Anthropic SDK.
    assert call_kwargs.get("system") == "be terse"
    assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert result.text == "claude reply"
    assert result.prompt_tokens == 8
    assert result.completion_tokens == 4
    assert result.finish_reason == "end_turn"


# ---------------------------------------------------------------------------
# GoogleGenAIClient
# ---------------------------------------------------------------------------


def test_google_genai_client_shapes_request_and_parses_response(monkeypatch):
    """Mock google.genai.Client; verify generate_content args and parsing."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

    fake_response = SimpleNamespace(
        text="gemini reply",
        usage_metadata=SimpleNamespace(
            prompt_token_count=15,
            candidates_token_count=7,
            cached_content_token_count=0,
        ),
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
        model_dump=lambda: {"id": "gen_abc"},
    )
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response

    # The client calls `from google import genai; genai.Client(...)`. Patch the
    # whole submodule's Client constructor.
    fake_genai = MagicMock()
    fake_genai.Client.return_value = fake_client
    fake_types = MagicMock()
    fake_types.HttpOptions = MagicMock()

    with patch.dict(
        "sys.modules",
        {"google.genai": fake_genai, "google.genai.types": fake_types},
    ):
        client = GoogleGenAIClient(model="gemini-2.5-flash", project="test-project")
        req = GenerateRequest(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=512,
        )
        result = client.generate(req)

    fake_client.models.generate_content.assert_called_once()
    call_kwargs = fake_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    # Gemini takes `contents`, not `messages` — confirm conversion happened.
    assert isinstance(call_kwargs["contents"], list)
    assert call_kwargs["contents"][0]["role"] == "user"
    config = call_kwargs["config"]
    assert config["max_output_tokens"] == 512
    assert result.text == "gemini reply"
    assert result.prompt_tokens == 15
    assert result.completion_tokens == 7
    assert result.finish_reason == "STOP"


# ---------------------------------------------------------------------------
# OpenAICompatClient (vLLM / OSS)
# ---------------------------------------------------------------------------


def test_openai_compat_client_shapes_request_and_parses_response(monkeypatch):
    """Mock openai.OpenAI; verify chat.completions args for an OSS endpoint."""
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="oss reply"),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=6),
        model_dump=lambda: {"id": "cmpl_abc"},
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch(_OSS_SDK_PATH, return_value=fake_client) as openai_ctor:
        client = OpenAICompatClient(
            model="Qwen/Qwen3-8B",
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            max_tokens=4096,
        )
        req = GenerateRequest(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=256,
        )
        result = client.generate(req)

    openai_ctor.assert_called_once()
    ctor_kwargs = openai_ctor.call_args.kwargs
    assert ctor_kwargs["base_url"] == "http://localhost:8000/v1"
    fake_client.chat.completions.create.assert_called_once()
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "Qwen/Qwen3-8B"
    assert call_kwargs["max_tokens"] == 256
    assert call_kwargs["messages"][-1] == {"role": "user", "content": "hi"}
    assert result.text == "oss reply"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 6
    assert result.finish_reason == "stop"


@pytest.mark.parametrize("cls", [
    OpenAIResponsesClient,
    AnthropicMessagesClient,
    GoogleGenAIClient,
    OpenAICompatClient,
])
def test_client_exposes_model_attribute(cls):
    """All four clients must expose a public ``model`` attribute (ModelClient protocol)."""
    if cls is GoogleGenAIClient:
        c = cls(model="gemini-2.5-flash", project="x")
    elif cls is OpenAICompatClient:
        c = cls(model="m", base_url="http://x", api_key="k")
    else:
        c = cls(model="m")
    assert c.model
