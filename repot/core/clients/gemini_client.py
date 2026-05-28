from __future__ import annotations

import os
import time
from dataclasses import dataclass

from repot.core.llm import GenerateRequest, GenerateResult
from repot.core.clients.prompting import messages_with_metadata


# Vertex AI Gemini client. Mirrors the shape of OpenAIResponsesClient so the
# rest of the harness can swap providers via configs/models.yaml only.
#
# Auth: relies on Application Default Credentials. Run `gcloud auth
# application-default login` once. Project + location come from the config,
# which reads ${GOOGLE_CLOUD_PROJECT}/${GOOGLE_CLOUD_LOCATION} env vars.


@dataclass
class GoogleGenAIClient:
    """ModelClient for Vertex AI Gemini, with optional `thinking_level` and JSON-schema enforcement."""
    model: str
    project: str = ""
    location: str = "global"
    temperature: float = 0.0
    max_tokens: int = 16384
    thinking_level: str = "MEDIUM"  # HIGH, MEDIUM, LOW, MINIMAL

    def generate(self, request: GenerateRequest) -> GenerateResult:
        """Send `request` to Vertex AI Gemini and return a `GenerateResult`."""
        try:
            from google import genai
            from google.genai.types import HttpOptions
        except ImportError as exc:
            raise RuntimeError(
                "Install google-genai with: .venv/bin/python -m pip install google-genai"
            ) from exc

        project = self.project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        location = self.location or os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        if not project:
            raise RuntimeError(
                "Gemini client needs a Vertex AI project. Set GOOGLE_CLOUD_PROJECT "
                "in your .env or pass `project:` in configs/models.yaml."
            )

        client = genai.Client(
            project=project,
            location=location,
            vertexai=True,
            http_options=HttpOptions(api_version="v1"),
        )
        messages = messages_with_metadata(request.messages, request.metadata)
        contents, system_instruction = _to_gemini_contents(messages)
        config = self._build_config(request, system_instruction)

        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=request.model or self.model,
            contents=contents,
            config=config,
        )
        latency_s = time.perf_counter() - t0

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        cached_tokens = getattr(usage, "cached_content_token_count", 0) if usage else 0
        finish_reason = _finish_reason(response)
        raw = response.model_dump() if hasattr(response, "model_dump") else {}
        raw["cached_tokens"] = cached_tokens
        return GenerateResult(
            text=getattr(response, "text", "") or "",
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            latency_s=latency_s,
            finish_reason=finish_reason,
            raw_response=raw,
            cached_tokens=int(cached_tokens or 0),
        )

    def _build_config(self, request: GenerateRequest, system_instruction: str | None) -> dict:
        cfg: dict = {
            "max_output_tokens": int(request.max_tokens or self.max_tokens),
        }
        temperature = request.temperature if request.temperature is not None else self.temperature
        if temperature is not None:
            cfg["temperature"] = float(temperature)
        if system_instruction:
            cfg["system_instruction"] = system_instruction
        if self.thinking_level and self.thinking_level.upper() != "NONE":
            cfg["thinking_config"] = {"thinking_level": self.thinking_level.upper()}
        # JSON schema: only enforced for non-PoT calls. Code-emitting prompts
        # need plain text. Mirrors OpenAIResponsesClient._create_args.
        if not request.metadata.get("program_of_thought"):
            cfg["response_mime_type"] = "application/json"
            schema = _response_schema(request.metadata)
            if schema is not None:
                cfg["response_schema"] = schema
        return cfg


def _to_gemini_contents(messages: list[dict[str, str]]) -> tuple[list[dict], str | None]:
    """Gemini takes `contents` (list of {role, parts}) plus a separate
    `system_instruction`. OpenAI-style messages put system content inline.
    Split system messages out and convert the rest to Gemini's role names.
    """
    system_chunks: list[str] = []
    contents: list[dict] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            system_chunks.append(content)
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": content}]})
    system_instruction = "\n\n".join(s for s in system_chunks if s) or None
    return contents, system_instruction


def _finish_reason(response) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return None
    return getattr(reason, "name", str(reason))


def _response_schema(metadata: dict) -> dict | None:
    """Schema for the JSON-output paths Gemini supports.

    Important Gemini quirk: unlike OpenAI's `strict: false` + open-content
    schemas, Vertex AI enforces `response_schema` *strictly*. Passing a
    permissive `{"type": "OBJECT"}` for action items yields minimum-conforming
    junk (e.g. `{"actions": [{}]}`). Therefore:

    - For `action_options` (single action with id + state + rationale) we
      keep a tight schema — fields are well-defined.
    - For `oracle_actions`/`chunked_actions` (free-form action lists where
      shape varies per env) we *omit* the schema and only set
      response_mime_type=application/json. The prompt already describes
      the action format clearly; the schema would over-constrain.

    PoT-style code emission is handled upstream by the
    ``program_of_thought`` metadata flag (no schema applied)."""
    if metadata.get("action_options"):
        properties = {
            "action_id": {"type": "STRING"},
            "predicted_next_state": {"type": "OBJECT"},
            "rationale": {"type": "STRING"},
        }
        required = ["action_id", "predicted_next_state", "rationale"]
        if metadata.get("rank_actions"):
            properties["ranked_action_ids"] = {"type": "ARRAY", "items": {"type": "STRING"}}
            required.append("ranked_action_ids")
        return {"type": "OBJECT", "properties": properties, "required": required}
    return None
