"""Canonical import for client implementations to subclass.

Re-exports the `ModelClient` protocol + request/response types from
`repot.core.llm` so concrete clients can write:

    from repot.core.clients.base import ModelClient, GenerateRequest, GenerateResult
"""

from repot.core.llm import GenerateRequest, GenerateResult, ModelClient

__all__ = ["ModelClient", "GenerateRequest", "GenerateResult"]
