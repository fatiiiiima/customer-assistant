"""STUB: optional cloud LLM provider for dev-only latency benchmarking.

PLACEHOLDER CODE. Not implemented, needs no API key, and is never reached by default.
When enabled later: set VOICE_AGENT_ALLOW_CLOUD=1, GEMINI_API_KEY and GEMINI_MODEL in a
gitignored .env, then check Google's current docs for the model name and SDK. Do NOT
hard-code a model name here. SYNTHETIC data only. STT and TTS stay local.
"""
from __future__ import annotations

from typing import Any

from .base import LlmResult


class CloudLlmStub:
    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "Cloud LLM provider is a disabled stub. Implement it against Google's current "
            "docs when you enable cloud benchmark mode."
        )

    async def generate_json(self, system: str, user: str, schema: dict[str, Any]) -> LlmResult:
        raise NotImplementedError

    async def generate_text(self, system: str, user: str, max_tokens: int) -> LlmResult:
        raise NotImplementedError
