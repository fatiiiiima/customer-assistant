"""LLM provider interface. Providers are swappable via config (llm.provider).

DRAFT: the method set may change after the Phase 0b spike and Phase 1 design.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class LlmResult:
    text: str                       # raw output (JSON string for constrained calls)
    parsed: dict[str, Any] | None   # parsed JSON when a schema was given
    ttft_ms: float | None           # time to first token
    total_ms: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None


class LlmProvider(Protocol):
    async def generate_json(self, system: str, user: str, schema: dict[str, Any]) -> LlmResult:
        """Schema-constrained output: used for intent + slots (NLU) and tool arguments."""
        ...

    async def generate_text(self, system: str, user: str, max_tokens: int) -> LlmResult:
        """Free-form reply; 1-2 short sentences, open questions only."""
        ...
