"""Optional LLM fallback contract; no remote provider is called by default."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FallbackResult:
    labels: list[str]
    latency_ms: float
    token_usage: int | None = None


def needs_fallback(structure_confidence: float, enabled: bool) -> bool:
    return enabled and structure_confidence < 0.45
