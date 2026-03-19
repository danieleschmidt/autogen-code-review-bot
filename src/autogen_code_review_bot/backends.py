"""LLM backend abstraction — swap mock ↔ real without touching agent logic."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Optional


class Backend(ABC):
    """Minimal interface all backends must implement."""

    @abstractmethod
    def complete(self, system: str, messages: List[dict]) -> str:
        """Given a system prompt and a conversation history, return the next assistant turn."""
        ...


# ---------------------------------------------------------------------------
# Mock backend — deterministic, no API calls, good for tests
# ---------------------------------------------------------------------------

class MockBackend(Backend):
    """
    A scripted backend for testing.

    Supply a list of responses; they are returned in order.  If the list is
    exhausted, the last response is repeated.  Callers can also supply a
    callable ``response_fn(system, messages) -> str`` for dynamic behaviour.
    """

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        response_fn=None,
    ):
        if response_fn is not None and responses is not None:
            raise ValueError("Provide either responses or response_fn, not both")
        self._responses = list(responses or [])
        self._fn = response_fn
        self._call_count = 0

    def complete(self, system: str, messages: List[dict]) -> str:
        self._call_count += 1
        if self._fn is not None:
            return self._fn(system, messages)
        if not self._responses:
            return "# No response configured"
        idx = min(self._call_count - 1, len(self._responses) - 1)
        return self._responses[idx]

    @property
    def call_count(self) -> int:
        return self._call_count


# ---------------------------------------------------------------------------
# Anthropic backend — real API, requires ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------

class AnthropicBackend(Backend):
    """
    Thin wrapper around the Anthropic Messages API.

    Usage::

        backend = AnthropicBackend(model="claude-3-5-haiku-20241022")
        result = run_review_loop(task, backend=backend)

    Requires ``pip install anthropic``.
    """

    DEFAULT_MODEL = "claude-3-5-haiku-20241022"
    DEFAULT_MAX_TOKENS = 2048

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: Optional[str] = None,
    ):
        try:
            import anthropic  # noqa: F401  (lazy import)
        except ImportError as exc:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from exc

        self._model = model
        self._max_tokens = max_tokens
        self._api_key = api_key  # None → picks up ANTHROPIC_API_KEY from env

    def complete(self, system: str, messages: List[dict]) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text
