"""LLM client interface, the real AnthropicClient, and a scripted FakeLLM for offline tests.

The anthropic and dotenv packages are imported lazily (inside functions) so `import pawpal_agent` stays offline.
"""

from __future__ import annotations

import copy
import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

DEFAULT_MODEL = "claude-haiku-5-5"
MAX_TOKENS = 4096  # Haiku 5.5 thinks by default and thinking counts toward this cap; our replies are short
EFFORT = "medium"  # Haiku 5.5's default effort, pinned explicitly (Phase 6 evals may try "low")
TIMEOUT_SECONDS = 60.0
MAX_RETRIES = 3  # SDK built-in retries: 408/409/429/5xx and connection errors
CACHE = {"type": "ephemeral"}
SETUP_MESSAGE = (
    "No ANTHROPIC_API_KEY found.\n"
    "  1. Copy .env.example to .env\n"
    "  2. Put your key after ANTHROPIC_API_KEY=\n"
    "  3. Run again, or use --mock to run the demo offline with scripted responses."
)


class LLMError(Exception):
    """A model call failed (setup, network, rate limit, or API error); the message is safe to show."""


class MissingAPIKeyError(LLMError):
    """ANTHROPIC_API_KEY is not set in the environment or .env."""

    def __init__(self) -> None:
        """Use the fixed setup message (never includes any key)."""
        super().__init__(SETUP_MESSAGE)


@dataclass(frozen=True)
class ToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class LLMResponse:
    """One model turn: either final text, or one or more tool calls (plus token counts)."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    input_tokens: int = 0  # total prompt tokens, including cached ones
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    stop_reason: str = "end_turn"  # "refusal" and "max_tokens" are handled by the agent
    # Provider-native assistant blocks (incl. thinking) to send back unchanged; empty for FakeLLM.
    assistant_content: tuple = ()


class LLMClient(Protocol):
    """Anything that can answer a system prompt + Anthropic-style messages + tool schemas."""

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Return the model's next turn."""
        ...


def load_api_key(dotenv_path: str | Path = ".env") -> str | None:
    """Load .env (without overriding real env vars) and return ANTHROPIC_API_KEY, or None if unset."""
    from dotenv import load_dotenv  # lazy: keeps `import pawpal_agent` dependency-light

    load_dotenv(dotenv_path, override=False)
    return os.environ.get("ANTHROPIC_API_KEY") or None


def _describe_api_error(err: Exception) -> str:
    """Return a short, key-free message for an anthropic SDK error."""
    import anthropic

    if isinstance(err, anthropic.AuthenticationError):
        return "The API key was rejected (401). Check ANTHROPIC_API_KEY in .env."
    if isinstance(err, anthropic.RateLimitError):
        return "Rate limited by the API (429) after retries. Wait a minute and try again."
    if isinstance(err, anthropic.APITimeoutError):
        return "The API request timed out after retries."
    if isinstance(err, anthropic.APIConnectionError):
        return "Could not reach the Anthropic API (network error) after retries."
    if isinstance(err, anthropic.APIStatusError):
        return f"The API returned an error ({err.status_code}): {err.message}"
    return f"API error: {err}"


def _usage_value(usage: Any, name: str) -> int:
    """Return a usage counter as int (the SDK uses None for counters that don't apply)."""
    return int(getattr(usage, name, 0) or 0)


class AnthropicClient:
    """Real LLMClient backed by the official anthropic SDK (Claude Haiku 5.5 by default)."""

    def __init__(self, model: str | None = None, api_key: str | None = None, client: Any = None) -> None:
        """Pick the model (arg, PAWPAL_MODEL, default) and build the SDK client; `client` injects a stub for tests."""
        self.model = model or os.environ.get("PAWPAL_MODEL") or DEFAULT_MODEL
        if client is not None:
            self._client = client
            return
        key = api_key or load_api_key()
        if not key:
            raise MissingAPIKeyError()
        import anthropic  # lazy import: only real runs need the SDK

        self._client = anthropic.Anthropic(api_key=key, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)

    def build_request(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        """Return the messages.create kwargs: cached system block, unchanged tools, automatic message caching."""
        return {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            # Render order is tools -> system -> messages: this breakpoint caches the tools AND the system prompt.
            "system": [{"type": "text", "text": system, "cache_control": CACHE}],
            "tools": tools,
            "messages": messages,
            # Automatic caching of the growing conversation, so later tool turns and retries re-read it.
            "cache_control": CACHE,
            "output_config": {"effort": EFFORT},
        }

    @staticmethod
    def to_llm_response(response: Any) -> LLMResponse:
        """Convert an SDK Message into an LLMResponse (text, tool calls, tokens, raw blocks)."""
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        usage = response.usage
        cache_read = _usage_value(usage, "cache_read_input_tokens")
        cache_write = _usage_value(usage, "cache_creation_input_tokens")
        return LLMResponse(
            text="".join(texts).strip(),
            tool_calls=tuple(calls),
            input_tokens=_usage_value(usage, "input_tokens") + cache_read + cache_write,
            output_tokens=_usage_value(usage, "output_tokens"),
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            stop_reason=response.stop_reason or "end_turn",
            assistant_content=tuple(response.content),
        )

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Call the Messages API once; SDK errors become LLMError with a safe message."""
        import anthropic

        try:
            response = self._client.messages.create(**self.build_request(system, messages, tools))
        except anthropic.APIError as err:
            raise LLMError(_describe_api_error(err)) from err
        return self.to_llm_response(response)


@dataclass(frozen=True)
class RecordedCall:
    """A snapshot of what the agent sent to FakeLLM on one call."""

    system: str
    messages: list[dict]
    tools: list[dict]


Step = LLMResponse | Callable[[list[dict]], LLMResponse]


@dataclass
class FakeLLM:
    """Scripted LLM: returns the given steps in order and records every call."""

    script: list[Step]
    calls: list[RecordedCall] = field(default_factory=list)

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Record the call and return the next scripted step; fail loudly if the script runs out."""
        self.calls.append(RecordedCall(system, copy.deepcopy(messages), copy.deepcopy(tools)))
        if len(self.calls) > len(self.script):
            raise AssertionError(f"FakeLLM script exhausted after {len(self.script)} steps")
        step = self.script[len(self.calls) - 1]
        return step(messages) if callable(step) else step


_call_ids = itertools.count(1)


def call(name: str, **arguments: object) -> ToolCall:
    """Build a ToolCall with a unique id (test/script helper)."""
    return ToolCall(id=f"toolu_{next(_call_ids):04d}", name=name, arguments=dict(arguments))


def tool_reply(*calls: ToolCall, input_tokens: int = 100, output_tokens: int = 20) -> LLMResponse:
    """Build an LLMResponse that asks for the given tool calls (test/script helper)."""
    return LLMResponse(tool_calls=tuple(calls), input_tokens=input_tokens, output_tokens=output_tokens)


def text_reply(text: str, input_tokens: int = 100, output_tokens: int = 30) -> LLMResponse:
    """Build a final-text LLMResponse (test/script helper)."""
    return LLMResponse(text=text, input_tokens=input_tokens, output_tokens=output_tokens)
