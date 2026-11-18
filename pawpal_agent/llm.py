"""LLM client interface and a scripted FakeLLM for offline tests (no network, no anthropic import)."""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Callable, Protocol


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
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    """Anything that can answer a system prompt + Anthropic-style messages + tool schemas."""

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Return the model's next turn."""
        ...


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
