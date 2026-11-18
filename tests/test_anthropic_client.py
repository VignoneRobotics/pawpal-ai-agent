"""AnthropicClient request/response conversion with a stubbed SDK client (no network, no API key)."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from agent_fixtures import TODAY, make_world

from pawpal_agent.agent import SYSTEM_PROMPT, run_agent
from pawpal_agent.llm import (
    DEFAULT_MODEL,
    EFFORT,
    MAX_TOKENS,
    SETUP_MESSAGE,
    AnthropicClient,
    LLMError,
    MissingAPIKeyError,
)
from pawpal_agent.tools import TOOL_SCHEMAS

ROOT = Path(__file__).resolve().parent.parent


def usage(input_tokens=50, output_tokens=20, cache_read=None, cache_write=None):
    """Build an SDK-like usage object (the SDK uses None for counters that don't apply)."""
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                           cache_read_input_tokens=cache_read, cache_creation_input_tokens=cache_write)


def message(*blocks, stop_reason="end_turn", **usage_kwargs):
    """Build an SDK-like Message."""
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, usage=usage(**usage_kwargs))


def text_block(text):
    """Build an SDK-like text block."""
    return SimpleNamespace(type="text", text=text)


def tool_block(id_, name, input_):
    """Build an SDK-like tool_use block."""
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def thinking_block():
    """Build an SDK-like thinking block (text omitted, signature kept)."""
    return SimpleNamespace(type="thinking", thinking="", signature="sig-abc")


class StubSDK:
    """Stands in for anthropic.Anthropic: messages.create records kwargs and returns queued replies."""

    def __init__(self, *replies):
        """Queue replies; an Exception instance is raised instead of returned."""
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        """Record a snapshot of the request and return (or raise) the next reply."""
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def final_text(summary="Done.", confidence=0.8):
    """Return a final-reply JSON text block."""
    return text_block(json.dumps({"summary": summary, "confidence": confidence, "notes": [], "refusal": None}))


# --- request conversion ---


def test_request_uses_cached_system_block_unchanged_tools_and_auto_caching(monkeypatch):
    """System prompt carries a cache breakpoint, tools pass through untouched, and the conversation is auto-cached."""
    monkeypatch.delenv("PAWPAL_MODEL", raising=False)
    tools_before = copy.deepcopy(TOOL_SCHEMAS)
    stub = StubSDK(message(final_text()))
    messages = [{"role": "user", "content": "hi"}]

    AnthropicClient(client=stub).complete(SYSTEM_PROMPT, messages, TOOL_SCHEMAS)

    request = stub.requests[0]
    assert request["model"] == DEFAULT_MODEL == "claude-haiku-5-5"
    assert request["max_tokens"] == MAX_TOKENS
    assert request["output_config"] == {"effort": EFFORT}
    assert request["system"] == [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    assert request["cache_control"] == {"type": "ephemeral"}
    assert request["tools"] is TOOL_SCHEMAS and TOOL_SCHEMAS == tools_before
    assert request["messages"] == messages


def test_model_comes_from_pawpal_model_env(monkeypatch):
    """PAWPAL_MODEL overrides the default; an explicit argument overrides both."""
    monkeypatch.setenv("PAWPAL_MODEL", "claude-sonnet-5-5")
    assert AnthropicClient(client=StubSDK()).model == "claude-sonnet-5-5"
    assert AnthropicClient(model="x-model", client=StubSDK()).model == "x-model"


# --- response conversion ---


def test_response_maps_text_tool_calls_tokens_and_raw_blocks():
    """Text is joined, tool_use becomes ToolCall, input tokens include cache reads/writes, raw blocks are kept."""
    blocks = (thinking_block(), text_block("Checking. "), tool_block("toolu_1", "list_tasks", {"day": "2026-01-05"}))
    response = AnthropicClient.to_llm_response(
        message(*blocks, stop_reason="tool_use", input_tokens=40, output_tokens=25, cache_read=900, cache_write=60)
    )
    assert response.text == "Checking."
    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [("toolu_1", "list_tasks", {"day": "2026-01-05"})]
    assert (response.input_tokens, response.output_tokens) == (1000, 25)
    assert (response.cache_read_tokens, response.cache_write_tokens) == (900, 60)
    assert response.stop_reason == "tool_use"
    assert response.assistant_content == blocks


def test_missing_cache_counters_count_as_zero():
    """None usage counters (no caching on this call) become 0."""
    response = AnthropicClient.to_llm_response(message(final_text(), input_tokens=70, output_tokens=10))
    assert (response.input_tokens, response.cache_read_tokens, response.cache_write_tokens) == (70, 0, 0)


# --- end to end through the agent loop ---


def test_agent_sends_thinking_and_tool_use_blocks_back_unchanged(tmp_path):
    """The second request replays the model's own blocks (incl. thinking), then the tool_result."""
    world = make_world()
    thinking, tool_use = thinking_block(), tool_block("toolu_9", "list_tasks", {"day": TODAY.isoformat()})
    stub = StubSDK(
        message(thinking, tool_use, stop_reason="tool_use", input_tokens=500, cache_write=1200),
        message(thinking_block(), final_text("Nothing to change."), input_tokens=80, cache_read=1200),
    )

    result = run_agent("What's on today's plan?", world.owner, TODAY, AnthropicClient(client=stub), log_dir=tmp_path)

    second = stub.requests[1]["messages"]
    assert second[1] == {"role": "assistant", "content": [thinking, tool_use]}
    assert second[1]["content"][0] is thinking  # same object, not rebuilt
    tool_result = second[2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_9"
    assert result.proposal.verified and result.proposal.summary == "Nothing to change."
    assert (result.usage.cache_read_tokens, result.usage.cache_write_tokens) == (1200, 1200)
    assert result.usage.input_tokens == 500 + 1200 + 80 + 1200


# --- setup and errors ---


def test_missing_key_raises_setup_message(monkeypatch, tmp_path):
    """No env var and no .env -> MissingAPIKeyError with the setup instructions."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    with pytest.raises(MissingAPIKeyError) as info:
        AnthropicClient()
    assert str(info.value) == SETUP_MESSAGE and ".env.example" in SETUP_MESSAGE


def test_key_is_loaded_from_dotenv_file(monkeypatch, tmp_path):
    """A key in .env is picked up (and the client builds without any network call)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")  # records the original state so teardown restores it
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-test-not-real\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    AnthropicClient()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-not-real"


def test_sdk_errors_become_llm_error_without_the_key():
    """A connection failure (after SDK retries) is wrapped in LLMError with a short message."""
    anthropic = pytest.importorskip("anthropic")
    import httpx2

    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    client = AnthropicClient(client=StubSDK(error))
    with pytest.raises(LLMError, match="Could not reach the Anthropic API"):
        client.complete("sys", [{"role": "user", "content": "hi"}], [])


# --- demo smoke test ---


def test_agent_demo_mock_runs_offline_and_saves_utf8(tmp_path):
    """--mock runs all five scenarios with no key and writes the same output to --save as UTF-8."""
    out_file = tmp_path / "demo.txt"
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    run = subprocess.run(
        [sys.executable, "agent_demo.py", "--mock", "--save", str(out_file)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env,
    )
    assert run.returncode == 0, run.stderr
    saved = out_file.read_text(encoding="utf-8")
    assert saved.count("Verified: True") == 5
    assert saved.count("Owner approves (simulated)") == 3
    assert "REFUSED (dosing)" in saved and "REFUSED (out_of_scope)" in saved
    assert "SAMPLE 25 mg" not in saved  # dose text never shown to (or by) the agent
    assert saved.strip() in run.stdout
