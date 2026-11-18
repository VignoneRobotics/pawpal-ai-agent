"""FakeLLM behavior and the offline guarantee."""

import subprocess
import sys

import pytest

from pawpal_agent.llm import FakeLLM, LLMResponse, call, text_reply, tool_reply


def test_fake_llm_returns_steps_in_order_and_records_calls():
    """Each complete() returns the next scripted step and snapshots what was sent."""
    first = tool_reply(call("list_tasks", day="2026-01-05"))
    llm = FakeLLM([first, text_reply("done")])
    messages = [{"role": "user", "content": "hi"}]
    assert llm.complete("sys", messages, []) is first
    messages.append({"role": "assistant", "content": "later edit"})
    assert llm.complete("sys", messages, []).text == "done"
    assert len(llm.calls) == 2
    assert llm.calls[0].messages == [{"role": "user", "content": "hi"}]  # deep copy, not a live list
    assert llm.calls[1].system == "sys"


def test_fake_llm_supports_callable_steps():
    """A callable step can build its response from the messages it receives."""
    llm = FakeLLM([lambda messages: LLMResponse(text=str(len(messages)))])
    assert llm.complete("sys", [{"role": "user", "content": "x"}], []).text == "1"


def test_fake_llm_raises_when_script_is_exhausted():
    """Running past the script fails loudly so runaway loops are caught."""
    llm = FakeLLM([text_reply("only one")])
    llm.complete("sys", [], [])
    with pytest.raises(AssertionError, match="exhausted"):
        llm.complete("sys", [], [])


def test_tool_call_ids_are_unique():
    """The call() helper gives every ToolCall its own id."""
    assert call("a").id != call("a").id


def test_importing_the_agent_package_does_not_import_anthropic():
    """Phase 2 is fully offline: the anthropic SDK is never imported."""
    code = "import sys, pawpal_agent; sys.exit(1 if 'anthropic' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
