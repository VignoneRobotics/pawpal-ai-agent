"""The agent loop, driven entirely by FakeLLM scripts."""

import copy
import json

import pytest
from agent_fixtures import TODAY, final, make_world

from pawpal_agent.agent import FAILED_SUMMARY, MAX_TOOL_TURNS, MODEL_REFUSAL, run_agent
from pawpal_agent.guardrails import DOSING_REFUSAL, OUTPUT_REDACTED_NOTE
from pawpal_agent.llm import FakeLLM, LLMResponse, call, text_reply, tool_reply
from pawpal_agent.proposal import MoveChange


def move(task_id: str, new_time: str):
    """Script step: propose moving a task."""
    return tool_reply(call("propose_move_task", task_id=task_id, new_time=new_time, rationale="Owner asked"))


def list_today():
    """Script step: list today's tasks."""
    return tool_reply(call("list_tasks", day=TODAY.isoformat()))


def last_text(recorded_call) -> str:
    """Return all text the agent sent in its most recent user message."""
    content = recorded_call.messages[-1]["content"]
    if isinstance(content, str):
        return content
    return " ".join(block.get("text", "") + block.get("content", "") for block in content)


def test_happy_path_returns_verified_proposal_without_touching_owner(tmp_path):
    """list -> propose -> final JSON gives one verified change; the real owner is unchanged."""
    world = make_world()
    before = copy.deepcopy(world.owner)
    first = list_today()
    llm = FakeLLM([first, move(world.play.task_id, "11:00"), final("Moved Play to 11:00.", 0.8)])

    result = run_agent("Move Mochi's play to 11:00", world.owner, TODAY, llm, log_dir=tmp_path)

    proposal = result.proposal
    assert proposal.verified and proposal.errors == []
    assert proposal.changes == [MoveChange(world.play.task_id, "11:00", None, "Owner asked")]
    assert (proposal.summary, proposal.model_confidence, proposal.confidence) == ("Moved Play to 11:00.", 0.8, 0.8)
    assert world.owner == before
    tool_result = llm.calls[1].messages[-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == first.tool_calls[0].id
    assert (result.usage.llm_calls, result.usage.tool_turns) == (3, 2)


def test_first_message_has_day_context_but_no_dose_or_vet_info(tmp_path):
    """Only the planning day's tasks and owner limits are sent; private medication details are not."""
    world = make_world(windows=["07:00-12:00", "17:00-22:00"])
    llm = FakeLLM([final()])
    run_agent("What's on today's plan?", world.owner, TODAY, llm, log_dir=tmp_path)
    first = llm.calls[0].messages[0]["content"]
    assert "Carprofen" in first and "07:00-12:00" in first and "Daily budget: 480" in first
    assert "25 mg" not in first and "Dr. Lee" not in first and "555-0100" not in first
    assert "25 mg" not in llm.calls[0].system


def test_dosing_request_is_refused_with_zero_llm_calls(tmp_path):
    """The input guard answers dosing questions before the model is ever called."""
    world = make_world()
    llm = FakeLLM([])
    result = run_agent("How much Carprofen should I give Mochi?", world.owner, TODAY, llm, log_dir=tmp_path)
    assert llm.calls == []
    assert result.proposal.refusal == DOSING_REFUSAL and result.proposal.changes == []


def test_model_refusal_is_returned_without_changes(tmp_path):
    """If the model refuses (and calls no tools), the proposal carries the refusal."""
    world = make_world()
    llm = FakeLLM([final("I can't help with that.", 0.9, refusal="Not a pet scheduling request")])
    result = run_agent("Plan my day with the dog and also book me a flight", world.owner, TODAY, llm, log_dir=tmp_path)
    assert result.proposal.refusal == "Not a pet scheduling request" and result.proposal.changes == []


def test_verifier_failure_is_fed_back_and_sandbox_reset(tmp_path):
    """A conflicting proposal is rejected, the errors go back to the model, and attempt 2 starts clean."""
    world = make_world()
    llm = FakeLLM([
        move(world.play.task_id, "08:15"), final(),  # attempt 1: conflicts with the walk
        list_today(), move(world.play.task_id, "11:00"), final(confidence=0.8),  # attempt 2
    ])
    result = run_agent("Move play earlier", world.owner, TODAY, llm, log_dir=tmp_path)

    assert "new conflict" in last_text(llm.calls[2]) and "reset to the original" in last_text(llm.calls[2])
    listed = json.loads(llm.calls[3].messages[-1]["content"][0]["content"])
    assert next(t for t in listed["tasks"] if t["description"] == "Play")["time"] == "10:00"
    proposal = result.proposal
    assert proposal.verified and proposal.retries == 1 and proposal.confidence == 0.68
    assert [c.new_time for c in proposal.changes] == ["11:00"]


def test_three_failed_attempts_give_unverified_empty_proposal(tmp_path):
    """After 2 retries the agent gives up: nothing to approve, confidence 0."""
    world = make_world()
    llm = FakeLLM([move(world.play.task_id, "08:15"), final()] * 3)
    result = run_agent("Move play earlier", world.owner, TODAY, llm, log_dir=tmp_path)
    proposal = result.proposal
    assert not proposal.verified and proposal.confidence == 0.0 and proposal.changes == []
    assert proposal.summary == FAILED_SUMMARY and "new conflict" in proposal.errors[0]
    assert len(llm.calls) == 6


def test_tool_turn_cap_fails_the_attempt(tmp_path):
    """A 7th tool turn is not executed; the attempt fails and the model is told why."""
    world = make_world()
    llm = FakeLLM([list_today() for _ in range(MAX_TOOL_TURNS + 1)] + [final()])
    result = run_agent("What's on today's plan?", world.owner, TODAY, llm, log_dir=tmp_path)

    assert result.usage.tool_turns == MAX_TOOL_TURNS
    retry_message = last_text(llm.calls[-1])
    assert "tool turn limit reached" in retry_message and "more than 6 tool turns" in retry_message
    assert result.proposal.verified and result.proposal.retries == 1
    roles = [m["role"] for m in llm.calls[-1].messages]
    assert all(a != b for a, b in zip(roles, roles[1:]))  # user/assistant alternate


@pytest.mark.parametrize(
    "bad_reply, expected",
    [
        (text_reply("Sure, I moved it!"), "not valid JSON"),
        (final(confidence=1.5), "between 0 and 1"),
    ],
)
def test_malformed_final_reply_is_fed_back_and_retried(tmp_path, bad_reply, expected):
    """Invalid JSON or out-of-range confidence costs one retry with the parse error fed back."""
    world = make_world()
    llm = FakeLLM([bad_reply, final(confidence=0.9)])
    result = run_agent("What's on today's plan?", world.owner, TODAY, llm, log_dir=tmp_path)
    assert expected in last_text(llm.calls[1])
    assert result.proposal.verified and result.proposal.retries == 1


def test_output_guard_redacts_dose_amount_in_final_reply(tmp_path):
    """A model summary that states a dose is replaced, noted, and logged."""
    world = make_world()
    llm = FakeLLM([final("Moved play. Give her 50 mg after dinner.")])
    result = run_agent("What's on today's plan?", world.owner, TODAY, llm, log_dir=tmp_path)
    assert result.proposal.summary == DOSING_REFUSAL
    assert OUTPUT_REDACTED_NOTE in result.proposal.notes
    assert any(e["event"] == "output_guard" for e in result.trace)


def test_trace_is_written_as_jsonl_with_usage_and_no_private_details(tmp_path):
    """Every event shares a request_id; the final event has token totals; no dose/vet text is logged."""
    world = make_world()
    llm = FakeLLM([list_today(), move(world.play.task_id, "11:00"), final()])
    result = run_agent("Move Mochi's play to 11:00", world.owner, TODAY, llm, log_dir=tmp_path)

    text = (tmp_path / "agent_trace.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in text.splitlines()]
    assert [e["event"] for e in events] == [
        "request", "guard", "llm_call", "tool_result", "llm_call", "tool_result", "llm_call", "verification", "final",
    ]
    assert len({e["request_id"] for e in events}) == 1
    assert events[-1]["input_tokens"] == result.usage.input_tokens == 300
    assert events[-1]["tool_turns"] == 2
    assert "25 mg" not in text and "Dr. Lee" not in text and "555-0100" not in text
    assert events == result.trace


def test_model_safety_refusal_returns_refusal_without_changes(tmp_path):
    """stop_reason "refusal" ends the run at once: a refusal proposal with no changes and no retries."""
    world = make_world()
    llm = FakeLLM([LLMResponse(stop_reason="refusal")])
    result = run_agent("Move Mochi's play to 11:00", world.owner, TODAY, llm, log_dir=tmp_path)
    assert result.proposal.refusal == MODEL_REFUSAL and result.proposal.changes == []
    assert len(llm.calls) == 1 and result.trace[-1]["refusal"] == "model"


def test_max_tokens_cutoff_is_fed_back_and_retried(tmp_path):
    """A reply cut off by max_tokens costs one retry with a "use fewer words" message."""
    world = make_world()
    llm = FakeLLM([LLMResponse(text='{"summary": "Moved', stop_reason="max_tokens"), final(confidence=0.9)])
    result = run_agent("What's on today's plan?", world.owner, TODAY, llm, log_dir=tmp_path)
    assert "cut off" in last_text(llm.calls[1])
    assert result.proposal.verified and result.proposal.retries == 1


def test_cache_counters_are_summed_into_usage_and_trace(tmp_path):
    """Per-call cache read/write tokens add up in Usage and appear in the final trace event."""
    world = make_world()
    llm = FakeLLM([
        LLMResponse(tool_calls=list_today().tool_calls, input_tokens=1500, cache_write_tokens=1200),
        LLMResponse(text=final().text, input_tokens=1600, cache_read_tokens=1200),
    ])
    result = run_agent("What's on today's plan?", world.owner, TODAY, llm, log_dir=tmp_path)
    assert (result.usage.cache_read_tokens, result.usage.cache_write_tokens) == (1200, 1200)
    assert result.trace[-1]["cache_read_tokens"] == 1200
