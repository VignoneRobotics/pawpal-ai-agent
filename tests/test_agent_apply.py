"""apply_proposal: the only path that changes real state."""

import copy
from dataclasses import replace

import pytest
from agent_fixtures import TODAY, TOMORROW, final, make_world, proposal_of

from pawpal_agent.agent import apply_proposal, run_agent
from pawpal_agent.llm import FakeLLM, call, tool_reply
from pawpal_agent.proposal import AddChange, CompleteChange, MoveChange
from pawpal_system import Task


def verified(proposal):
    """Mark a hand-built proposal as verified (as run_agent would)."""
    return replace(proposal, verified=True)


def test_apply_moves_adds_and_completes_on_the_real_owner():
    """All three change kinds land on the real owner; the added task keeps the previewed id."""
    world = make_world()
    proposal = verified(proposal_of(
        MoveChange(world.play.task_id, "11:00", None, "later"),
        AddChange("new00001", "Bean", "Brush Bean", "12:00", 15, TODAY, "requested"),
        CompleteChange(world.feed.task_id, "already fed"),
    ))
    lines = apply_proposal(world.owner, proposal, TODAY)

    assert len(lines) == 3 and lines[0].startswith("Moved Play")
    assert world.play.time == "11:00"
    assert world.owner.find_task("new00001")[1].pet_name == "Bean"
    assert world.feed.completed
    assert any(t.due_date == TOMORROW and t.description == "Feed Bean" for t in world.bean.tasks)


@pytest.mark.parametrize(
    "make_proposal, message",
    [
        (lambda w: proposal_of(MoveChange(w.play.task_id, "11:00", None, "x")), "verified"),
        (lambda w: replace(proposal_of(), verified=True, refusal="no"), "refusal"),
    ],
)
def test_apply_refuses_unverified_or_refusal_proposals(make_proposal, message):
    """Unverified proposals and refusals change nothing."""
    world = make_world()
    before = copy.deepcopy(world.owner)
    with pytest.raises(ValueError, match=message):
        apply_proposal(world.owner, make_proposal(world), TODAY)
    assert world.owner == before


def test_apply_refuses_a_stale_proposal_without_changing_anything():
    """If the schedule changed after verification, re-verification blocks the apply."""
    world = make_world()
    proposal = verified(proposal_of(MoveChange(world.play.task_id, "11:00", None, "later")))
    world.bean.add_task(Task("Vet call", "11:00", 30, "high", due_date=TODAY))  # owner edits in the meantime
    before = copy.deepcopy(world.owner)
    with pytest.raises(ValueError, match="no longer passes"):
        apply_proposal(world.owner, proposal, TODAY)
    assert world.owner == before


def test_end_to_end_agent_proposal_then_human_approval(tmp_path):
    """run_agent proposes on a sandbox; only apply_proposal changes the owner."""
    world = make_world()
    llm = FakeLLM([
        tool_reply(call("propose_add_task", pet_name="Mochi", description="Evening walk", time="18:00",
                        duration_minutes=30, due_date=TODAY.isoformat(), rationale="Owner asked")),
        final("Added an evening walk at 18:00.", 0.85),
    ])
    proposal = run_agent("Add an evening walk for Mochi", world.owner, TODAY, llm, log_dir=tmp_path).proposal
    new_id = proposal.changes[0].task_id
    assert world.owner.find_task(new_id) is None  # not applied yet

    apply_proposal(world.owner, proposal, TODAY)
    assert world.owner.find_task(new_id)[1].description == "Evening walk"
