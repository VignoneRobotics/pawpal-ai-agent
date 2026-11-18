"""Tool schemas and sandbox dispatch."""

import json

import pytest
from agent_fixtures import TODAY, TOMORROW, make_world

from pawpal_agent.proposal import AddChange, CompleteChange, MoveChange
from pawpal_agent.tools import TOOL_SCHEMAS, ToolSession


def run(session: ToolSession, name: str, **args) -> tuple[dict, bool]:
    """Dispatch a tool and decode its JSON result."""
    text, is_error = session.dispatch(name, args)
    return json.loads(text), is_error


def test_tool_set_is_exactly_the_planned_seven():
    """Only the approved tools exist; nothing deletes, overrides, edits medication, or contacts anyone."""
    names = {schema["name"] for schema in TOOL_SCHEMAS}
    assert names == {
        "list_tasks", "find_conflicts", "next_available_slot", "build_daily_plan",
        "propose_move_task", "propose_add_task", "propose_complete_task",
    }


def test_no_tool_schema_exposes_forbidden_fields():
    """No tool accepts medication, dose, override, or task_id-for-add arguments."""
    forbidden = ("medication", "dose", "override", "allow_overlap", "vet", "delete")
    for schema in TOOL_SCHEMAS:
        for prop in schema["input_schema"]["properties"]:
            assert not any(word in prop for word in forbidden), (schema["name"], prop)
        assert schema["input_schema"]["additionalProperties"] is False
    add = next(s for s in TOOL_SCHEMAS if s["name"] == "propose_add_task")
    assert "task_id" not in add["input_schema"]["properties"]


def test_list_tasks_shows_prescribed_times_but_hides_dose_and_vet_info():
    """The agent sees that a task is a medication and its times, never the dose or vet contact."""
    world = make_world()
    session = ToolSession(world.owner, TODAY)
    result, is_error = run(session, "list_tasks", day=TODAY.isoformat())
    assert not is_error
    assert [t["description"] for t in result["tasks"]] == ["Morning walk", "Feed Bean", "Play", "Carprofen"]
    dose = result["tasks"][-1]
    assert dose["is_medication"] is True and dose["prescribed_times"] == ["08:00", "20:00"]
    text = json.dumps(result)
    assert "25 mg" not in text and "Dr. Lee" not in text and "555-0100" not in text


def test_list_tasks_filters_by_pet_and_day():
    """pet_name and day narrow the list."""
    world = make_world()
    session = ToolSession(world.owner, TODAY)
    result, _ = run(session, "list_tasks", day=TODAY.isoformat(), pet_name="bean")
    assert [t["description"] for t in result["tasks"]] == ["Feed Bean"]
    result, _ = run(session, "list_tasks", day=TOMORROW.isoformat())
    assert result["tasks"] == []


def test_propose_move_changes_sandbox_only_and_records_change():
    """A proposed move is visible in the sandbox; the real owner is untouched."""
    world = make_world()
    session = ToolSession(world.owner, TODAY)
    result, is_error = run(session, "propose_move_task", task_id=world.play.task_id, new_time="11:00",
                           rationale="Free late morning")
    assert not is_error and result["ok"]
    assert world.play.time == "10:00"
    assert session.sandbox.find_task(world.play.task_id)[1].time == "11:00"
    assert session.changes == [MoveChange(world.play.task_id, "11:00", None, "Free late morning")]


def test_find_conflicts_reflects_a_proposed_move():
    """Read tools see earlier proposals because they run on the same sandbox."""
    world = make_world()
    session = ToolSession(world.owner, TODAY)
    run(session, "propose_move_task", task_id=world.play.task_id, new_time="08:15", rationale="test")
    result, _ = run(session, "find_conflicts", day=TODAY.isoformat())
    assert len(result["conflicts"]) == 1 and "Play" in result["conflicts"][0]
    assert ToolSession(world.owner, TODAY).scheduler.detect_conflicts() == []  # real owner unchanged


def test_propose_add_generates_task_id_and_records_change():
    """The tool picks the new id; the change carries it so preview and apply match."""
    world = make_world()
    session = ToolSession(world.owner, TODAY)
    result, is_error = run(session, "propose_add_task", pet_name="Bean", description="Brush Bean", time="11:00",
                           duration_minutes=15, due_date=TODAY.isoformat(), rationale="Owner asked")
    assert not is_error
    change = session.changes[0]
    assert isinstance(change, AddChange) and change.task_id == result["task_id"]
    assert session.sandbox.find_task(change.task_id)[1].pet_name == "Bean"
    assert world.owner.find_task(change.task_id) is None


def test_propose_complete_recurring_adds_next_occurrence_in_sandbox():
    """Completing a daily task in the sandbox creates tomorrow's copy there only."""
    world = make_world()
    session = ToolSession(world.owner, TODAY)
    result, is_error = run(session, "propose_complete_task", task_id=world.feed.task_id, rationale="Already fed")
    assert not is_error and "next one due" in result["result"]
    assert isinstance(session.changes[0], CompleteChange)
    assert len(session.sandbox.get_all_tasks()) == 5
    assert len(world.owner.get_all_tasks()) == 4 and not world.feed.completed


@pytest.mark.parametrize(
    "name, args",
    [
        ("propose_add_task", {"pet_name": "Mochi", "description": "Pill", "time": "11:00", "duration_minutes": 5,
                              "due_date": "2026-01-05", "rationale": "x", "medication": {"dose_text": "50 mg"}}),
        ("propose_move_task", {"task_id": "abc", "new_time": "09:00"}),  # missing rationale
        ("propose_move_task", {"task_id": "abc", "new_time": "09:00", "rationale": "  "}),  # empty rationale
        ("propose_move_task", {"task_id": "abc", "new_time": "09:00", "rationale": "x", "dose_text": "1"}),
        ("propose_move_task", {"task_id": "abc", "new_time": "09:00", "rationale": "x"}),  # unknown id
        ("propose_add_task", {"pet_name": "Mochi", "description": "Walk", "time": "11:00", "duration_minutes": "5",
                              "due_date": "2026-01-05", "rationale": "x"}),  # wrong type
        ("list_tasks", {"day": "05/01/2026"}),  # bad date
        ("delete_task", {"task_id": "abc"}),  # unknown tool
        ("allow_overlap", {"task_a": "a", "task_b": "b"}),  # unknown tool
    ],
)
def test_bad_tool_calls_return_errors_and_record_nothing(name, args):
    """Unknown tools, extra/missing/wrong-type args, and bad ids are errors, never exceptions."""
    world = make_world()
    session = ToolSession(world.owner, TODAY)
    result, is_error = run(session, name, **args)
    assert is_error and "error" in result
    assert session.changes == []


def test_bad_move_time_leaves_sandbox_unchanged():
    """A logic-layer ValueError is returned to the model and nothing moves."""
    world = make_world()
    session = ToolSession(world.owner, TODAY)
    result, is_error = run(session, "propose_move_task", task_id=world.walk.task_id, new_time="25:00", rationale="x")
    assert is_error and "Invalid time" in result["error"]
    assert session.sandbox.find_task(world.walk.task_id)[1].time == "08:00"


def test_reset_discards_sandbox_and_changes():
    """reset() starts again from the real owner."""
    world = make_world()
    session = ToolSession(world.owner, TODAY)
    run(session, "propose_move_task", task_id=world.play.task_id, new_time="11:00", rationale="x")
    session.reset()
    assert session.changes == []
    assert session.sandbox.find_task(world.play.task_id)[1].time == "10:00"


def test_next_slot_and_daily_plan_tools():
    """Read tools wrap the scheduler and return JSON-friendly results."""
    world = make_world(windows=["07:00-12:00", "17:00-22:00"])
    session = ToolSession(world.owner, TODAY)
    slot, _ = run(session, "next_available_slot", duration_minutes=60, day=TODAY.isoformat())
    assert slot["slot"] == "07:00"
    plan, _ = run(session, "build_daily_plan", day=TODAY.isoformat())
    assert [p["description"] for p in plan["planned"]] == ["Morning walk", "Feed Bean", "Play", "Carprofen"]
    assert plan["budget_minutes"] == 480 and plan["skipped"] == []
