"""Streamlit AppTest smoke tests for the agent UI: FakeLLM only, tmp data file and logs, no network, no key."""

import json
from pathlib import Path

import pytest
from agent_fixtures import TODAY, final, make_world
from streamlit.testing.v1 import AppTest

import pawpal_agent.llm as llm_module
from pawpal_agent.llm import FakeLLM, call, tool_reply
from pawpal_agent.tools import TOOL_SCHEMAS
from pawpal_system import Owner, Task

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
REAL_DATA_FILE = ROOT / "data.json"


@pytest.fixture(autouse=True)
def real_data_file_untouched():
    """Fail any test that creates, changes, or deletes the repo's real data.json."""
    before = REAL_DATA_FILE.stat().st_mtime_ns if REAL_DATA_FILE.exists() else None
    yield
    after = REAL_DATA_FILE.stat().st_mtime_ns if REAL_DATA_FILE.exists() else None
    assert before == after, "a test touched the real data.json"


def start_app(tmp_path: Path, owner: Owner, llm_factory=None, inject_factory: bool = True) -> AppTest:
    """Save owner to a tmp data file, point the app at it (plus tmp logs and a fixed today), and run once."""
    data_file = tmp_path / "data.json"
    owner.save_to_json(data_file)
    at = AppTest.from_file(str(APP), default_timeout=15)
    at.session_state["data_file"] = data_file
    at.session_state["log_dir"] = str(tmp_path / "logs")
    at.session_state["today"] = TODAY
    if inject_factory:
        at.session_state["llm_factory"] = llm_factory
    at.run()
    assert not at.exception, at.exception
    return at


def ask(at: AppTest, request: str) -> AppTest:
    """Type a request into the agent panel and submit it."""
    at.text_input(key="agent_request").input(request)
    at.button(key="ask").click().run()
    assert not at.exception, at.exception
    return at


def trace_events(tmp_path: Path) -> list[dict]:
    """Read every event from the tmp trace log."""
    path = tmp_path / "logs" / "agent_trace.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def texts(elements) -> str:
    """Join the values of a list of AppTest elements into one string for substring checks."""
    return "\n".join(str(e.value) for e in elements)


def button_keys(at: AppTest) -> set[str]:
    """Return the keys of all buttons on the page."""
    return {b.key for b in at.button}


def move_play_script(world, new_time: str = "11:00") -> list:
    """Scripted model turns: propose moving Play, then the final JSON."""
    return [
        tool_reply(call("propose_move_task", task_id=world.play.task_id, new_time=new_time,
                        rationale="Later morning is free.")),
        final("Moved Play to 11:00.", 0.8),
    ]


def test_proposal_renders_with_rationale_confidence_preview_and_trace(tmp_path):
    world = make_world()
    at = start_app(tmp_path, world.owner, lambda: FakeLLM(move_play_script(world)))
    ask(at, "Move Play to 11:00")

    page = texts(at.markdown) + texts(at.success) + texts(at.caption)
    assert "Moved Play to 11:00." in page
    assert "Later morning is free." in page
    assert "Verified" in texts(at.success)
    assert "Confidence:" in texts(at.caption)
    assert "**Before**" in texts(at.markdown) and "**After**" in texts(at.markdown)
    assert {"approve", "reject"} <= button_keys(at)
    assert "propose_move_task" in texts(at.code)


def test_approve_applies_saves_to_tmp_file_and_logs_decision(tmp_path):
    world = make_world()
    at = start_app(tmp_path, world.owner, lambda: FakeLLM(move_play_script(world)))
    ask(at, "Move Play to 11:00")
    at.button(key="approve").click().run()
    assert not at.exception, at.exception

    saved = Owner.load_from_json(tmp_path / "data.json")
    assert saved.find_task(world.play.task_id)[1].time == "11:00"
    assert "Approved and saved" in texts(at.success)
    assert at.session_state["agent_result"] is None
    decisions = [e for e in trace_events(tmp_path) if e["event"] == "human_decision"]
    assert len(decisions) == 1 and decisions[0]["decision"] == "approve" and decisions[0]["applied"] is True
    request_ids = {e["request_id"] for e in trace_events(tmp_path) if e["event"] == "request"}
    assert decisions[0]["request_id"] in request_ids  # joined to the proposal's trace


def test_reject_leaves_schedule_unchanged(tmp_path):
    world = make_world()
    at = start_app(tmp_path, world.owner, lambda: FakeLLM(move_play_script(world)))
    before = (tmp_path / "data.json").read_text(encoding="utf-8")
    ask(at, "Move Play to 11:00")
    at.button(key="reject").click().run()
    assert not at.exception, at.exception

    assert at.session_state["owner"].find_task(world.play.task_id)[1].time == "10:00"
    assert (tmp_path / "data.json").read_text(encoding="utf-8") == before
    assert at.session_state["agent_result"] is None
    decisions = [e for e in trace_events(tmp_path) if e["event"] == "human_decision"]
    assert [d["decision"] for d in decisions] == ["reject"]


def test_guard_refusal_shows_rule_based_without_confidence(tmp_path):
    world = make_world()
    fake = FakeLLM([])
    at = start_app(tmp_path, world.owner, lambda: fake)
    ask(at, "How much Carprofen should I give Mochi?")

    assert "Refused (rule-based)" in texts(at.warning)
    assert "Confidence" not in texts(at.caption)
    assert "approve" not in button_keys(at)
    assert fake.calls == []  # the guard refused before any model call


def test_do_both_creates_override_saves_and_logs(tmp_path):
    world = make_world()
    bath = Task("Bath", "08:15", 30, "medium", due_date=TODAY)  # overlaps the 08:00 walk
    world.mochi.add_task(bath)
    at = start_app(tmp_path, world.owner)
    at.button[[b.label for b in at.button].index("Generate schedule")].click().run()
    key = f"do_both_{world.walk.task_id}_{bath.task_id}"
    assert key in button_keys(at)

    at.button(key=key).click().run()
    assert not at.exception, at.exception
    owner = at.session_state["owner"]
    assert owner.is_overlap_allowed(world.walk.task_id, bath.task_id, TODAY)
    assert len(Owner.load_from_json(tmp_path / "data.json").overlap_overrides) == 1
    assert key not in button_keys(at)  # the pair is now allowed
    overrides = [e for e in trace_events(tmp_path) if e["event"] == "human_override"]
    assert len(overrides) == 1


def test_agent_has_no_tool_to_allow_overlaps():
    names = {schema["name"] for schema in TOOL_SCHEMAS}
    assert not any("overlap" in name or "allow" in name for name in names)


def test_stale_approve_shows_error_and_applies_nothing(tmp_path):
    world = make_world()
    at = start_app(tmp_path, world.owner, lambda: FakeLLM(move_play_script(world)))
    ask(at, "Move Play to 11:00")
    owner = at.session_state["owner"]
    owner.find_task(world.play.task_id)[1].mark_complete()  # schedule changes after the proposal
    at.button(key="approve").click().run()
    assert not at.exception, at.exception

    assert "out of date" in texts(at.error)
    assert owner.find_task(world.play.task_id)[1].time == "10:00"
    decisions = [e for e in trace_events(tmp_path) if e["event"] == "human_decision"]
    assert decisions[0]["applied"] is False and decisions[0]["error"]


def test_unverified_proposal_shows_errors_and_no_approve(tmp_path):
    world = make_world()

    def script():
        """Every attempt moves Play onto the 08:00 walk (a new conflict)."""
        return FakeLLM([step for _ in range(3) for step in move_play_script(world, "08:00")])

    at = start_app(tmp_path, world.owner, script)
    ask(at, "Move Play to 08:00")

    assert "Not verified" in texts(at.error)
    assert "approve" not in button_keys(at)
    assert "reject" in button_keys(at)


def test_missing_key_disables_agent_but_app_works(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_module, "load_api_key", lambda *args, **kwargs: None)
    world = make_world()
    at = start_app(tmp_path, world.owner, inject_factory=False)

    assert "no ANTHROPIC_API_KEY found" in texts(at.info)
    assert at.button(key="ask").disabled
    assert "Generate schedule" in [b.label for b in at.button]


def test_windows_editor_validates_and_saves(tmp_path):
    world = make_world()
    at = start_app(tmp_path, world.owner)

    at.text_input(key="windows_text").input("9:00-12:00")
    at.button(key="save_windows").click().run()
    assert "Invalid window" in texts(at.error)
    assert at.session_state["owner"].availability_windows == []

    at.text_input(key="windows_text").input("07:00-12:00, 16:00-22:00")
    at.button(key="save_windows").click().run()
    assert Owner.load_from_json(tmp_path / "data.json").availability_windows == ["07:00-12:00", "16:00-22:00"]
