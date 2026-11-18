"""PawPal+ agent CLI demo: five requests end to end (guard -> tools -> proposal -> verify -> simulated approval).

    python agent_demo.py                       # real Claude Haiku (needs ANTHROPIC_API_KEY in .env)
    python agent_demo.py --mock                # offline: FakeLLM scripted responses, no key, no network
    python agent_demo.py --mock --save PATH    # also write the output to PATH as UTF-8
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from pawpal_agent.agent import AgentResult, apply_proposal, run_agent
from pawpal_agent.llm import AnthropicClient, FakeLLM, LLMClient, LLMError, Step, call, text_reply, tool_reply
from pawpal_agent.proposal import AddChange, CompleteChange, MoveChange, Proposal
from pawpal_system import Medication, Owner, Pet, Scheduler, Task

# Piped Windows output defaults to cp1252; keep the demo output safe to redirect.
sys.stdout.reconfigure(encoding="utf-8")

TODAY = date(2026, 3, 2)
TOMORROW = TODAY + timedelta(days=1)
RULE = "=" * 72


@dataclass
class Household:
    """The sample owner plus the tasks the mock scripts need to reference by id."""

    owner: Owner
    bath: Task
    evening_dose: Task


@dataclass
class Scenario:
    """One demo request, the day whose plan to show afterwards, and its FakeLLM script for --mock."""

    key: str
    request: str
    show_day: date
    mock_script: Callable[[Household], list[Step]]


def make_household() -> Household:
    """Build the fixed sample household (fake data, fixed dates, no data.json)."""
    owner = Owner("Sam", available_minutes=150, availability_windows=["07:00-12:00", "16:00-22:00"])
    mochi, bean = Pet("Mochi", "dog", 4), Pet("Bean", "cat", 2)
    owner.add_pet(mochi)
    owner.add_pet(bean)
    carprofen = Medication(drug="Carprofen", dose_text="SAMPLE 25 mg tablet", times=("08:00", "20:00"), attested=True)

    mochi.add_task(Task("Morning walk", "07:30", 30, "high", "daily", due_date=TODAY))
    mochi.add_task(Task.medication_dose(carprofen, "08:00", 5, TODAY))
    bean.add_task(Task("Breakfast", "08:15", 10, "high", "daily", due_date=TODAY))
    bath = Task("Bath", "17:00", 30, "medium", due_date=TODAY)  # overlaps the evening walk
    mochi.add_task(bath)
    mochi.add_task(Task("Evening walk", "17:15", 30, "high", due_date=TODAY))
    bean.add_task(Task("Laser play", "19:00", 20, "low", due_date=TODAY))
    evening_dose = Task.medication_dose(carprofen, "20:00", 5, TODAY)
    mochi.add_task(evening_dose)

    mochi.add_task(Task("Morning walk", "07:30", 30, "high", due_date=TOMORROW))
    mochi.add_task(Task.medication_dose(carprofen, "08:00", 5, TOMORROW))
    mochi.add_task(Task.medication_dose(carprofen, "20:00", 5, TOMORROW))
    return Household(owner, bath, evening_dose)


def final_json(summary: str, confidence: float, notes: list[str] | None = None) -> str:
    """Return the strict final-reply JSON the model is asked to produce."""
    return json.dumps({"summary": summary, "confidence": confidence, "notes": notes or [], "refusal": None})


def script_conflict(h: Household) -> list[Step]:
    """(a) Look at the conflict, move the lower-priority bath to a free slot."""
    return [
        tool_reply(call("find_conflicts", day=TODAY.isoformat())),
        tool_reply(call("propose_move_task", task_id=h.bath.task_id, new_time="16:00",
                        rationale="Bath is lower priority than the walk, and 16:00-16:30 is free.")),
        text_reply(final_json("Moved Mochi's bath to 16:00 so it no longer overlaps the evening walk.", 0.85)),
    ]


def script_grooming(h: Household) -> list[Step]:
    """(b) Ask for a free 30-minute slot tomorrow, then add grooming there (slot read from the tool result)."""

    def add_at_slot(messages: list[dict]):
        """Read the slot from the last tool result and propose the add there."""
        slot = json.loads(messages[-1]["content"][0]["content"])["slot"]
        return tool_reply(call("propose_add_task", pet_name="Mochi", description="Grooming session", time=slot,
                               duration_minutes=30, due_date=TOMORROW.isoformat(), priority="medium",
                               rationale=f"{slot} is the first free 30-minute slot tomorrow."))

    return [
        tool_reply(call("next_available_slot", duration_minutes=30, day=TOMORROW.isoformat())),
        add_at_slot,
        text_reply(final_json("Added a 30-minute grooming session for Mochi tomorrow at the first free slot.", 0.9)),
    ]


def script_dose(h: Household) -> list[Step]:
    """(c) List Mochi's tasks, move the evening dose 30 min later (within the 60-min tolerance)."""
    return [
        tool_reply(call("list_tasks", day=TODAY.isoformat(), pet_name="Mochi")),
        tool_reply(call("propose_move_task", task_id=h.evening_dose.task_id, new_time="20:30",
                        rationale="Owner asked for 30 minutes later; 20:30 is within 60 min of the prescribed 20:00.")),
        text_reply(final_json("Moved Mochi's evening Carprofen dose half an hour later, to 20:30.", 0.9,
                              ["The dose amount is unchanged."])),
    ]


SCENARIOS = [
    Scenario("a", "Fix today's conflict between Mochi's bath and evening walk.", TODAY, script_conflict),
    Scenario("b", "Add a 30-minute grooming session for Mochi tomorrow.", TOMORROW, script_grooming),
    Scenario("c", "Move Mochi's evening Carprofen dose 30 minutes later.", TODAY, script_dose),
    Scenario("d", "How much Carprofen should I give Mochi?", TODAY, lambda h: []),
    Scenario("e", "Can you book me a flight to Denver?", TODAY, lambda h: []),
]


class Output:
    """Print lines and keep a copy so --save can write them as UTF-8."""

    def __init__(self) -> None:
        """Start with no lines."""
        self.lines: list[str] = []

    def __call__(self, line: str = "") -> None:
        """Print one line and remember it."""
        print(line)
        self.lines.append(line)

    def save(self, path: Path) -> None:
        """Write everything printed so far to path as UTF-8."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def describe_change(change: object, owner: Owner) -> str:
    """Return a one-line description of a proposed change using task names from the owner."""
    if isinstance(change, AddChange):
        return (f"ADD {change.description} ({change.pet_name}) on {change.due_date.isoformat()} at {change.time}, "
                f"{change.duration_minutes} min, {change.priority}")
    found = owner.find_task(change.task_id)
    name = f"{found[1].description} ({found[1].pet_name}) at {found[1].time}" if found else change.task_id
    if isinstance(change, MoveChange):
        target = f"{change.new_date.isoformat()} " if change.new_date else ""
        return f"MOVE {name} -> {target}{change.new_time}"
    if isinstance(change, CompleteChange):
        return f"COMPLETE {name}"
    return repr(change)


def print_plan(out: Output, owner: Owner, day: date) -> None:
    """Print the day's plan explanation and any conflicts."""
    scheduler = Scheduler(owner)
    plan = scheduler.build_daily_plan(day)
    out(f"  Plan for {day.isoformat()} ({plan.minutes_used}/{owner.available_minutes} min used):")
    for line in scheduler.explain_plan(plan):
        out(f"    {line}")
    pending = [t for t in owner.get_all_tasks() if t.due_date == day and not t.completed]
    conflicts = scheduler.detect_conflicts(pending)
    out("  Conflicts: " + ("; ".join(conflicts) if conflicts else "none"))


def print_result(out: Output, result: AgentResult, owner: Owner) -> None:
    """Print guard -> tool calls -> proposal -> usage from the result and its trace."""
    guard = next(e for e in result.trace if e["event"] == "guard")
    out(f"Guard: {'allowed' if guard['allowed'] else 'REFUSED (' + guard['category'] + ')'}")
    tools = [e for e in result.trace if e["event"] == "tool_result"]
    out("Tool calls:" + ("" if tools else " none"))
    for e in tools:
        args = ", ".join(f"{k}={v!r}" for k, v in e["args"].items() if k != "rationale")
        out(f"  - [attempt {e['attempt']}] {e['name']}({args}){'  ERROR' if e['is_error'] else ''}")

    p: Proposal = result.proposal
    out("Proposal:")
    if p.refusal:
        out(f"  Refusal: {p.refusal}")
    if p.summary:
        out(f"  Summary: {p.summary}")
    for change in p.changes:
        out(f"  Change: {describe_change(change, owner)}")
        out(f"    Rationale: {change.rationale}")
    for note in p.notes:
        out(f"  Note: {note}")
    for error in p.errors:
        out(f"  Verifier error: {error}")
    out(f"  Verified: {p.verified} | retries: {p.retries} | "
        f"confidence: model {p.model_confidence:.2f} -> final {p.confidence:.2f}")
    u = result.usage
    out(f"Usage: {u.llm_calls} LLM calls, {u.tool_turns} tool turns, tokens in {u.input_tokens} "
        f"(cache read {u.cache_read_tokens}, cache write {u.cache_write_tokens}), out {u.output_tokens}")


def run_scenario(out: Output, scenario: Scenario, llm_factory: Callable[[Household], LLMClient],
                 log_dir: str | None) -> None:
    """Run one request on a fresh household, then simulate owner approval and show the resulting plan."""
    household = make_household()
    owner = household.owner
    out(RULE)
    out(f"({scenario.key}) Request: {scenario.request}")
    out(RULE)
    try:
        result = run_agent(scenario.request, owner, TODAY, llm_factory(household), log_dir=log_dir)
    except LLMError as err:
        out(f"LLM error: {err}")
        return
    print_result(out, result, owner)

    p = result.proposal
    if p.verified and p.changes:
        out("Owner approves (simulated) -> apply_proposal:")
        for line in apply_proposal(owner, p, TODAY):
            out(f"  {line}")
        print_plan(out, owner, scenario.show_day)
    elif not p.changes:
        out("Nothing to approve; schedule unchanged.")
    out()


def main(argv: list[str] | None = None) -> int:
    """Parse flags, pick the real or scripted LLM, and run every scenario."""
    parser = argparse.ArgumentParser(description="PawPal+ agent demo")
    parser.add_argument("--mock", action="store_true", help="use FakeLLM scripted responses (no key, no network)")
    parser.add_argument("--save", type=Path, metavar="PATH", help="also write the output to PATH as UTF-8")
    args = parser.parse_args(argv)
    out = Output()

    client: LLMClient | None = None
    if args.mock:
        mode = "MOCK: FakeLLM scripted responses (no API calls; token counts are placeholders)"
        log_dir = None
    else:
        try:
            client = AnthropicClient()
        except LLMError as err:
            print(err)
            return 1
        mode = f"REAL: {client.model} via the Anthropic API (traces in logs/agent_trace.jsonl)"
        log_dir = "logs"

    out(f"PawPal+ agent demo | {mode}")
    out(f"Today is fixed to {TODAY.isoformat()} (sample data only)")
    out()
    for scenario in SCENARIOS:
        if client is None:  # mock: a fresh scripted FakeLLM per scenario (scripts need the household's task ids)
            run_scenario(out, scenario, lambda h, s=scenario: FakeLLM(s.mock_script(h)), log_dir)
        else:  # real: one client (and one prompt cache) for every scenario
            run_scenario(out, scenario, lambda h: client, log_dir)

    if args.save:
        out.save(args.save)
        print(f"Saved output to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
