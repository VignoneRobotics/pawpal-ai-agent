import copy
from dataclasses import replace
from datetime import date
from pathlib import Path

import streamlit as st

from pawpal_agent import llm as llm_module
from pawpal_agent.agent import AgentResult, apply_proposal, run_agent
from pawpal_agent.logging_utils import TraceLogger
from pawpal_agent.proposal import Proposal, change_kind, describe_change
from pawpal_system import DailyPlan, Owner, Pet, Scheduler, Task

APP_DIR = Path(__file__).parent

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")

# Test hooks: AppTest sets these before the first run; real runs get the defaults.
if "data_file" not in st.session_state:
    st.session_state.data_file = APP_DIR / "data.json"
if "log_dir" not in st.session_state:
    st.session_state.log_dir = str(APP_DIR / "logs")
if "today" not in st.session_state:
    st.session_state.today = date.today()
if "llm_factory" not in st.session_state:
    # Looked up via the module so tests can monkeypatch load_api_key; the key itself is never stored here.
    has_key = llm_module.load_api_key(APP_DIR / ".env") is not None
    st.session_state.llm_factory = llm_module.AnthropicClient if has_key else None
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None
if "toasts" not in st.session_state:
    st.session_state.toasts = []  # (text, icon) waiting to be shown
if "errors" not in st.session_state:
    st.session_state.errors = {}  # section -> inline error, kept until that section succeeds

data_file = Path(st.session_state.data_file)
today: date = st.session_state.today

# Create the single Owner once (from data.json if it exists); later reruns reuse the stored object.
if "owner" not in st.session_state:
    st.session_state.owner = Owner(name="Jordan")
    if data_file.exists():
        try:
            st.session_state.owner = Owner.load_from_json(data_file)
        except ValueError as err:
            st.warning(f"Could not read {data_file.name} ({err}); starting fresh.")
owner: Owner = st.session_state.owner
scheduler = Scheduler(owner)


# --- helpers (callbacks read session state, because they run before the script reruns) ---


def save() -> None:
    """Write the current owner to the data file."""
    st.session_state.owner.save_to_json(st.session_state.data_file)


def log_event(event: str, request_id: str | None = None, **data: object) -> None:
    """Append one event to the agent trace log (joined to a proposal when request_id is given)."""
    TraceLogger(st.session_state.log_dir, request_id=request_id).log(event, **data)


def notify(text: str, icon: str = "✅") -> None:
    """Queue a floating toast; queued toasts are shown at the top and bottom of every run (survives st.rerun)."""
    st.session_state.toasts.append((text, icon))


def show_toasts() -> None:
    """Show and clear every queued toast."""
    while st.session_state.toasts:
        text, icon = st.session_state.toasts.pop(0)
        st.toast(text, icon=icon)


def report_error(section: str, text: str) -> None:
    """Toast an error and keep it inline in its section until that section succeeds."""
    st.session_state.errors[section] = text
    notify(text, icon="⚠️")


def clear_error(section: str) -> None:
    """Forget a section's inline error after a successful action there."""
    st.session_state.errors.pop(section, None)


def show_error(section: str) -> None:
    """Render the section's persisting inline error, if any."""
    if section in st.session_state.errors:
        st.error(st.session_state.errors[section])


def affected_days(proposal: Proposal) -> list[date]:
    """Return the planning day first, then every other day a change touches (move origin/target, add, complete)."""
    days: set[date] = set()
    for change in proposal.changes:
        found = owner.find_task(change.task_id)
        if found is not None:
            days.add(found[1].due_date)
        target = getattr(change, "new_date", None) or getattr(change, "due_date", None)
        if target is not None:
            days.add(target)
    return [proposal.day] + sorted(days - {proposal.day})


def split_reason(reason: str) -> tuple[str, str]:
    """Split a plan reason like "skipped: budget" into ("skipped", "budget")."""
    status, _, detail = reason.partition(": ")
    return status, detail


def task_rows(tasks: list[Task], reasons: dict[str, str] | None = None) -> list[dict]:
    """Return table rows (one dict per task) for st.table; adds a reason column when reasons are given."""
    rows = []
    for t in tasks:
        row = {
            "date": t.due_date.isoformat(),
            "time": t.time,
            "pet": t.pet_name,
            "task": t.description,
            "minutes": t.duration_minutes,
            "priority": t.priority,
            "frequency": t.frequency,
            "combinable": "yes" if t.combinable else "",
            "status": "✅ Done" if t.completed else "⏳ Pending",
        }
        if reasons is not None:
            row["reason"] = split_reason(reasons.get(t.task_id, ""))[1] or "fits the budget"
        rows.append(row)
    return rows


def preview_rows(plan: DailyPlan, changed_ids: set[str]) -> list[dict]:
    """Return compact before/after rows: time, task (changed ones marked), and plan status."""
    rows = []
    for t in sorted(plan.planned + plan.skipped, key=lambda t: (t.time, t.description)):
        status, detail = split_reason(plan.reasons.get(t.task_id, ""))
        rows.append({
            "time": t.time,
            "task": f"{t.description} ({t.pet_name})" + (" ◀" if t.task_id in changed_ids else ""),
            "plan": f"{status} ({detail})" if detail else status,
        })
    return rows


def overlap_pairs(plan: DailyPlan) -> list[tuple[Task, Task]]:
    """Return (planned, skipped) pairs where a task was skipped for overlapping a planned one, not yet allowed."""
    pairs = []
    for skipped in plan.skipped:
        if not plan.reasons.get(skipped.task_id, "").startswith("skipped: overlaps"):
            continue
        for planned in plan.planned:
            if scheduler.find_conflict_pairs([planned, skipped]) and not owner.is_overlap_allowed(
                planned.task_id, skipped.task_id, plan.day
            ):
                pairs.append((planned, skipped))
    return pairs


def on_approve() -> None:
    """Human approved: re-verify and apply the pending proposal, save, log, and clear it."""
    result: AgentResult | None = st.session_state.agent_result
    if result is None:
        return
    proposal = result.proposal
    request_id = result.trace[0]["request_id"] if result.trace else None
    changes = [f"{change_kind(c)} {c.task_id}" for c in proposal.changes]
    try:
        lines = apply_proposal(st.session_state.owner, proposal, st.session_state.today)
    except ValueError as err:
        report_error("agent", f"This proposal is out of date and was not applied: {err}. Please ask again.")
        log_event("human_decision", request_id, decision="approve", applied=False, changes=changes, error=str(err))
    else:
        save()
        clear_error("agent")
        notify("Approved and saved: " + "; ".join(lines))
        log_event("human_decision", request_id, decision="approve", applied=True, changes=changes)
    st.session_state.agent_result = None


def on_decline(decision: str) -> None:
    """Human rejected or dismissed the pending proposal: log it and clear it; the schedule is untouched."""
    result: AgentResult | None = st.session_state.agent_result
    if result is None:
        return
    request_id = result.trace[0]["request_id"] if result.trace else None
    changes = [f"{change_kind(c)} {c.task_id}" for c in result.proposal.changes]
    log_event("human_decision", request_id, decision=decision, applied=False, changes=changes)
    st.session_state.agent_result = None
    clear_error("agent")
    if decision == "reject":
        notify("Proposal rejected; nothing was changed.", icon="↩️")
    else:
        notify("Dismissed.", icon="ℹ️")


def on_do_both(task_id_a: str, task_id_b: str, day: date) -> None:
    """Human-only path: allow two overlapping tasks on this day, save, and log it (the agent has no such tool)."""
    try:
        st.session_state.owner.allow_overlap(task_id_a, task_id_b, day)
    except ValueError as err:
        report_error("schedule", f"Could not allow the overlap: {err}")
        return
    save()
    clear_error("schedule")
    log_event("human_override", task_a=task_id_a, task_b=task_id_b, day=day.isoformat())
    notify("OK, both tasks are allowed to overlap on that day.")


def render_trace(result: AgentResult) -> None:
    """Show the guard result, tool calls, verification results, and usage for the current proposal."""
    lines: list[str] = []
    for e in result.trace:
        if e["event"] == "guard":
            lines.append("Guard: " + ("allowed" if e["allowed"] else f"REFUSED ({e['category']})"))
        elif e["event"] == "tool_result":
            args = ", ".join(f"{k}={v!r}" for k, v in e["args"].items() if k != "rationale")
            lines.append(f"[attempt {e['attempt']}] {e['name']}({args})" + ("  ERROR" if e["is_error"] else ""))
        elif e["event"] == "attempt_failed":
            lines.append(f"[attempt {e['attempt']}] failed: " + "; ".join(e["errors"]))
        elif e["event"] == "verification":
            status = "passed" if e["ok"] else "failed: " + "; ".join(e["errors"])
            lines.append(f"[attempt {e['attempt']}] verification {status}")
        elif e["event"] == "output_guard":
            lines.append(f"[attempt {e['attempt']}] output guard redacted dosing text")
    u = result.usage
    lines.append(
        f"Usage: {u.llm_calls} LLM calls, {u.tool_turns} tool turns, tokens in {u.input_tokens} "
        f"(cache read {u.cache_read_tokens}), out {u.output_tokens}"
    )
    with st.expander("Agent trace"):
        st.code("\n".join(lines), language=None, wrap_lines=True)


def render_proposal(result: AgentResult) -> None:
    """Show the pending proposal: refusal, or summary/changes/verification/confidence, before/after, and buttons."""
    p: Proposal = result.proposal
    guard = next((e for e in result.trace if e["event"] == "guard"), None)

    if p.refusal:
        if guard is not None and not guard["allowed"]:
            st.warning(f"**Refused (rule-based):** {p.refusal}")
        else:
            st.warning(f"**Declined by the assistant:** {p.refusal}")
        st.button("Dismiss", key="dismiss", on_click=on_decline, args=("dismiss",))
        return

    st.markdown(f"**Proposal for {p.day.isoformat()}** — _{p.request}_")
    if p.summary:
        st.write(p.summary)
    for change in p.changes:
        st.markdown(f"- `{describe_change(change, owner)}`  \n  _Why:_ {change.rationale}")
    for note in p.notes:
        st.caption(f"Note: {note}")

    if not p.verified:
        st.error("**Not verified** — the safety checks rejected every attempt:\n\n"
                 + "\n".join(f"- {e}" for e in p.errors))
        st.button("Reject", key="reject", on_click=on_decline, args=("reject",))
        return

    if not p.changes:  # nothing to verify or approve: just the explanation
        st.info("No changes proposed.")
        st.button("Dismiss", key="dismiss", on_click=on_decline, args=("dismiss",))
        return

    st.success("**Verified** by the deterministic checks")
    st.caption(f"Confidence: {p.confidence:.0%} (model said {p.model_confidence:.2f}; retries: {p.retries})")

    # Dry run on a deep copy: shows the result and catches a proposal that went stale.
    after_owner = copy.deepcopy(owner)
    try:
        apply_proposal(after_owner, p, today)
    except ValueError as err:
        st.warning(f"This proposal is out of date: {err}. Reject it and ask again.")
        st.button("Reject", key="reject", on_click=on_decline, args=("reject",))
        return

    changed_ids = {c.task_id for c in p.changes}
    for day in affected_days(p):  # the planning day, then e.g. tomorrow if a task moves there
        st.markdown(f"##### {day.isoformat()}")
        before_col, after_col = st.columns(2)
        for column, label, plan_owner in ((before_col, "Before", owner), (after_col, "After", after_owner)):
            plan = Scheduler(plan_owner).build_daily_plan(day)
            with column:
                st.markdown(f"**{label}**")
                rows = preview_rows(plan, changed_ids)
                if rows:
                    st.table(rows)
                else:
                    st.caption("No tasks this day.")
                st.caption(f"{plan.minutes_used}/{plan_owner.available_minutes} min used")

    approve_col, reject_col = st.columns(2)
    with approve_col:
        st.button("Approve", key="approve", type="primary", on_click=on_approve)
    with reject_col:
        st.button("Reject", key="reject", on_click=on_decline, args=("reject",))


# --- page ---

st.title("🐾 PawPal+")

dev_mode = st.sidebar.toggle("Developer mode", key="dev_mode", help="Show the agent trace (tool calls, verification).")

show_toasts()  # queued by callbacks or before an st.rerun

st.subheader("Owner settings")
name_col, budget_col = st.columns(2)
with name_col:
    # A blank name keeps the current one.
    new_name = st.text_input("Owner name", value=owner.name).strip() or owner.name
with budget_col:
    new_budget = int(
        st.number_input("Daily time budget (minutes)", min_value=1, max_value=1440, value=owner.available_minutes)
    )
if new_name != owner.name or new_budget != owner.available_minutes:
    owner.name = new_name
    owner.available_minutes = new_budget
    save()

with st.form("windows_form"):
    windows_text = st.text_input(
        "Availability windows",
        value=", ".join(owner.availability_windows),
        placeholder="07:00-12:00, 16:00-22:00",
        help="Comma-separated HH:MM-HH:MM ranges. Leave blank to be available all day.",
        key="windows_text",
    )
    if st.form_submit_button("Save windows", key="save_windows"):
        new_windows = [w.strip() for w in windows_text.split(",") if w.strip()]
        try:
            replace(owner, availability_windows=new_windows)  # throwaway copy: runs Owner validation
        except ValueError as err:
            report_error("windows", str(err))
        else:
            owner.availability_windows = new_windows
            save()
            clear_error("windows")
            notify("Availability saved: " + (", ".join(new_windows) or "all day"))
show_error("windows")

st.divider()

st.subheader("Add a pet")
with st.form("add_pet_form", clear_on_submit=True):
    pet_name = st.text_input("Pet name")
    species = st.selectbox("Species", ["dog", "cat", "other"])
    age = st.number_input("Age", min_value=0, max_value=50, value=1)
    if st.form_submit_button("Add pet"):
        if not pet_name.strip():
            report_error("pet", "Pet name is required.")
        else:
            try:
                owner.add_pet(Pet(name=pet_name.strip(), species=species, age=int(age)))
            except ValueError as err:
                report_error("pet", str(err))
            else:
                save()
                clear_error("pet")
                notify(f"Added {pet_name.strip()}.")
show_error("pet")

st.subheader("Add a task")
if not owner.pets:
    st.info("Add a pet first.")
else:
    with st.form("add_task_form", clear_on_submit=True):
        chosen_pet = st.selectbox("Pet", [p.name for p in owner.pets])
        description = st.text_input("Task description", placeholder="e.g. Morning walk", key="task_description")
        col1, col2 = st.columns(2)
        with col1:
            task_time = st.time_input("Time", value=None)
            priority = st.selectbox("Priority", ["low", "medium", "high"], index=2)
        with col2:
            duration = st.number_input("Duration (minutes)", min_value=1, max_value=240, value=20)
            frequency = st.selectbox("Frequency", ["once", "daily", "weekly"])
        due = st.date_input("Due date", value=today)
        combinable = st.checkbox(
            "Combinable", help="Can overlap other combinable tasks (e.g. feeding both pets at once)."
        )
        if st.form_submit_button("Add task"):
            if not description.strip() or task_time is None:
                report_error("task", "Description and time are required.")
            else:
                pet = owner.get_pet(chosen_pet)
                try:
                    task = Task(
                        description=description.strip(),
                        time=task_time.strftime("%H:%M"),
                        duration_minutes=int(duration),
                        priority=priority,
                        frequency=frequency,
                        due_date=due,
                        combinable=combinable,
                    )
                except ValueError as err:
                    report_error("task", str(err))
                else:
                    pet.add_task(task)
                    save()
                    clear_error("task")
                    notify(f"Added '{task.description}' to {pet.name}.")
    show_error("task")

st.divider()

st.subheader("🤖 Ask PawPal+")
st.caption("The assistant proposes changes, the scheduler verifies them, and you decide.")
llm_factory = st.session_state.llm_factory
if llm_factory is None:
    st.info(
        "The assistant is off: no ANTHROPIC_API_KEY found.\n\n"
        "1. Copy `.env.example` to `.env`\n"
        "2. Put your key after `ANTHROPIC_API_KEY=`\n"
        "3. Restart `streamlit run app.py`\n\n"
        "Everything else in PawPal+ works without a key."
    )
with st.form("agent_form"):
    request = st.text_input(
        "Request", key="agent_request", placeholder="Move Mochi's bath so it doesn't overlap the walk"
    )
    agent_day = st.date_input("Planning day", value=today, key="agent_day")
    asked = st.form_submit_button("Ask PawPal+", key="ask", disabled=llm_factory is None)
if asked and llm_factory is not None:
    try:
        with st.spinner("Planning..."):
            st.session_state.agent_result = run_agent(
                request, owner, today, llm_factory(), day=agent_day, log_dir=st.session_state.log_dir
            )
    except llm_module.LLMError as err:  # messages are key-free by design
        report_error("agent", str(err))
    else:
        clear_error("agent")
show_error("agent")

if st.session_state.agent_result is not None:
    render_proposal(st.session_state.agent_result)
    if dev_mode:
        render_trace(st.session_state.agent_result)

st.divider()

st.subheader("Overdue tasks")
overdue = scheduler.overdue_tasks(today)
if overdue:
    rows = task_rows(overdue)
    for row, t in zip(rows, overdue):
        row["days overdue"] = (today - t.due_date).days
    st.table(rows)
else:
    st.success("Nothing overdue.")

# Placed above the task list so the table below reflects a completion on the same run.
st.subheader("Mark a task complete")
pending = scheduler.sort_by_time(scheduler.filter_tasks(completed=False))
if not pending:
    st.info("No pending tasks.")
else:
    labels = {t.task_id: f"{t.due_date.isoformat()} {t.time} - {t.pet_name}: {t.description}" for t in pending}
    chosen_id = st.selectbox("Pending task", list(labels), format_func=labels.get)
    if st.button("Mark complete"):
        next_task = scheduler.mark_task_complete(chosen_id, today=today)
        save()
        if next_task is not None:
            notify(
                f"Completed. Next {next_task.frequency} occurrence of "
                f"'{next_task.description}' ({next_task.pet_name}) added for {next_task.due_date.isoformat()}."
            )
        else:
            notify("Completed. No next occurrence (one-time task).")
        st.rerun()  # rebuild the dropdown; the queued toast is shown on the next run

st.subheader("Task list")
if not owner.get_all_tasks():
    st.info("No tasks yet.")
else:
    filter_pet_col, filter_status_col = st.columns(2)
    with filter_pet_col:
        pet_filter = st.selectbox("Pet", ["All"] + [p.name for p in owner.pets], key="filter_pet")
    with filter_status_col:
        status_filter = st.selectbox("Status", ["All", "Pending", "Done"], key="filter_status")
    visible = scheduler.sort_by_time(
        scheduler.filter_tasks(
            pet_name=None if pet_filter == "All" else pet_filter,
            completed={"All": None, "Pending": False, "Done": True}[status_filter],
        )
    )
    if visible:
        st.table(task_rows(visible))
    else:
        st.info("No tasks match these filters.")

st.subheader("Find a free slot")
slot_minutes = int(st.number_input("Task duration (minutes)", min_value=1, max_value=240, value=30, key="slot_minutes"))
free_slot = scheduler.next_available_slot(slot_minutes, today)
if free_slot is not None:
    st.success(f"Next free slot today: {free_slot}")
else:
    st.warning("No free slot today for that duration.")

st.divider()

st.subheader("Build Schedule")
plan_day = st.date_input("Day", value=today, key="plan_day")
if st.button("Generate schedule"):
    st.session_state["show_plan"] = True

# Recomputed every run while the flag is set, so the plan reflects new tasks, completions, and overrides.
if st.session_state.get("show_plan"):
    plan = scheduler.build_daily_plan(plan_day)

    allowed = [w for w in plan.warnings if w.endswith("(allowed by owner)")]
    for warning in allowed:
        st.info(warning)

    show_error("schedule")
    pairs = overlap_pairs(plan)
    for planned_task, skipped_task in pairs:
        st.warning(scheduler.detect_conflicts([planned_task, skipped_task])[0]
                   + f" - '{skipped_task.description}' was skipped.")
        slot = scheduler.next_available_slot(skipped_task.duration_minutes, skipped_task.due_date)
        if slot is not None:
            st.info(f"💡 Suggestion: move '{skipped_task.description}' to {slot}, or do both at once.")
        else:
            st.info(f"💡 No free slot left for '{skipped_task.description}' that day.")
        st.button(
            "Do both",
            key=f"do_both_{planned_task.task_id}_{skipped_task.task_id}",
            on_click=on_do_both,
            args=(planned_task.task_id, skipped_task.task_id, plan.day),
        )
    if not pairs:
        st.success(f"No unresolved overlaps ({len(allowed)} allowed by you)" if allowed else "No overlaps")

    if plan.planned:
        st.table(task_rows(scheduler.sort_by_time(plan.planned), plan.reasons))
    else:
        st.info("No tasks planned for this day.")

    if plan.skipped:
        st.subheader("Skipped tasks")
        st.caption("Each skipped task shows why: budget, overlap, or outside your availability windows.")
        st.table(task_rows(plan.skipped, plan.reasons))

    with st.expander("Why this plan?"):
        for line in scheduler.explain_plan(plan):
            st.write(line)

show_toasts()  # queued during this run (form submits)
