"""Agent tools: JSON schemas and dispatch to Scheduler methods on a deep-copied sandbox Owner."""

from __future__ import annotations

import copy
import json
import uuid
from datetime import date

from pawpal_system import Owner, Scheduler, Task

from .proposal import Change, apply_change, parse_change

_DAY = {"type": "string", "description": "Date as YYYY-MM-DD"}
_HHMM = {"type": "string", "description": "Zero-padded 24h time HH:MM"}
_RATIONALE = {"type": "string", "description": "One short sentence: why this change helps"}

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "list_tasks",
        "description": "List all tasks due on a day (optionally for one pet).",
        "input_schema": {
            "type": "object",
            "properties": {"day": _DAY, "pet_name": {"type": "string"}},
            "required": ["day"],
            "additionalProperties": False,
        },
    },
    {
        "name": "find_conflicts",
        "description": "List overlapping pending tasks on a day.",
        "input_schema": {
            "type": "object",
            "properties": {"day": _DAY},
            "required": ["day"],
            "additionalProperties": False,
        },
    },
    {
        "name": "next_available_slot",
        "description": "First free start time on a day that fits the duration, inside the owner's availability.",
        "input_schema": {
            "type": "object",
            "properties": {
                "duration_minutes": {"type": "integer", "minimum": 1},
                "day": _DAY,
                "window_start": _HHMM,
                "window_end": _HHMM,
            },
            "required": ["duration_minutes", "day"],
            "additionalProperties": False,
        },
    },
    {
        "name": "build_daily_plan",
        "description": "Build the day's plan: planned and skipped tasks with reasons and minutes used.",
        "input_schema": {
            "type": "object",
            "properties": {"day": _DAY},
            "required": ["day"],
            "additionalProperties": False,
        },
    },
    {
        "name": "propose_move_task",
        "description": "Propose moving a pending task to a new time (and optional new date).",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "new_time": _HHMM,
                "new_date": _DAY,
                "rationale": _RATIONALE,
            },
            "required": ["task_id", "new_time", "rationale"],
            "additionalProperties": False,
        },
    },
    {
        "name": "propose_add_task",
        "description": "Propose adding a new care task (not medication) for a pet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pet_name": {"type": "string"},
                "description": {"type": "string"},
                "time": _HHMM,
                "duration_minutes": {"type": "integer", "minimum": 1},
                "due_date": _DAY,
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "frequency": {"type": "string", "enum": ["once", "daily", "weekly"]},
                "combinable": {"type": "boolean"},
                "rationale": _RATIONALE,
            },
            "required": ["pet_name", "description", "time", "duration_minutes", "due_date", "rationale"],
            "additionalProperties": False,
        },
    },
    {
        "name": "propose_complete_task",
        "description": "Propose marking a pending task complete.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}, "rationale": _RATIONALE},
            "required": ["task_id", "rationale"],
            "additionalProperties": False,
        },
    },
]

_SCHEMAS_BY_NAME = {schema["name"]: schema["input_schema"] for schema in TOOL_SCHEMAS}


def serialize_task(task: Task) -> dict:
    """Return the agent-visible view of a task: no dose text and no vet contact info."""
    return {
        "task_id": task.task_id,
        "pet": task.pet_name,
        "description": task.description,
        "time": task.time,
        "duration_minutes": task.duration_minutes,
        "priority": task.priority,
        "frequency": task.frequency,
        "due_date": task.due_date.isoformat(),
        "completed": task.completed,
        "combinable": task.combinable,
        "is_medication": task.medication is not None,
        "prescribed_times": list(task.medication.times) if task.medication else [],
    }


def tasks_for_day(owner: Owner, day: date, pet_name: str | None = None) -> list[dict]:
    """Return serialized tasks due on day (optionally one pet), sorted by time."""
    scheduler = Scheduler(owner)
    tasks = [t for t in scheduler.filter_tasks(pet_name=pet_name) if t.due_date == day]
    return [serialize_task(t) for t in scheduler.sort_by_time(tasks)]


def _check_args(name: str, args: object) -> dict:
    """Raise ValueError unless args is a dict with the tool's required keys and no unknown keys."""
    if not isinstance(args, dict):
        raise ValueError(f"{name} arguments must be a JSON object")
    schema = _SCHEMAS_BY_NAME[name]
    missing = set(schema["required"]) - set(args)
    extra = set(args) - set(schema["properties"])
    if missing:
        raise ValueError(f"{name} is missing {sorted(missing)}")
    if extra:
        raise ValueError(f"{name} does not accept {sorted(extra)}")
    return args


def _parse_day(value: object) -> date:
    """Parse a "YYYY-MM-DD" string; raise ValueError otherwise."""
    if not isinstance(value, str):
        raise ValueError(f"day must be a YYYY-MM-DD string, got {value!r}")
    return date.fromisoformat(value)


class ToolSession:
    """Runs tools against a deep copy of the owner and records proposed changes."""

    def __init__(self, owner: Owner, today: date) -> None:
        """Keep the real owner for resets; all tool work happens on a sandbox copy."""
        self._original = owner
        self.today = today
        self.reset()

    def reset(self) -> None:
        """Throw away the sandbox and recorded changes and start again from the real owner."""
        self.sandbox = copy.deepcopy(self._original)
        self.scheduler = Scheduler(self.sandbox)
        self.changes: list[Change] = []

    def dispatch(self, name: str, args: object) -> tuple[str, bool]:
        """Run one tool; return (JSON result text, is_error). Never raises for bad model input."""
        if name not in _SCHEMAS_BY_NAME:
            return json.dumps({"error": f"Unknown tool {name!r}"}), True
        try:
            result = getattr(self, f"_tool_{name}")(_check_args(name, args))
        except (ValueError, TypeError) as err:
            return json.dumps({"error": str(err)}), True
        return json.dumps(result), False

    # --- read-only tools ---

    def _tool_list_tasks(self, args: dict) -> dict:
        """List tasks due on a day."""
        day = _parse_day(args["day"])
        return {"day": day.isoformat(), "tasks": tasks_for_day(self.sandbox, day, args.get("pet_name"))}

    def _tool_find_conflicts(self, args: dict) -> dict:
        """List conflict warnings for a day."""
        day = _parse_day(args["day"])
        tasks = [t for t in self.sandbox.get_all_tasks() if t.due_date == day]
        return {"day": day.isoformat(), "conflicts": self.scheduler.detect_conflicts(tasks)}

    def _tool_next_available_slot(self, args: dict) -> dict:
        """Find the first free slot for a duration on a day."""
        duration = args["duration_minutes"]
        if not isinstance(duration, int) or isinstance(duration, bool):
            raise ValueError("duration_minutes must be an integer")
        day = _parse_day(args["day"])
        slot = self.scheduler.next_available_slot(
            duration,
            day,
            args.get("window_start", "07:00"),
            args.get("window_end", "22:00"),
        )
        return {"day": day.isoformat(), "slot": slot}

    def _tool_build_daily_plan(self, args: dict) -> dict:
        """Build the daily plan for a day."""
        day = _parse_day(args["day"])
        plan = self.scheduler.build_daily_plan(day)
        return {
            "day": day.isoformat(),
            "budget_minutes": self.sandbox.available_minutes,
            "minutes_used": plan.minutes_used,
            "planned": [
                {"task_id": t.task_id, "description": t.description, "time": t.time, "reason": plan.reasons[t.task_id]}
                for t in self.scheduler.sort_by_time(plan.planned)
            ],
            "skipped": [
                {"task_id": t.task_id, "description": t.description, "time": t.time, "reason": plan.reasons[t.task_id]}
                for t in plan.skipped
            ],
            "warnings": plan.warnings,
        }

    # --- propose tools (sandbox only; recorded as changes) ---

    def _propose(self, change_data: dict) -> dict:
        """Parse, apply to the sandbox, and record one change; return a short result with the day's conflicts."""
        change = parse_change(change_data)
        message = apply_change(self.scheduler, change, self.today)
        self.changes.append(change)
        task = self.sandbox.find_task(change.task_id)[1]
        day_tasks = [t for t in self.sandbox.get_all_tasks() if t.due_date == task.due_date]
        return {
            "ok": True,
            "task_id": change.task_id,
            "result": message,
            "conflicts_now": self.scheduler.detect_conflicts(day_tasks),
        }

    def _tool_propose_move_task(self, args: dict) -> dict:
        """Propose a move."""
        return self._propose({"kind": "move", **args})

    def _tool_propose_add_task(self, args: dict) -> dict:
        """Propose an add; the tool (not the model) chooses the new task id."""
        return self._propose({"kind": "add", "task_id": uuid.uuid4().hex[:8], **args})

    def _tool_propose_complete_task(self, args: dict) -> dict:
        """Propose completing a task."""
        return self._propose({"kind": "complete", **args})
