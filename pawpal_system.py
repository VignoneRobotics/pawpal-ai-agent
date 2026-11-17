"""PawPal+ logic layer: Task, Pet, Owner, Scheduler (Phase 1 skeleton)."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path

PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}
FREQUENCY_DAYS = {"daily": 1, "weekly": 7}


def _to_minutes(time_str: str) -> int:
    """Convert a zero-padded "HH:MM" string to minutes since midnight."""
    hours, minutes = time_str.split(":")
    return int(hours) * 60 + int(minutes)


def _to_hhmm(minutes: int) -> str:
    """Convert minutes since midnight to a zero-padded "HH:MM" string."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _describe_range(task: Task) -> str:
    """Return "Description HH:MM-HH:MM" for a task's start and end time."""
    end = _to_minutes(task.time) + task.duration_minutes
    return f"{task.description} {task.time}-{end // 60:02d}:{end % 60:02d}"


def _time_range(task: Task) -> tuple[int, int]:
    """Return a task's (start, end) in minutes since midnight."""
    start = _to_minutes(task.time)
    return start, start + task.duration_minutes


def _parse_windows(windows: list[str]) -> list[tuple[int, int]]:
    """Parse "HH:MM-HH:MM" windows into sorted (start, end) minutes; raise ValueError if invalid or overlapping."""
    ranges: list[tuple[int, int]] = []
    for window in windows:
        match = re.fullmatch(r"([0-9]{2}):([0-9]{2})-([0-9]{2}):([0-9]{2})", str(window))
        if match is None:
            raise ValueError(f"Invalid window {window!r}: use HH:MM-HH:MM")
        sh, sm, eh, em = (int(part) for part in match.groups())
        start, end = sh * 60 + sm, eh * 60 + em
        if sh > 23 or sm > 59 or em > 59 or end > 24 * 60:
            raise ValueError(f"Invalid window {window!r}: times must be 00:00-24:00")
        if start >= end:
            raise ValueError(f"Invalid window {window!r}: start must be before end")
        ranges.append((start, end))
    ranges.sort()
    for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:]):
        if next_start < prev_end:
            raise ValueError(f"Availability windows must not overlap: {windows}")
    return ranges


def _first_gap(busy: list[tuple[int, int]], start: int, end: int, duration: int) -> int | None:
    """Return the first minute in [start, end] where duration fits between sorted busy ranges, or None."""
    cursor = start
    for busy_start, busy_end in busy:
        if cursor + duration <= min(busy_start, end):  # back-to-back is allowed
            return cursor
        cursor = max(cursor, busy_end)
    if cursor + duration <= end:
        return cursor
    return None


def _covered_minutes(start: int, end: int, intervals: list[tuple[int, int]]) -> int:
    """Return how many minutes of [start, end] fall inside the union of the given intervals."""
    clipped = sorted((max(s, start), min(e, end)) for s, e in intervals if s < end and start < e)
    covered = 0
    cursor = start
    for s, e in clipped:
        s = max(s, cursor)  # skip minutes already counted
        if e > s:
            covered += e - s
            cursor = e
    return covered


def _overlaps(a: Task, b: Task) -> bool:
    """Return True if two tasks are due the same day and their time ranges overlap (back-to-back is not)."""
    if a.due_date != b.due_date:
        return False
    a_start, a_end = _time_range(a)
    b_start, b_end = _time_range(b)
    return a_start < b_end and b_start < a_end


def _conflict_warning(first: Task, second: Task) -> str:
    """Return the warning string for two overlapping tasks, earlier one first."""
    if first.pet_name == second.pet_name:
        label = f"same pet: {first.pet_name}"
    else:
        label = f"cross-pet: {first.pet_name}, {second.pet_name}"
    return f"Conflict ({label}): {_describe_range(first)} overlaps {_describe_range(second)}"


def _validate_hhmm(value: object) -> None:
    """Raise ValueError unless value is a zero-padded "HH:MM" string within 00:00-23:59."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{2}:[0-9]{2}", value):
        raise ValueError(f"Invalid time {value!r}: use zero-padded HH:MM (00:00-23:59)")
    hours, minutes = int(value[:2]), int(value[3:])
    if hours > 23 or minutes > 59:
        raise ValueError(f"Invalid time {value!r}: hour must be 00-23 and minute 00-59")


@dataclass(frozen=True)
class Medication:
    """Prescription details for a medication task; dose_text is never changed by the agent."""

    drug: str
    dose_text: str
    times: tuple[str, ...]  # the vet's prescribed "HH:MM" times, e.g. ("08:00", "20:00")
    vet_name: str = ""
    vet_clinic: str = ""
    vet_phone: str = ""
    vet_consent: bool = False
    attested: bool = False
    label_verified: bool = False

    def __post_init__(self) -> None:
        """Reject an empty drug or dose, bad or duplicate times, or vet details stored without consent."""
        if not self.drug.strip():
            raise ValueError("Medication drug name must not be empty")
        if not self.dose_text.strip():
            raise ValueError("Medication dose must not be empty")
        if not isinstance(self.times, tuple) or not self.times:
            raise ValueError("Medication times must be a non-empty tuple of HH:MM strings")
        for time_str in self.times:
            _validate_hhmm(time_str)
        if len(set(self.times)) != len(self.times):
            raise ValueError(f"Medication times must not repeat: {self.times}")
        if (self.vet_name or self.vet_clinic or self.vet_phone) and not self.vet_consent:
            raise ValueError("Vet contact details need the owner's consent before they are stored")


@dataclass
class Task:
    """A single pet-care task."""

    description: str
    time: str
    duration_minutes: int
    priority: str = "medium"
    frequency: str = "once"
    due_date: date = field(default_factory=date.today)
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    completed: bool = False
    pet_name: str = ""
    combinable: bool = False
    medication: Medication | None = None

    @classmethod
    def medication_dose(
        cls,
        medication: Medication,
        time: str,
        duration_minutes: int,
        due_date: date,
        priority: str = "high",
        frequency: str = "daily",
        description: str = "",
    ) -> Task:
        """Create one dose task at a prescribed time; raise ValueError if time is not in medication.times."""
        if time not in medication.times:
            raise ValueError(
                f"Dose time {time!r} is not one of the prescribed times {', '.join(medication.times)}"
            )
        return cls(
            description or medication.drug,
            time,
            duration_minutes,
            priority,
            frequency,
            due_date=due_date,
            medication=medication,
        )

    def __post_init__(self) -> None:
        """Reject a malformed time, unknown priority/frequency, bad duration, or a task past midnight."""
        _validate_hhmm(self.time)
        if self.priority not in PRIORITY_RANK:
            raise ValueError(
                f"Unknown priority {self.priority!r}: use {', '.join(PRIORITY_RANK)}"
            )
        if self.frequency != "once" and self.frequency not in FREQUENCY_DAYS:
            raise ValueError(
                f"Unknown frequency {self.frequency!r}: use once, {', '.join(FREQUENCY_DAYS)}"
            )
        if self.duration_minutes <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration_minutes}")
        if _to_minutes(self.time) + self.duration_minutes > 24 * 60:
            raise ValueError(
                f"Task {self.description!r} at {self.time} for {self.duration_minutes} min "
                "runs past midnight"
            )

    def mark_complete(self) -> None:
        """Mark this task as completed."""
        self.completed = True

    def next_occurrence(self, today: date | None = None) -> Task | None:
        """Return the next recurrence (strictly after today, if given), or None for "once"."""
        if self.frequency not in FREQUENCY_DAYS:
            return None
        step = timedelta(days=FREQUENCY_DAYS[self.frequency])
        next_due = self.due_date + step
        if today is not None:
            while next_due <= today:  # completed late: skip the missed occurrences
                next_due += step
        return replace(self, task_id=uuid.uuid4().hex[:8], completed=False, due_date=next_due)


@dataclass
class Pet:
    """A pet and the tasks that belong to it."""

    name: str
    species: str
    age: int = 0
    tasks: list[Task] = field(default_factory=list)

    def add_task(self, task: Task) -> None:
        """Add a task to this pet and stamp it with the pet's name."""
        task.pet_name = self.name
        self.tasks.append(task)

    def remove_task(self, task_id: str) -> None:
        """Remove the task with the given id from this pet."""
        self.tasks = [t for t in self.tasks if t.task_id != task_id]

    def get_tasks(self) -> list[Task]:
        """Return all tasks belonging to this pet."""
        return list(self.tasks)


@dataclass(frozen=True)
class OverlapOverride:
    """Owner said "do both" for this pair of overlapping tasks on this day only."""

    task_a: str  # normalized so task_a < task_b (order-insensitive pair)
    task_b: str
    day: date


def _make_override(task_id_a: str, task_id_b: str, day: date) -> OverlapOverride:
    """Return an OverlapOverride with the two task ids in sorted order."""
    first, second = sorted((task_id_a, task_id_b))
    return OverlapOverride(task_a=first, task_b=second, day=day)


@dataclass
class Owner:
    """A pet owner with a daily time budget and a list of pets."""

    name: str
    available_minutes: int = 480
    pets: list[Pet] = field(default_factory=list)
    availability_windows: list[str] = field(default_factory=list)  # "HH:MM-HH:MM"; empty = all day
    overlap_overrides: list[OverlapOverride] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject malformed or overlapping availability windows."""
        _parse_windows(self.availability_windows)

    def window_ranges(self) -> list[tuple[int, int]]:
        """Return availability windows as sorted (start, end) minutes; the whole day if none are set."""
        return _parse_windows(self.availability_windows) or [(0, 24 * 60)]

    def add_pet(self, pet: Pet) -> None:
        """Add a pet to this owner, rejecting a duplicate name."""
        if self.get_pet(pet.name) is not None:
            raise ValueError(f"Owner already has a pet named {pet.name!r}")
        self.pets.append(pet)

    def get_pet(self, name: str) -> Pet | None:
        """Return the pet with the given name (case-insensitive), or None if not found."""
        for pet in self.pets:
            if pet.name.lower() == name.lower():
                return pet
        return None

    def get_all_tasks(self) -> list[Task]:
        """Return every task across every pet."""
        return [task for pet in self.pets for task in pet.get_tasks()]

    def find_task(self, task_id: str) -> tuple[Pet, Task] | None:
        """Return the (pet, task) pair for the given task id, or None if not found."""
        for pet in self.pets:
            for task in pet.tasks:
                if task.task_id == task_id:
                    return pet, task
        return None

    def allow_overlap(self, task_id_a: str, task_id_b: str, day: date) -> None:
        """Record that two tasks may overlap on this day ("do both"); raise ValueError on bad ids or day."""
        if task_id_a == task_id_b:
            raise ValueError("A task cannot overlap with itself")
        for task_id in (task_id_a, task_id_b):
            found = self.find_task(task_id)
            if found is None:
                raise ValueError(f"Unknown task id {task_id!r}")
            if found[1].due_date != day:
                raise ValueError(f"Task {task_id!r} is not due on {day.isoformat()}")
        override = _make_override(task_id_a, task_id_b, day)
        if override not in self.overlap_overrides:
            self.overlap_overrides.append(override)

    def is_overlap_allowed(self, task_id_a: str, task_id_b: str, day: date) -> bool:
        """Return True if the owner allowed these two tasks to overlap on this day."""
        return _make_override(task_id_a, task_id_b, day) in self.overlap_overrides

    def save_to_json(self, path: str | Path) -> None:
        """Write this owner, their pets, and all tasks to a JSON file (dates as ISO strings)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=lambda value: value.isoformat())

    @classmethod
    def load_from_json(cls, path: str | Path) -> Owner:
        """Rebuild an owner from a JSON file, re-running Task validation; raise ValueError on bad data."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        try:
            owner = cls(
                name=data["name"],
                available_minutes=data["available_minutes"],
                availability_windows=data.get("availability_windows", []),  # absent in old files
                overlap_overrides=[
                    OverlapOverride(o["task_a"], o["task_b"], date.fromisoformat(o["day"]))
                    for o in data.get("overlap_overrides", [])
                ],
            )
            for pet_data in data["pets"]:
                pet = Pet(name=pet_data["name"], species=pet_data["species"], age=pet_data["age"])
                for task_data in pet_data["tasks"]:
                    due = date.fromisoformat(task_data["due_date"])
                    med_data = task_data.get("medication")  # absent in old files
                    medication = (
                        Medication(**{**med_data, "times": tuple(med_data["times"])})
                        if med_data is not None
                        else None
                    )
                    pet.add_task(Task(**{**task_data, "due_date": due, "medication": medication}))
                owner.add_pet(pet)
        except (KeyError, TypeError) as err:
            raise ValueError(f"Invalid data in {path}: {err!r}") from err
        return owner


@dataclass
class DailyPlan:
    """Result of build_daily_plan: what was planned, what was skipped, and why."""

    day: date
    planned: list[Task] = field(default_factory=list)  # selection order (priority, time, description)
    skipped: list[Task] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)  # task_id -> reason string
    warnings: list[str] = field(default_factory=list)  # overlap skips and owner-allowed overlaps
    minutes_used: int = 0  # combinable overlap counted once


@dataclass
class Scheduler:
    """Builds sorted, filtered, conflict-checked plans from an owner's tasks."""

    owner: Owner

    def sort_by_time(self, tasks: list[Task] | None = None) -> list[Task]:
        """Return tasks sorted by due date, then "HH:MM" time, priority, and description."""
        if tasks is None:
            tasks = self.owner.get_all_tasks()
        return sorted(
            tasks,
            key=lambda t: (t.due_date, t.time, PRIORITY_RANK[t.priority], t.description),
        )

    def sort_by_priority(self, tasks: list[Task] | None = None) -> list[Task]:
        """Return tasks sorted by due date, then priority, "HH:MM" time, and description."""
        if tasks is None:
            tasks = self.owner.get_all_tasks()
        return sorted(
            tasks,
            key=lambda t: (t.due_date, PRIORITY_RANK[t.priority], t.time, t.description),
        )

    def filter_tasks(
        self,
        tasks: list[Task] | None = None,
        pet_name: str | None = None,
        completed: bool | None = None,
    ) -> list[Task]:
        """Return tasks matching the given pet name and completion status."""
        if tasks is None:
            tasks = self.owner.get_all_tasks()
        result = list(tasks)
        if pet_name is not None:
            result = [t for t in result if t.pet_name.lower() == pet_name.lower()]
        if completed is not None:
            result = [t for t in result if t.completed == completed]
        return result

    def find_conflict_pairs(self, tasks: list[Task] | None = None) -> list[tuple[Task, Task]]:
        """Return (earlier, later) pairs of pending same-day overlapping tasks, unless both are combinable."""
        if tasks is None:
            tasks = self.owner.get_all_tasks()
        pending = [t for t in tasks if not t.completed]

        pairs: list[tuple[Task, Task]] = []
        for a, b in combinations(pending, 2):  # each pair is checked once
            if a.combinable and b.combinable:
                continue  # e.g. feeding both pets at once is not a conflict
            if _overlaps(a, b):
                pairs.append((a, b) if _to_minutes(a.time) <= _to_minutes(b.time) else (b, a))
        return pairs

    def detect_conflicts(self, tasks: list[Task] | None = None) -> list[str]:
        """Return warning strings for tasks with overlapping time ranges."""
        return [_conflict_warning(first, second) for first, second in self.find_conflict_pairs(tasks)]

    def next_available_slot(
        self,
        duration_minutes: int,
        day: date,
        window_start: str = "07:00",
        window_end: str = "22:00",
    ) -> str | None:
        """Return the first "HH:MM" where a task fits on the day, inside the window and the owner's availability, or None."""
        if duration_minutes <= 0:
            raise ValueError(f"Duration must be positive, got {duration_minutes}")
        busy = sorted(
            _time_range(t)
            for t in self.owner.get_all_tasks()
            if t.due_date == day and not t.completed
        )
        search_start = _to_minutes(window_start)
        search_end = _to_minutes(window_end)
        for owner_start, owner_end in self.owner.window_ranges():
            start = max(owner_start, search_start)
            end = min(owner_end, search_end)
            if start >= end:
                continue  # this owner window is outside the search window
            slot = _first_gap(busy, start, end, duration_minutes)
            if slot is not None:
                return _to_hhmm(slot)
        return None

    def overdue_tasks(self, today: date, tasks: list[Task] | None = None) -> list[Task]:
        """Return pending tasks due before today, oldest first."""
        if tasks is None:
            tasks = self.owner.get_all_tasks()
        return self.sort_by_time([t for t in tasks if not t.completed and t.due_date < today])

    def move_task(self, task_id: str, new_time: str, new_date: date | None = None) -> Task:
        """Move a pending task to a new time (and optionally date) after full validation; raise ValueError if invalid."""
        found = self.owner.find_task(task_id)
        if found is None:
            raise ValueError(f"Unknown task id {task_id!r}")
        _, task = found
        if task.completed:
            raise ValueError(f"Task {task.description!r} is completed and cannot be moved")
        target_date = new_date if new_date is not None else task.due_date
        replace(task, time=new_time, due_date=target_date)  # throwaway copy: runs Task validation
        task.time = new_time
        task.due_date = target_date
        return task

    def mark_task_complete(self, task_id: str, today: date | None = None) -> Task | None:
        """Complete a task, add its next occurrence (after today, if given) if recurring, and return it or None."""
        found = self.owner.find_task(task_id)
        if found is None:
            return None
        pet, task = found
        if task.completed:
            return None
        task.mark_complete()
        next_task = task.next_occurrence(today)
        if next_task is not None:
            pet.add_task(next_task)
        return next_task

    def build_daily_plan(self, day: date) -> DailyPlan:
        """Plan the day's pending tasks by priority; skip (with a reason) tasks outside windows, overlapping, or over budget."""
        due = [
            t for t in self.owner.get_all_tasks()
            if t.due_date == day and not t.completed
        ]
        due.sort(key=lambda t: (PRIORITY_RANK[t.priority], _to_minutes(t.time), t.description))

        plan = DailyPlan(day=day)
        windows = self.owner.window_ranges()
        remaining = self.owner.available_minutes
        for task in due:
            start, end = _time_range(task)

            # 1) The whole task must fit inside one availability window.
            if not any(w_start <= start and end <= w_end for w_start, w_end in windows):
                plan.skipped.append(task)
                plan.reasons[task.task_id] = "skipped: outside windows"
                continue

            # 2) It must not overlap an already-planned (higher priority) task, unless both are
            #    combinable or the owner allowed this pair on this day.
            blockers = [
                p for p in plan.planned
                if _overlaps(p, task) and not (p.combinable and task.combinable)
            ]
            not_allowed = [
                p for p in blockers
                if not self.owner.is_overlap_allowed(p.task_id, task.task_id, day)
            ]
            if not_allowed:
                first = not_allowed[0]
                plan.skipped.append(task)
                plan.reasons[task.task_id] = f"skipped: overlaps {first.description}"
                plan.warnings.append(
                    f"Skipped (overlap): {_describe_range(task)} overlaps "
                    f"{_describe_range(first)} (planned first by priority/time)"
                )
                continue

            # 3) Budget: minutes shared with combinable partners are counted once.
            shared = [
                _time_range(p) for p in plan.planned
                if p.combinable and task.combinable and _overlaps(p, task)
            ]
            cost = task.duration_minutes - _covered_minutes(start, end, shared)
            if cost > remaining:
                plan.skipped.append(task)
                plan.reasons[task.task_id] = "skipped: budget"
                continue

            plan.planned.append(task)
            remaining -= cost
            plan.minutes_used += cost
            if blockers:
                plan.reasons[task.task_id] = "planned: overlap allowed by owner"
                for other in blockers:
                    first, second = sorted((other, task), key=lambda t: _to_minutes(t.time))
                    plan.warnings.append(_conflict_warning(first, second) + " (allowed by owner)")
            else:
                plan.reasons[task.task_id] = "planned"
        return plan

    def explain_plan(self, plan: DailyPlan) -> list[str]:
        """Return one human-readable line per planned and skipped task, using the plan's stored reasons."""
        if not plan.planned and not plan.skipped:
            return ["No tasks due."]

        budget = self.owner.available_minutes
        lines: list[str] = []
        for t in plan.planned:
            head = (
                f"Planned: {t.description} ({t.pet_name}) at {t.time}, "
                f"{t.duration_minutes} min, {t.priority} priority - "
            )
            if plan.reasons[t.task_id] == "planned: overlap allowed by owner":
                partners = [
                    p.description for p in plan.planned
                    if p is not t and _overlaps(p, t)
                    and self.owner.is_overlap_allowed(p.task_id, t.task_id, plan.day)
                ]
                lines.append(head + f"overlap with {', '.join(partners)} allowed by owner.")
            else:
                lines.append(head + f"fits in the {budget}-minute budget.")

        for t in plan.skipped:
            head = f"Skipped: {t.description} ({t.pet_name}) at {t.time}"
            reason = plan.reasons[t.task_id]
            if reason == "skipped: budget":
                lines.append(
                    f"{head} needs {t.duration_minutes} min; did not fit in the "
                    f"{budget}-minute budget (planned tasks use {plan.minutes_used})."
                )
            elif reason == "skipped: outside windows":
                windows = ", ".join(self.owner.availability_windows)
                lines.append(f"{head} - outside the owner's availability windows ({windows}).")
            else:
                other = reason.removeprefix("skipped: overlaps ")
                lines.append(f"{head} - overlaps {other}, which was planned first.")
        return lines
