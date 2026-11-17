import json
from dataclasses import FrozenInstanceError
from datetime import date, timedelta

import pytest

from pawpal_system import Medication, OverlapOverride, Owner, Pet, Scheduler, Task

DAY = date(2026, 1, 5)
NEXT_DAY = DAY + timedelta(days=1)


def make_scheduler(budget: int = 480) -> tuple[Scheduler, Pet, Pet]:
    """Return a Scheduler for an owner with two empty pets, Mochi and Bean."""
    owner = Owner("Jordan", available_minutes=budget)
    mochi = Pet("Mochi", "dog")
    bean = Pet("Bean", "cat")
    owner.add_pet(mochi)
    owner.add_pet(bean)
    return Scheduler(owner), mochi, bean


def add(pet, description, time, duration, priority="medium", frequency="once", due=DAY) -> Task:
    """Create a task with an explicit due date, add it to the pet, and return it."""
    task = Task(description, time, duration, priority, frequency, due_date=due)
    pet.add_task(task)
    return task


def test_mark_complete_sets_completed_true():
    """mark_complete() flips a new task from not completed to completed."""
    task = Task("Morning walk", "07:30", 30, due_date=DAY)
    assert task.completed is False

    task.mark_complete()

    assert task.completed is True


def test_add_task_increases_count_and_stamps_pet_name():
    """Pet.add_task() adds the task and sets its pet_name to the pet's name."""
    pet = Pet("Mochi", "dog")
    task = Task("Brush coat", "18:00", 20, due_date=DAY)
    assert len(pet.get_tasks()) == 0
    assert task.pet_name == ""

    pet.add_task(task)

    assert len(pet.get_tasks()) == 1
    assert task.pet_name == "Mochi"


# --- Validation ---


@pytest.mark.parametrize("bad_time", ["7:30", "25:00", "24:00", "12:60", "0730", "07:3", "ab:cd", ""])
def test_task_rejects_bad_time(bad_time):
    """A time that is not zero-padded HH:MM within 00:00-23:59 raises ValueError."""
    with pytest.raises(ValueError, match="time"):
        Task("Walk", bad_time, 30, due_date=DAY)


def test_task_rejects_unknown_priority():
    """An unknown priority string raises ValueError."""
    with pytest.raises(ValueError, match="priority"):
        Task("Walk", "08:00", 30, priority="urgent", due_date=DAY)


def test_task_rejects_unknown_frequency():
    """An unknown frequency string raises ValueError."""
    with pytest.raises(ValueError, match="frequency"):
        Task("Walk", "08:00", 30, frequency="monthly", due_date=DAY)


@pytest.mark.parametrize("bad_duration", [0, -5])
def test_task_rejects_non_positive_duration(bad_duration):
    """A duration of zero or less raises ValueError."""
    with pytest.raises(ValueError, match="uration"):
        Task("Walk", "08:00", bad_duration, due_date=DAY)


def test_task_rejects_running_past_midnight():
    """A task whose start + duration passes 24:00 raises ValueError."""
    with pytest.raises(ValueError, match="midnight"):
        Task("Late walk", "23:30", 45, due_date=DAY)


def test_task_allows_ending_exactly_at_midnight():
    """A task that ends exactly at 24:00 is valid."""
    task = Task("Late walk", "23:30", 30, due_date=DAY)
    assert task.time == "23:30"


def test_task_accepts_normal_values():
    """A typical task constructs without error."""
    task = Task("Walk", "08:00", 30, priority="high", frequency="daily", due_date=DAY)
    assert task.priority == "high"


# --- Owner / Pet ---


def test_add_pet_rejects_duplicate_name_case_insensitive():
    """Adding a pet whose name matches an existing pet (any case) raises ValueError."""
    scheduler, _, _ = make_scheduler()
    with pytest.raises(ValueError):
        scheduler.owner.add_pet(Pet("mochi", "cat"))


def test_get_pet_is_case_insensitive():
    """get_pet finds a pet regardless of case and returns None when missing."""
    scheduler, mochi, _ = make_scheduler()
    assert scheduler.owner.get_pet("MOCHI") is mochi
    assert scheduler.owner.get_pet("Rex") is None


def test_get_all_tasks_spans_every_pet():
    """get_all_tasks returns tasks from all pets."""
    scheduler, mochi, bean = make_scheduler()
    add(mochi, "Walk", "08:00", 30)
    add(bean, "Feed", "09:00", 10)
    assert {t.description for t in scheduler.owner.get_all_tasks()} == {"Walk", "Feed"}


def test_remove_task_removes_only_that_task():
    """remove_task deletes the task with the given id and keeps the others."""
    _, mochi, _ = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 30)
    bath = add(mochi, "Bath", "10:00", 30)
    mochi.remove_task(walk.task_id)
    assert mochi.get_tasks() == [bath]


def test_find_task_returns_pet_and_task_or_none():
    """find_task returns the owning pet with the task, or None for an unknown id."""
    scheduler, _, bean = make_scheduler()
    feed = add(bean, "Feed", "09:00", 10)
    assert scheduler.owner.find_task(feed.task_id) == (bean, feed)
    assert scheduler.owner.find_task("nope") is None


# --- Sorting ---


def test_sort_by_time_orders_out_of_order_tasks():
    """Tasks added out of order come back in ascending time order."""
    scheduler, mochi, _ = make_scheduler()
    add(mochi, "C", "09:00", 10)
    add(mochi, "A", "07:30", 10)
    add(mochi, "B", "08:00", 10)
    assert [t.time for t in scheduler.sort_by_time()] == ["07:30", "08:00", "09:00"]


def test_sort_by_time_breaks_ties_by_priority():
    """Two tasks at the same time sort high priority before low."""
    scheduler, mochi, bean = make_scheduler()
    add(mochi, "Low one", "08:00", 10, priority="low")
    add(bean, "High one", "08:00", 10, priority="high")
    assert [t.description for t in scheduler.sort_by_time()] == ["High one", "Low one"]


def test_sort_by_time_puts_next_day_after_today():
    """A next-day 07:00 task sorts after the same-day 08:00 task."""
    scheduler, mochi, _ = make_scheduler()
    add(mochi, "Tomorrow early", "07:00", 10, due=NEXT_DAY)
    add(mochi, "Today later", "08:00", 10, due=DAY)
    assert [t.description for t in scheduler.sort_by_time()] == ["Today later", "Tomorrow early"]


def test_sort_by_time_with_no_tasks_returns_empty_list():
    """Sorting an owner with no tasks returns an empty list."""
    scheduler, _, _ = make_scheduler()
    assert scheduler.sort_by_time() == []


def test_sort_by_priority_puts_high_before_low_at_later_time():
    """A high-priority task at 09:00 sorts before a low-priority task at 07:00 on the same day."""
    scheduler, mochi, _ = make_scheduler()
    add(mochi, "Low early", "07:00", 10, priority="low")
    add(mochi, "High late", "09:00", 10, priority="high")
    assert [t.description for t in scheduler.sort_by_priority()] == ["High late", "Low early"]


def test_sort_by_priority_equal_priority_falls_back_to_time():
    """Tasks with the same priority sort by time."""
    scheduler, mochi, _ = make_scheduler()
    add(mochi, "Later", "10:00", 10, priority="medium")
    add(mochi, "Earlier", "08:00", 10, priority="medium")
    assert [t.description for t in scheduler.sort_by_priority()] == ["Earlier", "Later"]


def test_sort_by_priority_next_day_high_sorts_after_today():
    """A next-day high-priority task still sorts after today's low-priority task."""
    scheduler, mochi, _ = make_scheduler()
    add(mochi, "Tomorrow high", "07:00", 10, priority="high", due=NEXT_DAY)
    add(mochi, "Today low", "08:00", 10, priority="low", due=DAY)
    assert [t.description for t in scheduler.sort_by_priority()] == ["Today low", "Tomorrow high"]


def test_sort_and_filter_compose():
    """sort_by_time accepts the output of filter_tasks."""
    scheduler, mochi, bean = make_scheduler()
    add(mochi, "Late", "10:00", 10)
    add(mochi, "Early", "07:00", 10)
    add(bean, "Other", "08:00", 10)
    result = scheduler.sort_by_time(scheduler.filter_tasks(pet_name="Mochi"))
    assert [t.description for t in result] == ["Early", "Late"]


# --- Filtering ---


def test_filter_by_pet_name_is_case_insensitive():
    """filter_tasks matches the pet name regardless of case."""
    scheduler, mochi, bean = make_scheduler()
    add(mochi, "Walk", "08:00", 10)
    add(bean, "Feed", "09:00", 10)
    assert [t.description for t in scheduler.filter_tasks(pet_name="mOcHi")] == ["Walk"]


def test_filter_by_completed_status():
    """filter_tasks separates completed from pending tasks."""
    scheduler, mochi, _ = make_scheduler()
    done = add(mochi, "Done", "08:00", 10)
    add(mochi, "Pending", "09:00", 10)
    done.mark_complete()
    assert [t.description for t in scheduler.filter_tasks(completed=True)] == ["Done"]
    assert [t.description for t in scheduler.filter_tasks(completed=False)] == ["Pending"]


def test_filters_combine_and_none_means_no_filter():
    """Pet and completed filters combine; None leaves tasks unfiltered."""
    scheduler, mochi, bean = make_scheduler()
    add(mochi, "Mochi pending", "08:00", 10)
    done = add(mochi, "Mochi done", "09:00", 10)
    add(bean, "Bean pending", "10:00", 10)
    done.mark_complete()
    both = scheduler.filter_tasks(pet_name="Mochi", completed=False)
    assert [t.description for t in both] == ["Mochi pending"]
    assert len(scheduler.filter_tasks()) == 3


# --- Recurrence ---


def test_completing_daily_task_creates_next_day_copy():
    """Completing a daily task keeps it and adds an equal task due the next day."""
    scheduler, mochi, _ = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 30, priority="high", frequency="daily")

    new_task = scheduler.mark_task_complete(walk.task_id)

    assert walk.completed is True
    assert new_task is not None
    assert new_task.due_date == NEXT_DAY
    assert new_task.completed is False
    assert new_task.task_id != walk.task_id
    assert (new_task.description, new_task.time, new_task.duration_minutes) == ("Walk", "08:00", 30)
    assert new_task.pet_name == "Mochi"
    assert len(mochi.get_tasks()) == 2


def test_completing_weekly_task_creates_copy_seven_days_later():
    """Completing a weekly task adds a copy due 7 days later."""
    scheduler, mochi, _ = make_scheduler()
    bath = add(mochi, "Bath", "10:00", 30, frequency="weekly")
    new_task = scheduler.mark_task_complete(bath.task_id)
    assert new_task.due_date == DAY + timedelta(days=7)


def test_late_daily_task_skips_to_day_after_completion():
    """A daily task completed 4 days late gets a copy due the day after today."""
    task = Task("Walk", "08:00", 30, frequency="daily", due_date=DAY)
    today = DAY + timedelta(days=4)
    assert task.next_occurrence(today).due_date == today + timedelta(days=1)


def test_late_weekly_task_keeps_its_weekday():
    """A weekly task completed 3 weeks late skips ahead in whole weeks, on the same weekday."""
    task = Task("Bath", "10:00", 30, frequency="weekly", due_date=DAY)
    today = DAY + timedelta(days=17)
    next_task = task.next_occurrence(today)
    assert next_task.due_date == DAY + timedelta(days=21)
    assert next_task.due_date.weekday() == DAY.weekday()


def test_completing_exactly_on_cadence_date_moves_one_interval():
    """Completing on the due date itself moves the copy one interval ahead (daily and weekly)."""
    daily = Task("Walk", "08:00", 30, frequency="daily", due_date=DAY)
    weekly = Task("Bath", "10:00", 30, frequency="weekly", due_date=DAY)
    assert daily.next_occurrence(DAY).due_date == DAY + timedelta(days=1)
    assert weekly.next_occurrence(DAY).due_date == DAY + timedelta(days=7)


def test_next_candidate_equal_to_today_moves_to_day_after():
    """If the plain next due date equals today, the copy is due one interval after today."""
    daily = Task("Walk", "08:00", 30, frequency="daily", due_date=DAY)
    weekly = Task("Bath", "10:00", 30, frequency="weekly", due_date=DAY)
    assert daily.next_occurrence(DAY + timedelta(days=1)).due_date == DAY + timedelta(days=2)
    assert weekly.next_occurrence(DAY + timedelta(days=7)).due_date == DAY + timedelta(days=14)


def test_early_or_on_time_completion_is_unchanged():
    """With no today, or today on or before the due date, the copy is due_date + one interval."""
    task = Task("Walk", "08:00", 30, frequency="daily", due_date=DAY)
    assert task.next_occurrence().due_date == NEXT_DAY
    assert task.next_occurrence(DAY - timedelta(days=3)).due_date == NEXT_DAY
    assert task.next_occurrence(DAY).due_date == NEXT_DAY


def test_once_task_returns_none_even_when_late():
    """A one-off task has no next occurrence however late it is completed."""
    task = Task("Vet", "10:00", 60, due_date=DAY)
    assert task.next_occurrence(DAY + timedelta(days=30)) is None


def test_scheduler_adds_skipped_ahead_copy_to_same_pet():
    """mark_task_complete(today=...) adds the skipped-ahead copy, with a new id, to the owning pet."""
    scheduler, mochi, bean = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 30, frequency="daily")
    today = DAY + timedelta(days=10)

    new_task = scheduler.mark_task_complete(walk.task_id, today=today)

    assert new_task.due_date == today + timedelta(days=1)
    assert new_task.task_id != walk.task_id
    assert new_task.pet_name == "Mochi"
    assert walk.completed is True
    assert len(mochi.get_tasks()) == 2
    assert bean.get_tasks() == []


def test_completing_once_task_creates_no_copy():
    """Completing a one-off task returns None and adds nothing."""
    scheduler, mochi, _ = make_scheduler()
    vet = add(mochi, "Vet", "10:00", 60)
    assert scheduler.mark_task_complete(vet.task_id) is None
    assert vet.completed is True
    assert len(mochi.get_tasks()) == 1
    assert vet.next_occurrence() is None


def test_completing_twice_does_not_duplicate_next_occurrence():
    """A second completion of the same task returns None and adds no extra copy."""
    scheduler, mochi, _ = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 30, frequency="daily")
    scheduler.mark_task_complete(walk.task_id)
    assert scheduler.mark_task_complete(walk.task_id) is None
    assert len(mochi.get_tasks()) == 2


def test_completing_unknown_task_id_returns_none():
    """An unknown task id is ignored and returns None."""
    scheduler, _, _ = make_scheduler()
    assert scheduler.mark_task_complete("missing") is None


# --- Conflicts ---


def test_overlap_on_same_pet_gives_one_exact_warning():
    """Two overlapping tasks on one pet produce a single same-pet warning."""
    scheduler, mochi, _ = make_scheduler()
    add(mochi, "Walk", "08:00", 30)
    add(mochi, "Bath", "08:15", 30)
    assert scheduler.detect_conflicts() == [
        "Conflict (same pet: Mochi): Walk 08:00-08:30 overlaps Bath 08:15-08:45"
    ]


def test_overlap_across_pets_is_labelled_cross_pet():
    """Overlapping tasks on different pets get the cross-pet label, earlier task first."""
    scheduler, mochi, bean = make_scheduler()
    add(mochi, "Walk", "08:00", 30)
    add(bean, "Feed", "08:10", 10)
    assert scheduler.detect_conflicts() == [
        "Conflict (cross-pet: Mochi, Bean): Walk 08:00-08:30 overlaps Feed 08:10-08:20"
    ]


def test_same_start_time_is_a_conflict():
    """Two tasks starting at exactly the same time conflict."""
    scheduler, mochi, bean = make_scheduler()
    add(mochi, "Walk", "08:00", 30)
    add(bean, "Feed", "08:00", 10)
    assert len(scheduler.detect_conflicts()) == 1


def test_back_to_back_tasks_do_not_conflict():
    """A task starting exactly when another ends is not a conflict."""
    scheduler, mochi, _ = make_scheduler()
    add(mochi, "Walk", "08:00", 30)
    add(mochi, "Bath", "08:30", 30)
    assert scheduler.detect_conflicts() == []


def test_back_to_back_tasks_do_not_conflict_reverse_order():
    """Back-to-back tasks are not a conflict when the later task is added first."""
    scheduler, mochi, _ = make_scheduler()
    add(mochi, "Bath", "08:30", 30)
    add(mochi, "Walk", "08:00", 30)
    assert scheduler.detect_conflicts() == []


def test_same_time_on_different_days_does_not_conflict():
    """Tasks at the same time on different due dates do not conflict."""
    scheduler, mochi, _ = make_scheduler()
    add(mochi, "Walk", "08:00", 30, due=DAY)
    add(mochi, "Walk", "08:00", 30, due=NEXT_DAY)
    assert scheduler.detect_conflicts() == []


def test_completed_tasks_are_ignored_by_conflict_detection():
    """A completed task no longer conflicts with an overlapping pending task."""
    scheduler, mochi, _ = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 30)
    add(mochi, "Bath", "08:15", 30)
    walk.mark_complete()
    assert scheduler.detect_conflicts() == []


def test_three_mutually_overlapping_tasks_give_three_warnings():
    """Each overlapping pair is reported exactly once."""
    scheduler, mochi, _ = make_scheduler()
    add(mochi, "A", "08:00", 60)
    add(mochi, "B", "08:10", 60)
    add(mochi, "C", "08:20", 60)
    assert len(scheduler.detect_conflicts()) == 3


def test_detect_conflicts_with_no_tasks_returns_empty_list():
    """No tasks means no warnings."""
    scheduler, _, _ = make_scheduler()
    assert scheduler.detect_conflicts() == []


def test_find_conflict_pairs_returns_earlier_task_first():
    """Pairs come back (earlier, later) even if added in reverse; back-to-back and completed are excluded."""
    scheduler, mochi, bean = make_scheduler()
    bath = add(mochi, "Bath", "08:15", 30)  # added first but starts later
    walk = add(mochi, "Walk", "08:00", 30)
    add(bean, "Feed", "08:45", 15)  # back-to-back with Bath: no conflict
    done = add(bean, "Done", "08:10", 30)
    done.mark_complete()
    assert scheduler.find_conflict_pairs() == [(walk, bath)]
    assert len(scheduler.detect_conflicts()) == 1


# --- Combinable tasks ---


def test_two_combinable_overlapping_tasks_are_not_a_conflict():
    """Overlapping tasks that are both combinable do not conflict."""
    scheduler, mochi, bean = make_scheduler()
    add(mochi, "Feed Mochi", "08:00", 15).combinable = True
    add(bean, "Feed Bean", "08:05", 15).combinable = True
    assert scheduler.find_conflict_pairs() == []
    assert scheduler.detect_conflicts() == []


def test_one_combinable_task_still_conflicts():
    """If only one of two overlapping tasks is combinable, it is still a conflict."""
    scheduler, mochi, bean = make_scheduler()
    add(mochi, "Feed Mochi", "08:00", 15).combinable = True
    add(bean, "Vet call", "08:05", 15)
    assert len(scheduler.detect_conflicts()) == 1


def test_combinable_back_to_back_still_no_conflict():
    """Back-to-back combinable tasks stay conflict-free."""
    scheduler, mochi, _ = make_scheduler()
    add(mochi, "Feed", "08:00", 15).combinable = True
    add(mochi, "Water", "08:15", 15).combinable = True
    assert scheduler.detect_conflicts() == []


# --- Daily plan / budget ---


def test_plan_includes_only_pending_tasks_due_that_day():
    """Tasks due on other days and completed tasks are left out of the plan."""
    scheduler, mochi, _ = make_scheduler()
    today = add(mochi, "Today", "08:00", 10, due=DAY)
    add(mochi, "Tomorrow", "09:00", 10, due=NEXT_DAY)
    done = add(mochi, "Done", "10:00", 10, due=DAY)
    done.mark_complete()
    plan = scheduler.build_daily_plan(DAY)
    assert plan.planned == [today]
    assert plan.skipped == []


def test_plan_selects_by_priority_then_time_within_budget():
    """With room for one task, the high-priority task wins over an earlier low one."""
    scheduler, mochi, _ = make_scheduler(budget=30)
    add(mochi, "Early low", "07:00", 30, priority="low")
    high = add(mochi, "Later high", "09:00", 30, priority="high")
    plan = scheduler.build_daily_plan(DAY)
    assert plan.planned == [high]
    assert [t.description for t in plan.skipped] == ["Early low"]


def test_plan_total_never_exceeds_budget():
    """The planned minutes stay within the owner's available minutes."""
    scheduler, mochi, _ = make_scheduler(budget=50)
    for i, minutes in enumerate([30, 25, 20, 10]):
        add(mochi, f"T{i}", f"{8 + i:02d}:00", minutes)
    plan = scheduler.build_daily_plan(DAY)
    assert sum(t.duration_minutes for t in plan.planned) <= 50


def test_plan_skips_oversized_task_and_keeps_going():
    """A task too big for the remaining budget is skipped, and later smaller ones still fit."""
    scheduler, mochi, _ = make_scheduler(budget=60)
    add(mochi, "First", "08:00", 40, priority="high")
    big = add(mochi, "Too big", "09:00", 40, priority="medium")
    small = add(mochi, "Small", "10:00", 15, priority="low")
    plan = scheduler.build_daily_plan(DAY)
    assert small in plan.planned
    assert plan.skipped == [big]


def test_plan_accepts_task_that_exactly_fills_budget():
    """A task whose duration equals the remaining budget is planned."""
    scheduler, mochi, _ = make_scheduler(budget=30)
    task = add(mochi, "Exact", "08:00", 30)
    plan = scheduler.build_daily_plan(DAY)
    assert plan.planned == [task]
    assert plan.skipped == []


def test_plan_with_zero_budget_skips_everything():
    """A zero-minute budget plans nothing."""
    scheduler, mochi, _ = make_scheduler(budget=0)
    add(mochi, "Walk", "08:00", 10)
    plan = scheduler.build_daily_plan(DAY)
    assert plan.planned == []
    assert len(plan.skipped) == 1


def test_plan_skips_later_equal_priority_overlap():
    """Project 4 change: of two equal-priority overlapping tasks, the later one is skipped with a reason and warning."""
    scheduler, mochi, bean = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 30)
    feed = add(bean, "Feed", "08:10", 10)
    plan = scheduler.build_daily_plan(DAY)
    assert plan.planned == [walk]
    assert plan.skipped == [feed]
    assert plan.reasons[feed.task_id] == "skipped: overlaps Walk"
    assert plan.warnings == [
        "Skipped (overlap): Feed 08:10-08:20 overlaps Walk 08:00-08:30 (planned first by priority/time)"
    ]
    assert scheduler.detect_conflicts(plan.planned) == []


def test_plan_with_no_tasks_is_empty():
    """An owner with no tasks gets an empty plan."""
    scheduler, _, _ = make_scheduler()
    plan = scheduler.build_daily_plan(DAY)
    assert (plan.planned, plan.skipped) == ([], [])


# --- explain_plan ---


def test_explain_plan_with_nothing_due():
    """With no planned or skipped tasks the explanation says nothing is due."""
    scheduler, _, _ = make_scheduler()
    assert scheduler.explain_plan(scheduler.build_daily_plan(DAY)) == ["No tasks due."]


def test_explain_plan_has_one_line_per_task_and_mentions_budget():
    """Each planned and skipped task gets a line; skipped lines cite the budget."""
    scheduler, mochi, _ = make_scheduler(budget=30)
    add(mochi, "Fits", "08:00", 30, priority="high")
    add(mochi, "Too long", "09:00", 20, priority="low")
    plan = scheduler.build_daily_plan(DAY)
    lines = scheduler.explain_plan(plan)
    assert len(lines) == 2
    assert lines[0].startswith("Planned: Fits")
    assert lines[1].startswith("Skipped: Too long")
    assert "30-minute budget" in lines[1]


# --- Persistence ---


def saved_owner(tmp_path):
    """Save a two-pet owner (one completed daily task) and return (owner, path)."""
    scheduler, mochi, bean = make_scheduler(budget=90)
    walk = add(mochi, "Walk", "08:00", 30, priority="high", frequency="daily")
    add(bean, "Brush", "09:00", 15, priority="low", due=NEXT_DAY)
    scheduler.mark_task_complete(walk.task_id)  # keeps Walk as done and adds tomorrow's copy
    path = tmp_path / "data.json"
    scheduler.owner.save_to_json(path)
    return scheduler.owner, path


def test_round_trip_preserves_pets_tasks_ids_dates_and_completed(tmp_path):
    """Loading a saved owner gives back the same pets, tasks, ids, dates, and completed flags."""
    owner, path = saved_owner(tmp_path)
    loaded = Owner.load_from_json(path)

    assert (loaded.name, loaded.available_minutes) == ("Jordan", 90)
    assert [(p.name, p.species, p.age) for p in loaded.pets] == [(p.name, p.species, p.age) for p in owner.pets]
    original = owner.get_all_tasks()
    restored = loaded.get_all_tasks()
    assert [t.task_id for t in restored] == [t.task_id for t in original]
    assert [t.due_date for t in restored] == [t.due_date for t in original]
    assert all(isinstance(t.due_date, date) for t in restored)
    assert [t.completed for t in restored] == [True, False, False]
    assert [t.pet_name for t in restored] == [t.pet_name for t in original]
    assert loaded == owner


def test_saved_file_stores_dates_as_iso_strings(tmp_path):
    """The JSON file holds due dates as ISO strings, not objects."""
    _, path = saved_owner(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["pets"][0]["tasks"][0]["due_date"] == DAY.isoformat()


def test_load_missing_file_raises_file_not_found(tmp_path):
    """Loading a path that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        Owner.load_from_json(tmp_path / "missing.json")


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("priority", "urgent"),
        ("time", "7:30"),
        ("due_date", "not-a-date"),
        ("duration_minutes", 0),
        ("description", None),  # removed from the file below
    ],
)
def test_load_invalid_task_data_raises_value_error(tmp_path, field_name, value):
    """A bad or missing task field in the file raises ValueError."""
    _, path = saved_owner(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    task = data["pets"][0]["tasks"][0]
    if field_name == "description":
        del task[field_name]
    else:
        task[field_name] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        Owner.load_from_json(path)


# --- Next available slot ---


def test_next_slot_on_empty_day_is_window_start():
    """With nothing scheduled, the first slot is the start of the window."""
    scheduler, _, _ = make_scheduler()
    assert scheduler.next_available_slot(30, DAY) == "07:00"
    assert scheduler.next_available_slot(30, DAY, window_start="09:15") == "09:15"


def test_next_slot_finds_gap_between_tasks():
    """A task fits in the gap between two tasks; a longer one goes after the last task."""
    scheduler, mochi, bean = make_scheduler()
    add(mochi, "First", "07:00", 60)
    add(bean, "Second", "09:00", 60)
    assert scheduler.next_available_slot(45, DAY) == "08:00"
    assert scheduler.next_available_slot(61, DAY) == "10:00"


def test_next_slot_fits_exactly_back_to_back():
    """A task that exactly fills the gap (touching both neighbours) is allowed."""
    scheduler, mochi, _ = make_scheduler()
    add(mochi, "First", "07:00", 60)
    add(mochi, "Second", "08:30", 30)
    assert scheduler.next_available_slot(30, DAY) == "08:00"
    assert scheduler.next_available_slot(31, DAY) == "09:00"


def test_next_slot_returns_none_when_no_room():
    """If every gap is too small, the result is None."""
    scheduler, mochi, _ = make_scheduler()
    add(mochi, "All morning", "07:00", 120)
    assert scheduler.next_available_slot(30, DAY, window_start="07:00", window_end="09:00") is None


def test_next_slot_ignores_completed_tasks_and_other_days():
    """Completed tasks and tasks due on other days do not block the slot."""
    scheduler, mochi, _ = make_scheduler()
    done = add(mochi, "Done already", "07:00", 60)
    done.mark_complete()
    add(mochi, "Tomorrow", "07:00", 60, due=NEXT_DAY)
    assert scheduler.next_available_slot(60, DAY) == "07:00"


def test_next_slot_respects_window_end():
    """A slot must end by window_end, even if the gap before the next task is larger."""
    scheduler, mochi, _ = make_scheduler()
    add(mochi, "Most of the day", "07:00", 14 * 60)  # 07:00-21:00
    assert scheduler.next_available_slot(60, DAY) == "21:00"  # ends exactly at 22:00
    assert scheduler.next_available_slot(90, DAY) is None
    add(mochi, "After the window", "09:00", 30, due=NEXT_DAY)
    assert scheduler.next_available_slot(90, NEXT_DAY, window_start="07:00", window_end="08:00") is None


def test_next_slot_skips_past_task_that_starts_before_window():
    """A task that began before window_start still blocks until it ends."""
    scheduler, mochi, _ = make_scheduler()
    add(mochi, "Early", "06:00", 90)  # 06:00-07:30
    assert scheduler.next_available_slot(30, DAY) == "07:30"


def test_next_slot_rejects_non_positive_duration():
    """A zero or negative duration raises ValueError."""
    scheduler, _, _ = make_scheduler()
    with pytest.raises(ValueError):
        scheduler.next_available_slot(0, DAY)


# --- Availability windows ---


@pytest.mark.parametrize(
    "bad_windows",
    [["8:00-10:00"], ["08:00-25:00"], ["10:00-09:00"], ["09:00-09:00"], ["08:00-10:00", "09:30-11:00"], ["junk"]],
)
def test_owner_rejects_bad_availability_windows(bad_windows):
    """Malformed, reversed, empty, or overlapping windows raise ValueError."""
    with pytest.raises(ValueError, match="indow"):
        Owner("Jordan", availability_windows=bad_windows)


def test_owner_accepts_window_ending_at_midnight():
    """A window may end at 24:00, and windows are returned sorted."""
    owner = Owner("Jordan", availability_windows=["20:00-24:00", "06:00-08:00"])
    assert owner.window_ranges() == [(360, 480), (1200, 1440)]


def test_no_windows_means_available_all_day():
    """An owner without windows is available 00:00-24:00."""
    assert Owner("Jordan").window_ranges() == [(0, 24 * 60)]


def test_next_slot_skips_time_outside_owner_windows():
    """With the morning window full, the slot moves to the evening window."""
    scheduler, mochi, _ = make_scheduler()
    scheduler.owner.availability_windows = ["08:00-10:00", "17:00-20:00"]
    add(mochi, "Morning block", "08:00", 120)
    assert scheduler.next_available_slot(30, DAY) == "17:00"


def test_next_slot_uses_intersection_of_window_params_and_owner_windows():
    """The search is limited to both the window_start/window_end range and the owner's windows."""
    scheduler, _, _ = make_scheduler()
    scheduler.owner.availability_windows = ["06:00-09:00", "18:00-21:00"]
    assert scheduler.next_available_slot(30, DAY) == "07:00"  # default window_start 07:00
    assert scheduler.next_available_slot(30, DAY, window_start="10:00") == "18:00"
    assert scheduler.next_available_slot(30, DAY, window_start="20:45") is None


def test_next_slot_returns_none_when_no_owner_window_has_room():
    """A task longer than every owner window gets no slot."""
    scheduler, _, _ = make_scheduler()
    scheduler.owner.availability_windows = ["08:00-08:30", "12:00-12:45"]
    assert scheduler.next_available_slot(45, DAY) == "12:00"
    assert scheduler.next_available_slot(60, DAY) is None


# --- Overlap-aware daily plan ---


def test_plan_skips_lower_priority_overlap_even_if_it_starts_first():
    """The high-priority task is planned; the overlapping low-priority one is skipped with a reason."""
    scheduler, mochi, _ = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 30, priority="high")
    bath = add(mochi, "Bath", "07:50", 30, priority="low")
    plan = scheduler.build_daily_plan(DAY)
    assert plan.planned == [walk]
    assert plan.skipped == [bath]
    assert plan.reasons == {walk.task_id: "planned", bath.task_id: "skipped: overlaps Walk"}
    assert len(plan.warnings) == 1


def test_plan_combinable_overlap_both_planned_minutes_counted_once():
    """Two combinable overlapping tasks are both planned; shared minutes count once."""
    scheduler, mochi, bean = make_scheduler(budget=20)
    feed_m = add(mochi, "Feed Mochi", "08:00", 15)
    feed_b = add(bean, "Feed Bean", "08:05", 15)
    feed_m.combinable = feed_b.combinable = True
    plan = scheduler.build_daily_plan(DAY)
    assert plan.planned == [feed_m, feed_b]
    assert plan.minutes_used == 20  # 08:00-08:20, not 30
    assert plan.warnings == []


def test_plan_three_way_combinable_overlap_counted_once():
    """Three mutually overlapping combinable tasks use only the minutes of their union."""
    scheduler, mochi, _ = make_scheduler()
    for name, time in [("A", "08:00"), ("B", "08:10"), ("C", "08:20")]:
        add(mochi, name, time, 30).combinable = True
    plan = scheduler.build_daily_plan(DAY)
    assert len(plan.planned) == 3
    assert plan.minutes_used == 50  # 08:00-08:50


def test_plan_one_combinable_task_still_skips_overlap():
    """If only one of two overlapping tasks is combinable, the later one is still skipped."""
    scheduler, mochi, bean = make_scheduler()
    add(mochi, "Walk", "08:00", 30)
    feed = add(bean, "Feed", "08:10", 10)
    feed.combinable = True
    plan = scheduler.build_daily_plan(DAY)
    assert plan.skipped == [feed]


def test_plan_skips_tasks_outside_windows_without_using_budget():
    """Tasks not fully inside an availability window are skipped and use no budget."""
    scheduler, mochi, _ = make_scheduler(budget=30)
    scheduler.owner.availability_windows = ["08:00-12:00"]
    early = add(mochi, "Early", "06:00", 30, priority="high")
    straddle = add(mochi, "Straddle", "11:45", 30, priority="high")
    walk = add(mochi, "Walk", "09:00", 30, priority="low")
    plan = scheduler.build_daily_plan(DAY)
    assert plan.planned == [walk]
    assert plan.reasons[early.task_id] == "skipped: outside windows"
    assert plan.reasons[straddle.task_id] == "skipped: outside windows"
    assert plan.minutes_used == 30


def test_plan_gives_every_task_a_reason_of_each_kind():
    """One plan can hold planned, budget, overlap, and window reasons; every task has one."""
    scheduler, mochi, bean = make_scheduler(budget=40)
    scheduler.owner.availability_windows = ["07:00-20:00"]
    walk = add(mochi, "Walk", "08:00", 30, priority="high")
    bath = add(mochi, "Bath", "08:15", 20, priority="medium")
    groom = add(bean, "Groom", "10:00", 30, priority="low")
    night = add(bean, "Night feed", "21:00", 10, priority="low")
    plan = scheduler.build_daily_plan(DAY)
    assert plan.reasons == {
        walk.task_id: "planned",
        bath.task_id: "skipped: overlaps Walk",
        groom.task_id: "skipped: budget",
        night.task_id: "skipped: outside windows",
    }
    assert set(plan.reasons) == {t.task_id for t in plan.planned + plan.skipped}


# --- "Do both" overlap overrides ---


def test_allowed_overlap_is_planned_with_reason_warning_and_full_minutes():
    """An owner-allowed overlapping pair is planned; minutes count in full and a warning remains."""
    scheduler, mochi, bean = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 30)
    feed = add(bean, "Feed", "08:10", 10)
    scheduler.owner.allow_overlap(walk.task_id, feed.task_id, DAY)
    plan = scheduler.build_daily_plan(DAY)
    assert plan.planned == [walk, feed]
    assert plan.reasons[feed.task_id] == "planned: overlap allowed by owner"
    assert plan.minutes_used == 40
    assert plan.warnings == [
        "Conflict (cross-pet: Mochi, Bean): Walk 08:00-08:30 overlaps Feed 08:10-08:20 (allowed by owner)"
    ]


def test_allow_overlap_is_order_insensitive_and_not_duplicated():
    """The override pair is normalized, so either order matches and adding twice stores one entry."""
    scheduler, mochi, bean = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 30)
    feed = add(bean, "Feed", "08:10", 10)
    scheduler.owner.allow_overlap(feed.task_id, walk.task_id, DAY)
    scheduler.owner.allow_overlap(walk.task_id, feed.task_id, DAY)
    assert scheduler.owner.is_overlap_allowed(walk.task_id, feed.task_id, DAY)
    assert len(scheduler.owner.overlap_overrides) == 1
    override = scheduler.owner.overlap_overrides[0]
    assert override.task_a < override.task_b


def test_override_for_another_date_is_ignored():
    """An override recorded for a different day does not allow the overlap today."""
    scheduler, mochi, bean = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 30)
    feed = add(bean, "Feed", "08:10", 10)
    a, b = sorted((walk.task_id, feed.task_id))
    scheduler.owner.overlap_overrides.append(OverlapOverride(a, b, NEXT_DAY))
    plan = scheduler.build_daily_plan(DAY)
    assert plan.skipped == [feed]


def test_override_does_not_carry_to_next_recurring_copy():
    """Recurring copies get new task ids and dates, so yesterday's override no longer applies."""
    scheduler, mochi, bean = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 30, frequency="daily")
    feed = add(bean, "Feed", "08:10", 10, frequency="daily")
    scheduler.owner.allow_overlap(walk.task_id, feed.task_id, DAY)
    assert len(scheduler.build_daily_plan(DAY).planned) == 2

    scheduler.mark_task_complete(walk.task_id)
    next_feed = scheduler.mark_task_complete(feed.task_id)
    plan = scheduler.build_daily_plan(NEXT_DAY)
    assert plan.skipped == [next_feed]
    assert plan.reasons[next_feed.task_id] == "skipped: overlaps Walk"


def test_allow_overlap_rejects_unknown_same_or_wrong_day():
    """allow_overlap raises ValueError for an unknown id, the same id twice, or a task not due that day."""
    scheduler, mochi, bean = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 30)
    feed = add(bean, "Feed", "08:10", 10)
    with pytest.raises(ValueError):
        scheduler.owner.allow_overlap(walk.task_id, "missing", DAY)
    with pytest.raises(ValueError):
        scheduler.owner.allow_overlap(walk.task_id, walk.task_id, DAY)
    with pytest.raises(ValueError):
        scheduler.owner.allow_overlap(walk.task_id, feed.task_id, NEXT_DAY)
    assert scheduler.owner.overlap_overrides == []


def test_windows_and_overrides_round_trip_through_json(tmp_path):
    """Availability windows and overlap overrides survive save and load."""
    scheduler, mochi, bean = make_scheduler()
    scheduler.owner.availability_windows = ["07:00-12:00", "17:00-21:00"]
    walk = add(mochi, "Walk", "08:00", 30)
    feed = add(bean, "Feed", "08:10", 10)
    scheduler.owner.allow_overlap(walk.task_id, feed.task_id, DAY)
    path = tmp_path / "data.json"
    scheduler.owner.save_to_json(path)

    loaded = Owner.load_from_json(path)
    assert loaded == scheduler.owner
    assert loaded.overlap_overrides[0].day == DAY
    stored = json.loads(path.read_text(encoding="utf-8"))["overlap_overrides"][0]
    assert stored["day"] == DAY.isoformat()
    assert set(stored) == {"task_a", "task_b", "day"}


def test_old_json_without_new_keys_still_loads(tmp_path):
    """A Module 2 data file (no windows, overrides, combinable, or medication) still loads with defaults."""
    _, path = saved_owner(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("availability_windows", "overlap_overrides"):
        data.pop(key)
    for pet in data["pets"]:
        for task in pet["tasks"]:
            task.pop("combinable")
            task.pop("medication", None)
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = Owner.load_from_json(path)
    assert loaded.availability_windows == []
    assert loaded.overlap_overrides == []
    assert all(t.combinable is False for t in loaded.get_all_tasks())


def test_explain_plan_has_a_line_for_each_reason_type():
    """explain_plan words each reason: planned, allowed overlap, budget, overlap, and windows."""
    scheduler, mochi, bean = make_scheduler(budget=50)
    scheduler.owner.availability_windows = ["07:00-20:00"]
    walk = add(mochi, "Walk", "08:00", 30, priority="high")
    feed = add(bean, "Feed", "08:10", 10, priority="high")
    add(mochi, "Bath", "08:15", 20, priority="medium")
    add(bean, "Groom", "10:00", 30, priority="low")
    add(bean, "Night feed", "21:00", 10, priority="low")
    scheduler.owner.allow_overlap(walk.task_id, feed.task_id, DAY)
    lines = scheduler.explain_plan(scheduler.build_daily_plan(DAY))
    assert lines == [
        "Planned: Walk (Mochi) at 08:00, 30 min, high priority - fits in the 50-minute budget.",
        "Planned: Feed (Bean) at 08:10, 10 min, high priority - overlap with Walk allowed by owner.",
        "Skipped: Bath (Mochi) at 08:15 - overlaps Walk, which was planned first.",
        "Skipped: Groom (Bean) at 10:00 needs 30 min; did not fit in the 50-minute budget (planned tasks use 40).",
        "Skipped: Night feed (Bean) at 21:00 - outside the owner's availability windows (07:00-20:00).",
    ]


# --- Overdue tasks ---


def test_overdue_tasks_returns_only_past_pending_tasks_oldest_first():
    """Pending tasks due before today are overdue; today, future, and completed tasks are not."""
    scheduler, mochi, bean = make_scheduler()
    today = DAY + timedelta(days=3)
    newer = add(mochi, "Newer", "07:00", 10, due=DAY + timedelta(days=2))
    older = add(bean, "Older", "09:00", 10, due=DAY)
    add(mochi, "Due today", "08:00", 10, due=today)
    add(mochi, "Future", "08:00", 10, due=today + timedelta(days=1))
    add(bean, "Done", "08:00", 10, due=DAY).mark_complete()
    assert scheduler.overdue_tasks(today) == [older, newer]


def test_overdue_tasks_composes_with_filter():
    """overdue_tasks accepts a filtered task list."""
    scheduler, mochi, bean = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 10, due=DAY)
    add(bean, "Feed", "08:00", 10, due=DAY)
    assert scheduler.overdue_tasks(NEXT_DAY, scheduler.filter_tasks(pet_name="Mochi")) == [walk]


# --- move_task ---


def test_move_task_changes_time_and_keeps_same_task():
    """Moving a task updates its time in place; id, pet, and other fields are unchanged."""
    scheduler, mochi, _ = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 30, priority="high")
    moved = scheduler.move_task(walk.task_id, "09:15")
    assert moved is walk
    assert (walk.time, walk.due_date, walk.pet_name, walk.priority) == ("09:15", DAY, "Mochi", "high")
    assert mochi.get_tasks() == [walk]


def test_move_task_changes_time_and_date():
    """A new date moves the task to that day."""
    scheduler, mochi, _ = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 30)
    scheduler.move_task(walk.task_id, "07:00", NEXT_DAY)
    assert (walk.time, walk.due_date) == ("07:00", NEXT_DAY)


@pytest.mark.parametrize("bad_time", ["7:00", "24:00", "23:45", None])
def test_move_task_rejects_invalid_time_and_leaves_task_unchanged(bad_time):
    """A bad format, out-of-range time, or a move past midnight raises ValueError; the task is untouched."""
    scheduler, mochi, _ = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 30)
    with pytest.raises(ValueError):
        scheduler.move_task(walk.task_id, bad_time, NEXT_DAY)
    assert (walk.time, walk.due_date) == ("08:00", DAY)


def test_move_task_rejects_unknown_id():
    """An unknown task id raises ValueError."""
    scheduler, _, _ = make_scheduler()
    with pytest.raises(ValueError, match="Unknown"):
        scheduler.move_task("missing", "09:00")


def test_move_task_rejects_completed_task():
    """Completed tasks are history and cannot be moved."""
    scheduler, mochi, _ = make_scheduler()
    walk = add(mochi, "Walk", "08:00", 30)
    walk.mark_complete()
    with pytest.raises(ValueError, match="completed"):
        scheduler.move_task(walk.task_id, "09:00")
    assert walk.time == "08:00"


# --- Medication ---


def make_med(**overrides) -> Medication:
    """Return a valid twice-daily Medication, with any fields overridden."""
    values = {"drug": "Carprofen", "dose_text": "25 mg tablet", "times": ("08:00", "20:00"), "attested": True}
    values.update(overrides)
    return Medication(**values)


def test_medication_builds_with_valid_values():
    """A valid medication keeps its prescribed times as a tuple."""
    med = make_med(vet_name="Dr. Lee", vet_phone="555-0100", vet_consent=True)
    assert med.times == ("08:00", "20:00")
    assert med.label_verified is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"drug": "  "},
        {"dose_text": ""},
        {"times": ()},
        {"times": ("8:00",)},
        {"times": ("08:00", "08:00")},
        {"times": ["08:00"]},  # must be a tuple
    ],
)
def test_medication_rejects_bad_values(overrides):
    """Empty drug or dose, missing, malformed, duplicate, or non-tuple times raise ValueError."""
    with pytest.raises(ValueError):
        make_med(**overrides)


def test_medication_rejects_vet_details_without_consent():
    """Vet contact info cannot be stored unless vet_consent is True."""
    with pytest.raises(ValueError, match="consent"):
        make_med(vet_phone="555-0100")


def test_medication_dose_cannot_be_changed_in_place():
    """Medication is frozen, so dose_text cannot be reassigned."""
    med = make_med()
    with pytest.raises(FrozenInstanceError):
        med.dose_text = "50 mg"


def test_medication_dose_tasks_share_one_medication():
    """Each prescribed time is its own task; doses share the same Medication object."""
    _, mochi, _ = make_scheduler()
    med = make_med()
    morning = Task.medication_dose(med, "08:00", 5, DAY)
    evening = Task.medication_dose(med, "20:00", 5, DAY)
    mochi.add_task(morning)
    mochi.add_task(evening)
    assert morning.medication is evening.medication is med
    assert (morning.description, morning.priority, morning.frequency) == ("Carprofen", "high", "daily")


def test_medication_dose_rejects_non_prescribed_time():
    """Creating a dose at a time not in medication.times raises ValueError."""
    with pytest.raises(ValueError, match="prescribed"):
        Task.medication_dose(make_med(), "09:00", 5, DAY)


def test_moving_a_dose_off_prescribed_time_is_allowed_in_phase_1():
    """move_task does not check prescribed times; the Phase 2 verifier will enforce a tolerance."""
    scheduler, mochi, _ = make_scheduler()
    dose = Task.medication_dose(make_med(), "08:00", 5, DAY)
    mochi.add_task(dose)
    scheduler.move_task(dose.task_id, "08:30")
    assert dose.time == "08:30"
    assert dose.medication.times == ("08:00", "20:00")


def test_recurring_dose_keeps_its_medication():
    """Completing a daily dose creates a copy with the same Medication."""
    scheduler, mochi, _ = make_scheduler()
    med = make_med()
    dose = Task.medication_dose(med, "08:00", 5, DAY)
    mochi.add_task(dose)
    next_dose = scheduler.mark_task_complete(dose.task_id)
    assert next_dose.medication is med
    assert next_dose.due_date == NEXT_DAY


def test_medication_round_trips_through_json(tmp_path):
    """A task's medication survives save and load; tasks without one load as None."""
    scheduler, mochi, bean = make_scheduler()
    med = make_med(vet_clinic="Oak Vet", vet_consent=True, label_verified=True)
    mochi.add_task(Task.medication_dose(med, "20:00", 5, DAY))
    add(bean, "Feed", "09:00", 10)
    path = tmp_path / "data.json"
    scheduler.owner.save_to_json(path)

    loaded = Owner.load_from_json(path)
    assert loaded == scheduler.owner
    dose, feed = loaded.get_all_tasks()
    assert dose.medication == med
    assert isinstance(dose.medication.times, tuple)
    assert feed.medication is None
