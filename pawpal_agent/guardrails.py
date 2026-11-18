"""Guardrails: input guard, output guard, deterministic proposal verifier, and confidence scoring."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field, replace
from datetime import date

from pawpal_system import Owner, Scheduler, Task

from .proposal import (
    CHANGE_TYPES,
    AddChange,
    CompleteChange,
    FinalReply,
    MoveChange,
    Proposal,
    apply_change,
    change_label,
)

DOSING_REFUSAL = (
    "I can't give dosing advice. Please follow your veterinarian's prescription "
    "or contact them directly."
)
OUT_OF_SCOPE_REFUSAL = "I can only help with scheduling your pets' care tasks."
EMPTY_REFUSAL = "Please type a scheduling request, for example: 'Move Mochi's walk to 09:00'."
TOO_LONG_REFUSAL = "That request is too long. Please keep it under 500 characters."
OUTPUT_REDACTED_NOTE = "output guard redacted dosing text"
REMOVED_NOTE = "[note removed by medication safety filter]"
MEDICATION_FORM_ERROR = "medication tasks must be created via the medication form"

MAX_REQUEST_CHARS = 500
DOSE_TOLERANCE_MINUTES = 60

# --- dosing detector (shared by the input guard, output guard, and the add-description check) ---

_DOSE_WORDS = re.compile(
    r"\b(doses?|mg|mcg|ml|cc|milligrams?|millilit(?:er|re)s?|tablets?|pills?|capsules?|drops)\b"
)
_STRONG_DOSE_WORDS = re.compile(r"\b(dosage|dosing|overdos\w*)\b")  # always about amount
_AMOUNT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(mg|mcg|ml|cc|milligrams?|millilit(?:er|re)s?|units?|iu|tablets?|pills?|capsules?|drops)\b"
)
_DOSE_INTENT = re.compile(
    r"\b(how much|how many|increase|decrease|reduce|raise|lower|double|triple|halve|"
    r"extra|more|less|adjust|convert|calculate|safe|missed|skip a dose|right amount|"
    r"change the dose|can i give)\b"
    r"|\bhalf\b(?!\s+(?:an\s+)?hours?\b)"  # "half a tablet" is dosing; "half an hour later" is timing
    r"|\bshould\b.*\b(get|give|take|have)\b"
)
# "How much ... give" / "should I give ..." is a dosing question even for drug names we don't know,
# unless it names a care activity or an existing (non-medication) task and has no dose word or amount.
_ADMIN_QUESTION = re.compile(
    r"\b(how much|how many)\b.*\b(give|giving|administer\w*)\b"
    r"|\bshould i\b.*\b(give|administer\w*)\b"
)
_CARE_WORDS = re.compile(
    r"\b(walks?|walking|baths?|bathing|brush|brushing|groom|grooming|play|playtime|feed|feeding|"
    r"food|kibble|meals?|treats?|water|cups?)\b"
)
_MED_WORDS = re.compile(r"\b(medication|medicine|meds|drug|prescription|antibiotics?|painkillers?)\b")
_SCHEDULING_VERB = re.compile(
    r"\b(move|moved|moving|reschedule|rescheduled|shift|shifted|push|pushed|change the time)\b"
)
# An amount next to one of these is always dosing, even in a scheduling sentence ("Moved play. Give her 50 mg").
_ADMIN_VERB = re.compile(r"\b(give|giving|gave|administer\w*)\b")
_SCOPE_WORDS = re.compile(
    r"\b(schedule|reschedule|plan|planning|task|tasks|walk|walks|feed|feeding|food|meal|groom|grooming|"
    r"bath|brush|vet|appointment|play|litter|clean|medication|meds|dose|pill|move|shift|push|add|"
    r"complete|done|finish|finished|today|tomorrow|tonight|morning|afternoon|evening|time|conflict|"
    r"conflicts|overlap|slot|free|busy|remind|pet|pets|dog|cat|later|earlier|overdue|am|pm)\b"
    r"|\b\d{1,2}:\d{2}\b"
)


def has_dose_words(text: str) -> bool:
    """Return True if text mentions dose units or dose forms (mg, tablet, dose, ...)."""
    lowered = text.lower()
    return bool(_DOSE_WORDS.search(lowered) or _STRONG_DOSE_WORDS.search(lowered))


def has_amount(text: str) -> bool:
    """Return True if text contains a number followed by a dose unit, e.g. "25 mg"."""
    return bool(_AMOUNT.search(text.lower()))


def _names_care(lowered: str, task_names: tuple[str, ...]) -> bool:
    """Return True if lowered text names a care activity or one of the given task descriptions."""
    return bool(_CARE_WORDS.search(lowered)) or any(
        re.search(rf"\b{re.escape(n.lower().strip())}\b", lowered) for n in task_names if n.strip()
    )


def is_dosing_text(
    text: str,
    drug_names: tuple[str, ...] = (),
    allow_scheduling_amounts: bool = False,
    task_names: tuple[str, ...] = (),
) -> bool:
    """Return True if text asks about or states a dose amount; optionally let "move the 25 mg dose" through."""
    lowered = text.lower()
    if _STRONG_DOSE_WORDS.search(lowered):
        return True
    if _ADMIN_QUESTION.search(lowered):
        about_care = _names_care(lowered, task_names) and not (
            _DOSE_WORDS.search(lowered)
            or _AMOUNT.search(lowered)
            or _MED_WORDS.search(lowered)
            or any(re.search(rf"\b{re.escape(d.lower())}\b", lowered) for d in drug_names if d.strip())
        )
        if not about_care:
            return True
    has_intent = bool(_DOSE_INTENT.search(lowered))
    mentions_med = bool(
        _DOSE_WORDS.search(lowered)
        or _MED_WORDS.search(lowered)
        or any(re.search(rf"\b{re.escape(d.lower())}\b", lowered) for d in drug_names if d.strip())
    )
    if has_intent and mentions_med:
        return True
    if _AMOUNT.search(lowered):
        is_scheduling = allow_scheduling_amounts and bool(_SCHEDULING_VERB.search(lowered))
        return not is_scheduling or has_intent or bool(_ADMIN_VERB.search(lowered))
    return False


# --- input guard ---


@dataclass(frozen=True)
class GuardResult:
    """Outcome of the input guard: allowed, or a category and scripted refusal message."""

    allowed: bool
    category: str = "ok"  # ok | empty | too_long | dosing | out_of_scope
    message: str = ""


def check_request(
    text: str,
    pet_names: tuple[str, ...] = (),
    drug_names: tuple[str, ...] = (),
    task_names: tuple[str, ...] = (),
) -> GuardResult:
    """Refuse empty, overlong, dosing, or out-of-scope requests before any LLM call (task_names: non-med tasks)."""
    stripped = text.strip()
    if not stripped:
        return GuardResult(False, "empty", EMPTY_REFUSAL)
    if len(stripped) > MAX_REQUEST_CHARS:
        return GuardResult(False, "too_long", TOO_LONG_REFUSAL)
    if is_dosing_text(stripped, drug_names, allow_scheduling_amounts=True, task_names=task_names):
        return GuardResult(False, "dosing", DOSING_REFUSAL)
    lowered = stripped.lower()
    names_hit = any(re.search(rf"\b{re.escape(n.lower())}\b", lowered) for n in pet_names if n.strip())
    if not (_SCOPE_WORDS.search(lowered) or names_hit):
        return GuardResult(False, "out_of_scope", OUT_OF_SCOPE_REFUSAL)
    return GuardResult(True)


# --- output guard ---


def redact_dosing(
    reply: FinalReply, drug_names: tuple[str, ...] = (), has_changes: bool = False
) -> tuple[FinalReply, list[str]]:
    """Replace dosing text in the reply; return the cleaned reply and the original flagged texts (for the trace).

    On a proposal with changes, a flagged summary/note becomes REMOVED_NOTE so a valid change doesn't read
    like a refusal; the refusal field, or a reply with no changes, gets the scripted DOSING_REFUSAL.
    """
    flagged: list[str] = []

    def clean(text: str, replacement: str) -> str:
        """Return replacement if text is dosing text (scheduling statements pass), else text unchanged."""
        if is_dosing_text(text, drug_names, allow_scheduling_amounts=True):
            flagged.append(text)
            return replacement
        return text

    body_replacement = REMOVED_NOTE if has_changes else DOSING_REFUSAL
    summary = clean(reply.summary, body_replacement)
    notes = [clean(n, body_replacement) for n in reply.notes]
    refusal = clean(reply.refusal, DOSING_REFUSAL) if reply.refusal else reply.refusal
    if flagged:
        notes.append(OUTPUT_REDACTED_NOTE)
    return replace(reply, summary=summary, notes=notes, refusal=refusal), flagged


# --- verifier ---


@dataclass
class VerificationResult:
    """Outcome of verify_proposal: ok, or a list of ASCII error strings to feed back to the model."""

    ok: bool
    errors: list[str] = field(default_factory=list)


def _minutes(hhmm: str) -> int:
    """Convert "HH:MM" to minutes since midnight."""
    return int(hhmm[:2]) * 60 + int(hhmm[3:])


def dose_anchor(task: Task) -> str:
    """Return the prescribed time nearest the dose task's current time (its own slot)."""
    return min(task.medication.times, key=lambda t: abs(_minutes(t) - _minutes(task.time)))


def dose_time_ok(task: Task, new_time: str) -> bool:
    """Return True if new_time is within DOSE_TOLERANCE_MINUTES of the dose's own prescribed time."""
    return abs(_minutes(new_time) - _minutes(dose_anchor(task))) <= DOSE_TOLERANCE_MINUTES


def check_invariants(before: Owner, after: Owner) -> list[str]:
    """Return errors if overrides changed, any task vanished, or any medication was added or changed."""
    errors: list[str] = []
    if after.overlap_overrides != before.overlap_overrides:
        errors.append("overlap overrides can only be granted by the owner, not proposed")
    before_ids = set()
    for task in before.get_all_tasks():
        before_ids.add(task.task_id)
        found = after.find_task(task.task_id)
        if found is None:
            errors.append(f"task {task.task_id} would be removed; deleting tasks is not allowed")
        elif found[1].medication != task.medication:
            errors.append(f"medication details of task {task.task_id} cannot be changed")
    for task in after.get_all_tasks():
        if task.task_id not in before_ids and task.medication is not None:
            errors.append(f"new task {task.task_id}: {MEDICATION_FORM_ERROR}")
    return errors


def _in_a_window(owner: Owner, task: Task) -> bool:
    """Return True if the whole task sits inside one of the owner's availability windows."""
    start = _minutes(task.time)
    end = start + task.duration_minutes
    return any(w_start <= start and end <= w_end for w_start, w_end in owner.window_ranges())


def _day_tasks(owner: Owner, day: date) -> list[Task]:
    """Return every task due on day."""
    return [t for t in owner.get_all_tasks() if t.due_date == day]


def verify_proposal(owner: Owner, proposal: Proposal, today: date) -> VerificationResult:
    """Check every change on a deep copy of the owner; the real owner is never modified."""
    errors: list[str] = []
    candidate = copy.deepcopy(owner)
    before = Scheduler(owner)  # read-only use
    after = Scheduler(candidate)
    seen: set[str] = set()
    touched: list[str] = []  # moved or added task ids
    days: set[date] = set()

    # 1) Per-change checks, applying each change to the candidate in order.
    for index, change in enumerate(proposal.changes, start=1):
        label = change_label(index, change)
        if not isinstance(change, CHANGE_TYPES):
            errors.append(f"{label}: unsupported change kind; only move, add, and complete are allowed")
            continue
        if not change.rationale.strip():
            errors.append(f"{label}: rationale must not be empty")
            continue
        if change.task_id in seen:
            errors.append(f"{label}: only one change per task is allowed")
            continue
        seen.add(change.task_id)
        if isinstance(change, AddChange) and (has_dose_words(change.description) or has_amount(change.description)):
            errors.append(f"{label}: {MEDICATION_FORM_ERROR}")
            continue
        target = change.new_date if isinstance(change, MoveChange) else getattr(change, "due_date", None)
        if target is not None and target < today:
            errors.append(f"{label}: cannot schedule on {target.isoformat()}, which is before today")
            continue

        found = owner.find_task(change.task_id) if not isinstance(change, AddChange) else None
        try:
            apply_change(after, change, today)
        except ValueError as err:
            errors.append(f"{label}: {err}")
            continue

        if isinstance(change, MoveChange):
            original = found[1]
            if original.medication is not None:
                if change.new_date is not None and change.new_date != original.due_date:
                    errors.append(f"{label}: medication doses cannot be moved to another day")
                    continue
                if not dose_time_ok(original, change.new_time):
                    errors.append(
                        f"{label}: medication dose must stay within {DOSE_TOLERANCE_MINUTES} min of its "
                        f"prescribed time {dose_anchor(original)} (proposed {change.new_time})"
                    )
                    continue
            days.add(original.due_date)
        if isinstance(change, CompleteChange):
            days.add(found[1].due_date)
        if not isinstance(change, CompleteChange):
            touched.append(change.task_id)
            days.add(candidate.find_task(change.task_id)[1].due_date)

    errors += check_invariants(owner, candidate)
    if errors:
        return VerificationResult(False, errors)

    # 2) Day-level checks: new conflicts, windows, and budget / plan regression.
    flagged: set[str] = set()
    for day in sorted(days):
        old_pairs = {frozenset((a.task_id, b.task_id)) for a, b in before.find_conflict_pairs(_day_tasks(owner, day))}
        for a, b in after.find_conflict_pairs(_day_tasks(candidate, day)):
            if frozenset((a.task_id, b.task_id)) not in old_pairs:
                errors.append(f"new conflict: {after.detect_conflicts([a, b])[0]}")
                flagged.update((a.task_id, b.task_id))

        for task_id in touched:
            task = candidate.find_task(task_id)[1]
            if task.due_date == day and task_id not in flagged and not _in_a_window(candidate, task):
                windows = ", ".join(candidate.availability_windows)
                errors.append(f"{task.description} at {task.time} is outside the owner's availability windows ({windows})")
                flagged.add(task_id)

        old_plan = before.build_daily_plan(day)
        new_plan = after.build_daily_plan(day)
        new_planned = {t.task_id for t in new_plan.planned}
        for task_id in touched:
            task = candidate.find_task(task_id)[1]
            if task.due_date == day and task_id not in flagged and task_id not in new_planned:
                errors.append(f"{task.description} would not fit the day's plan ({new_plan.reasons[task_id]})")
                flagged.add(task_id)
        for old in old_plan.planned:
            if old.task_id in flagged or old.task_id in touched:
                continue
            now = candidate.find_task(old.task_id)[1]
            if not now.completed and now.due_date == day and old.task_id not in new_planned:
                errors.append(f"{now.description} would be pushed out of the plan ({new_plan.reasons[old.task_id]})")
                flagged.add(old.task_id)

    return VerificationResult(not errors, errors)


# --- confidence ---


def score_confidence(model_confidence: float, retries: int, verified: bool) -> float:
    """Return 0 if unverified, else the model's confidence reduced 15% per verifier retry, clamped to [0, 1]."""
    if not verified:
        return 0.0
    return round(max(0.0, min(1.0, model_confidence * (1 - 0.15 * retries))), 2)
