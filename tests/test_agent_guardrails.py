"""Input guard, output guard, verifier, and confidence scoring."""

import copy
from datetime import date

import pytest
from agent_fixtures import TODAY, TOMORROW, YESTERDAY, final, make_med, make_world, proposal_of

from pawpal_agent.guardrails import (
    DOSING_REFUSAL,
    MEDICATION_FORM_ERROR,
    OUT_OF_SCOPE_REFUSAL,
    OUTPUT_REDACTED_NOTE,
    REMOVED_NOTE,
    check_invariants,
    check_request,
    redact_dosing,
    score_confidence,
    verify_proposal,
)
from pawpal_agent.proposal import (
    AddChange,
    CompleteChange,
    FinalReply,
    MoveChange,
    parse_change,
    parse_final_reply,
)
from pawpal_system import Medication, Task

PETS = ("Mochi", "Bean")
DRUGS = ("Carprofen",)


# --- input guard ---


@pytest.mark.parametrize(
    "request_text",
    [
        "How much Carprofen should I give Mochi?",
        "Can I double Bean's dose tonight?",
        "What's the right dosage for a 10 kg dog?",
        "Give Mochi 50 mg now",
        "Should Mochi get 50 mg?",
        "I missed a dose, should I give two pills?",
        "Move the Carprofen to 08:30 and increase the dose",
        "Is it an overdose if he ate two tablets?",
    ],
)
def test_dosing_questions_get_the_scripted_refusal(request_text):
    """Anything about dose amounts is refused with the exact scripted text."""
    result = check_request(request_text, PETS, DRUGS)
    assert (result.allowed, result.category, result.message) == (False, "dosing", DOSING_REFUSAL)


@pytest.mark.parametrize(
    "request_text",
    [
        "Move Mochi's 8am medication to 8:30",
        "Move Mochi's 25 mg Carprofen dose to 08:30",
        "Reschedule Bean's feeding to 9:30",
        "What's on Mochi's plan today?",
    ],
)
def test_scheduling_requests_are_allowed(request_text):
    """Scheduling a medication (without dosing intent) is in scope."""
    assert check_request(request_text, PETS, DRUGS).allowed


@pytest.mark.parametrize("request_text", ["Write me a poem about the ocean", "What is the capital of France?"])
def test_out_of_scope_requests_are_refused(request_text):
    """Non-scheduling requests get the out-of-scope refusal."""
    result = check_request(request_text, PETS, DRUGS)
    assert (result.allowed, result.message) == (False, OUT_OF_SCOPE_REFUSAL)


@pytest.mark.parametrize(
    "request_text",
    [
        "How much Carprofen should I give Mochi?",
        "How many Apoquel should I give Bean?",
        "Should I give Mochi the Rimadyl now?",
        "How much Gabapentin do I administer?",
    ],
)
def test_give_questions_are_dosing_even_for_unknown_drugs(request_text):
    """No medication tasks (no known drug names): "how much ... give" / "should I give" is still dosing."""
    result = check_request(request_text, PETS, drug_names=())
    assert (result.allowed, result.category) == (False, "dosing")


@pytest.mark.parametrize(
    "request_text",
    [
        "How much food should I give Mochi?",
        "How many treats should I give Bean after play?",
        "Should I give Mochi her kibble at 18:00 instead?",
        "How many cups of water should I give Bean?",
    ],
)
def test_give_questions_about_food_are_allowed(request_text):
    """Food amounts are not dosing (food, kibble, treats, water, meal, cups)."""
    assert check_request(request_text, PETS, drug_names=()).allowed


@pytest.mark.parametrize(
    "request_text",
    ["Should I give Mochi her walk at 9?", "Should I give Mochi her bath later?",
     "Should I give Bean his brushing after dinner?"],
)
def test_give_questions_about_care_activities_are_allowed(request_text):
    """Naming a care activity (and no dose word/amount) makes "should I give" a scheduling question."""
    assert check_request(request_text, PETS, drug_names=()).allowed


def test_give_question_naming_an_existing_task_is_allowed():
    """An existing non-medication task description also counts as a care activity."""
    text = "Should I give Mochi her zoomies session at 18:00?"
    assert not check_request(text, PETS, drug_names=()).allowed  # unknown words: still refused
    assert check_request(text, PETS, drug_names=(), task_names=("Zoomies session",)).allowed


@pytest.mark.parametrize(
    "request_text",
    ["Should I give Mochi her walk and 2 tablets?", "Should I give Mochi a pill after her walk?",
     "Should I give Mochi her meds with breakfast food?", "Should I give Mochi Carprofen after her walk?"],
)
def test_care_words_do_not_exempt_dose_words_amounts_or_drugs(request_text):
    """A dose word, amount, medication word, or known drug keeps it dosing even next to a care activity."""
    assert check_request(request_text, PETS, DRUGS).category == "dosing"


def test_medication_task_descriptions_do_not_exempt_dosing_questions():
    """run_agent passes only non-medication task names, so a 'Carprofen' dose task can't exempt the question."""
    from pawpal_agent.agent import run_agent
    from pawpal_agent.llm import FakeLLM

    world = make_world()  # has a Carprofen dose task
    fake = FakeLLM([])
    result = run_agent("How much Carprofen should I give Mochi?", world.owner, TODAY, fake, log_dir=None)
    assert result.proposal.refusal == DOSING_REFUSAL and fake.calls == []


def test_unknown_drug_dosing_question_never_reaches_the_model():
    """End to end with no medication tasks: the input guard refuses with 0 LLM calls."""
    from pawpal_agent.agent import run_agent
    from pawpal_agent.llm import FakeLLM
    from pawpal_system import Owner, Pet

    owner = Owner("Jordan")
    owner.add_pet(Pet("Mochi", "dog"))
    fake = FakeLLM([])
    result = run_agent("How much Carprofen should I give Mochi?", owner, TODAY, fake, log_dir=None)
    assert result.proposal.refusal == DOSING_REFUSAL
    assert fake.calls == [] and result.usage.llm_calls == 0


def test_empty_and_too_long_requests_are_refused():
    """Blank and >500-char requests are refused before any LLM call."""
    assert check_request("   ", PETS).category == "empty"
    assert check_request("walk " * 101, PETS).category == "too_long"


# --- output guard ---


def test_output_guard_redacts_dose_amounts():
    """A summary or note stating a dose is replaced with the scripted refusal and flagged."""
    reply = FinalReply("Moved the walk; give her 50 mg after.", 0.9, ["Carprofen is fine", "give her 50 mg"], None)
    cleaned, flagged = redact_dosing(reply, DRUGS)
    assert flagged == ["Moved the walk; give her 50 mg after.", "give her 50 mg"]
    assert cleaned.summary == DOSING_REFUSAL
    assert cleaned.notes == ["Carprofen is fine", DOSING_REFUSAL, OUTPUT_REDACTED_NOTE]


def test_output_guard_on_a_proposal_with_changes_uses_a_neutral_placeholder():
    """With changes, flagged summary/notes become REMOVED_NOTE (not a refusal); the refusal field still gets it."""
    reply = FinalReply("Moved the dose; give her 50 mg.", 0.9, ["Dose stays the same.", "give her 50 mg"], None)
    cleaned, flagged = redact_dosing(reply, DRUGS, has_changes=True)
    assert cleaned.summary == REMOVED_NOTE
    assert cleaned.notes == ["Dose stays the same.", REMOVED_NOTE, OUTPUT_REDACTED_NOTE]
    assert DOSING_REFUSAL not in cleaned.notes
    assert flagged == ["Moved the dose; give her 50 mg.", "give her 50 mg"]

    refusing = FinalReply("No change.", 0.5, [], "You should give her 50 mg.")
    assert redact_dosing(refusing, DRUGS, has_changes=True)[0].refusal == DOSING_REFUSAL


def test_output_guard_leaves_normal_text_alone():
    """Scheduling summaries, including moving a dose, pass untouched."""
    reply = FinalReply("Moved Mochi's Carprofen dose to 20:30 to avoid the walk.", 0.8, ["No conflicts."], None)
    assert redact_dosing(reply, DRUGS) == (reply, [])


@pytest.mark.parametrize(
    "summary",
    [
        "Moved Mochi's 25 mg Carprofen dose to 08:30",
        "Moved the evening pill half an hour later",
        "Rescheduled the 25 mg dose to 20:30.",
        "Shifted the evening tablet a half hour later.",
    ],
)
def test_output_guard_allows_verified_scheduling_summaries(summary):
    """Past-tense scheduling statements that mention a dose (or "half an hour") are not dosing advice."""
    reply = FinalReply(summary, 0.8, [], None)
    assert redact_dosing(reply, DRUGS) == (reply, [])


@pytest.mark.parametrize(
    "summary",
    [
        "give her 50 mg",
        "Moved the walk; give her 50 mg after.",
        "Moved the dose to 08:30 and administered 25 mg.",
        "Moved the dose and gave half a tablet.",
    ],
)
def test_output_guard_still_redacts_amounts_with_give_or_half_a_dose(summary):
    """An amount next to give/administer, or "half a tablet", is redacted even inside a scheduling sentence."""
    cleaned, flagged = redact_dosing(FinalReply(summary, 0.8, [], None), DRUGS)
    assert flagged == [summary] and cleaned.summary == DOSING_REFUSAL


def test_input_guard_half_an_hour_is_timing_but_half_a_tablet_is_dosing():
    """'half an hour' is a time; 'half a tablet' and 'give 50 mg' are dosing, even with a scheduling verb."""
    assert check_request("Move the evening pill half an hour later", PETS, DRUGS).allowed
    assert check_request("Can I give half a tablet?", PETS, DRUGS).category == "dosing"
    assert check_request("Move the Carprofen to 08:30 and give 50 mg", PETS, DRUGS).category == "dosing"


# --- parsing ---


@pytest.mark.parametrize(
    "data",
    [
        {"kind": "allow_overlap", "task_a": "a", "task_b": "b", "rationale": "x"},
        {"kind": "delete", "task_id": "a", "rationale": "x"},
        {"kind": "move", "task_id": "a", "new_time": "09:00", "rationale": "x", "dose_text": "50 mg"},
        {"kind": "add", "task_id": "a", "pet_name": "Mochi", "description": "Pill", "time": "09:00",
         "duration_minutes": 5, "due_date": "2026-01-05", "rationale": "x", "medication": {}},
        {"kind": "complete", "task_id": "a"},
    ],
)
def test_parse_change_rejects_forbidden_kinds_and_fields(data):
    """Only move/add/complete with their exact fields can be parsed."""
    with pytest.raises(ValueError):
        parse_change(data)


@pytest.mark.parametrize(
    "text",
    [
        "Sure! Here is the JSON",
        '{"summary": "x", "confidence": 1.5, "notes": [], "refusal": null}',
        '{"summary": "x", "confidence": true, "notes": [], "refusal": null}',
        '{"summary": "x", "confidence": 0.5, "notes": []}',
        '{"summary": "x", "confidence": 0.5, "notes": [], "refusal": null, "changes": []}',
    ],
)
def test_parse_final_reply_is_strict(text):
    """Non-JSON, out-of-range or boolean confidence, and missing/extra keys are rejected."""
    with pytest.raises(ValueError):
        parse_final_reply(text, 0)


def test_parse_final_reply_rejects_refusal_with_changes():
    """A refusal cannot come with proposed changes."""
    with pytest.raises(ValueError, match="refusal"):
        parse_final_reply(final(refusal="no").text, change_count=1)


# --- verifier: accepted proposals ---


def test_valid_move_add_and_complete_pass_without_touching_owner():
    """A clean proposal verifies, and the real owner is not modified."""
    world = make_world()
    before = copy.deepcopy(world.owner)
    proposal = proposal_of(
        MoveChange(world.play.task_id, "11:00", None, "later"),
        AddChange("new00001", "Bean", "Brush Bean", "12:00", 15, TODAY, "requested"),
        CompleteChange(world.feed.task_id, "already fed"),
    )
    assert verify_proposal(world.owner, proposal, TODAY).ok
    assert world.owner == before


# --- verifier: rejected proposals ---


@pytest.mark.parametrize(
    "make_change, message",
    [
        (lambda w: MoveChange("nope0000", "09:00", None, "x"), "Unknown task id"),
        (lambda w: AddChange("new00001", "Rex", "Walk Rex", "11:00", 30, TODAY, "x"), "Unknown pet"),
        (lambda w: MoveChange(w.walk.task_id, "24:00", None, "x"), "Invalid time"),
        (lambda w: MoveChange(w.walk.task_id, "23:50", None, "x"), "past midnight"),
        (lambda w: MoveChange(w.walk.task_id, "11:00", YESTERDAY, "x"), "before today"),
        (lambda w: AddChange("new00001", "Bean", "Brush", "11:00", 5, YESTERDAY, "x"), "before today"),
        (lambda w: MoveChange(w.walk.task_id, "11:00", None, " "), "rationale"),
    ],
)
def test_structural_errors_are_rejected(make_change, message):
    """Unknown ids/pets, invalid times, past dates, and empty rationale fail verification."""
    world = make_world()
    result = verify_proposal(world.owner, proposal_of(make_change(world)), TODAY)
    assert not result.ok
    assert message in result.errors[0]


def test_moving_a_completed_task_is_rejected():
    """Completed tasks cannot be moved."""
    world = make_world()
    world.walk.mark_complete()
    result = verify_proposal(world.owner, proposal_of(MoveChange(world.walk.task_id, "11:00", None, "x")), TODAY)
    assert not result.ok and "completed" in result.errors[0]


def test_two_changes_to_one_task_are_rejected():
    """At most one change per task id."""
    world = make_world()
    proposal = proposal_of(
        MoveChange(world.play.task_id, "11:00", None, "x"),
        CompleteChange(world.play.task_id, "y"),
    )
    result = verify_proposal(world.owner, proposal, TODAY)
    assert not result.ok and "one change per task" in result.errors[0]


def test_unsupported_change_object_is_rejected():
    """Anything that is not Move/Add/Complete (e.g. an override request) is rejected."""
    world = make_world()
    result = verify_proposal(world.owner, proposal_of({"kind": "allow_overlap"}), TODAY)
    assert not result.ok and "unsupported change kind" in result.errors[0]


def test_move_creating_a_new_conflict_is_rejected():
    """Moving Play onto the walk creates a conflict the verifier reports."""
    world = make_world()
    result = verify_proposal(world.owner, proposal_of(MoveChange(world.play.task_id, "08:15", None, "x")), TODAY)
    assert not result.ok
    assert result.errors[0].startswith("new conflict:") and "Play" in result.errors[0]


def test_pre_existing_conflict_is_not_blamed_on_the_proposal():
    """An overlap that already existed does not block an unrelated change on the same day."""
    world = make_world()
    world.mochi.add_task(Task("Brush", "08:10", 10, "low", due_date=TODAY))  # overlaps the walk already
    result = verify_proposal(world.owner, proposal_of(MoveChange(world.play.task_id, "11:00", None, "x")), TODAY)
    assert result.ok, result.errors


def test_move_outside_availability_windows_is_rejected():
    """A task moved into a gap between windows fails."""
    world = make_world(windows=["07:00-12:00", "17:00-22:00"])
    result = verify_proposal(world.owner, proposal_of(MoveChange(world.play.task_id, "13:00", None, "x")), TODAY)
    assert not result.ok and "availability windows" in result.errors[0]


def test_add_over_budget_is_rejected():
    """An added task that does not fit the remaining minutes fails."""
    world = make_world(budget=80)  # walk 30 + feed 10 + play 30 + dose 5 = 75
    add = AddChange("new00001", "Bean", "Brush Bean", "11:00", 15, TODAY, "x", priority="low")
    result = verify_proposal(world.owner, proposal_of(add), TODAY)
    assert not result.ok and "would not fit the day's plan (skipped: budget)" in result.errors[0]


def test_add_that_pushes_a_planned_task_out_is_rejected():
    """A high-priority add that displaces a lower-priority planned task fails."""
    world = make_world(budget=80)
    add = AddChange("new00001", "Bean", "Vet visit", "13:00", 30, TODAY, "x", priority="high")
    result = verify_proposal(world.owner, proposal_of(add), TODAY)
    assert not result.ok
    assert result.errors == ["Play would be pushed out of the plan (skipped: budget)"]


@pytest.mark.parametrize("description", ["Give Carprofen 25 mg", "Extra pill for Mochi", "Evening dose"])
def test_add_that_looks_like_medication_is_rejected(description):
    """Medication look-alikes must go through the human medication form."""
    world = make_world()
    add = AddChange("new00001", "Mochi", description, "11:00", 5, TODAY, "x")
    result = verify_proposal(world.owner, proposal_of(add), TODAY)
    assert not result.ok and MEDICATION_FORM_ERROR in result.errors[0]


def test_add_with_normal_description_is_allowed():
    """An ordinary care task is not mistaken for medication."""
    world = make_world()
    add = AddChange("new00001", "Mochi", "Evening walk", "18:00", 30, TODAY, "x")
    assert verify_proposal(world.owner, proposal_of(add), TODAY).ok


# --- verifier: medication tolerance ---


def test_dose_moved_within_tolerance_passes():
    """20:00 -> 20:30 is within 60 min of the prescribed time."""
    world = make_world()
    assert verify_proposal(world.owner, proposal_of(MoveChange(world.dose.task_id, "20:30", None, "x")), TODAY).ok


@pytest.mark.parametrize("new_time", ["21:30", "18:59"])
def test_dose_moved_beyond_tolerance_is_rejected(new_time):
    """More than 60 min from the prescribed time fails, in either direction."""
    world = make_world()
    result = verify_proposal(world.owner, proposal_of(MoveChange(world.dose.task_id, new_time, None, "x")), TODAY)
    assert not result.ok and "prescribed time 20:00" in result.errors[0]


def test_dose_cannot_slide_to_the_other_prescribed_time():
    """08:00 -> 19:30 is near 20:00 but far from its own slot, so it is rejected."""
    world = make_world()
    morning = Task.medication_dose(world.dose.medication, "08:00", 5, TODAY)
    world.mochi.add_task(morning)
    result = verify_proposal(world.owner, proposal_of(MoveChange(morning.task_id, "19:30", None, "x")), TODAY)
    assert not result.ok and "prescribed time 08:00" in result.errors[0]


def test_dose_cannot_move_to_another_day():
    """Medication doses stay on their day."""
    world = make_world()
    result = verify_proposal(world.owner, proposal_of(MoveChange(world.dose.task_id, "20:00", TOMORROW, "x")), TODAY)
    assert not result.ok and "another day" in result.errors[0]


# --- invariants (defense in depth) ---


def test_invariants_reject_added_override_and_changed_medication():
    """Overrides and medication details can only change through the human, never via a proposal."""
    world = make_world()
    after = copy.deepcopy(world.owner)
    after.allow_overlap(world.walk.task_id, world.play.task_id, TODAY)
    after.find_task(world.dose.task_id)[1].medication = Medication(
        drug="Carprofen", dose_text="50 mg tablet", times=("08:00", "20:00"), attested=True
    )
    errors = check_invariants(world.owner, after)
    assert any("overrides" in e for e in errors)
    assert any("medication details" in e for e in errors)


def test_invariants_reject_removed_task_and_new_medication_task():
    """A deleted task or a new medication task is caught."""
    world = make_world()
    after = copy.deepcopy(world.owner)
    after.get_pet("Mochi").remove_task(world.play.task_id)
    after.get_pet("Bean").add_task(Task.medication_dose(make_med(), "08:00", 5, date(2026, 1, 5)))
    errors = check_invariants(world.owner, after)
    assert any("deleting tasks is not allowed" in e for e in errors)
    assert any(MEDICATION_FORM_ERROR in e for e in errors)


# --- confidence ---


@pytest.mark.parametrize(
    "model_conf, retries, verified, expected",
    [(0.8, 0, True, 0.8), (0.8, 1, True, 0.68), (0.8, 2, True, 0.56), (0.9, 0, False, 0.0), (1.0, 0, True, 1.0)],
)
def test_score_confidence(model_conf, retries, verified, expected):
    """Unverified is 0; each retry costs 15%."""
    assert score_confidence(model_conf, retries, verified) == expected
