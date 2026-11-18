"""Phase 5 medication flow: form validation, label cross-check (stubbed vision), privacy of the verification log."""

import base64
import hashlib
import io
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from agent_fixtures import TODAY, make_world
from PIL import Image

from pawpal_agent.llm import LLMError
from pawpal_agent.medication import (
    LABEL_SCHEMA,
    MAX_IMAGE_EDGE,
    AnthropicVisionClient,
    FakeVisionClient,
    LabelReading,
    MedicationEntry,
    build_dose_tasks,
    compare_label,
    cross_check_label,
    dose_problems,
    doses_match,
    label_frequency,
    parse_label_json,
    prepare_image,
    register_medication,
    save_label_copy,
    validate_entry,
)
from pawpal_agent.proposal import parse_change
from pawpal_agent.tools import TOOL_SCHEMAS
from pawpal_system import Medication, Task

ROOT = Path(__file__).resolve().parent.parent
LOG_NAME = "medication_verification.jsonl"
MATCHING = LabelReading("Fakeprofen chewable tablets", "25 mg (1 tablet)", "give twice daily (every 12 hours)",
                        "Mochi (dog)", True)


def entry(**overrides) -> MedicationEntry:
    """Return a valid twice-daily Fakeprofen entry for Mochi (override any field)."""
    values = dict(pet_name="Mochi", drug="Fakeprofen", dose_text="25 mg", dose_retyped="25 mg",
                  times_text="08:00, 20:00", start_date=TODAY, attested=True)
    return MedicationEntry(**{**values, **overrides})


def png_bytes(size: tuple[int, int] = (120, 80)) -> bytes:
    """Return a small in-memory PNG (never written to disk)."""
    out = io.BytesIO()
    Image.new("RGB", size, "white").save(out, format="PNG")
    return out.getvalue()


def log_records(log_dir: Path) -> list[dict]:
    """Read every verification record."""
    return [json.loads(line) for line in (log_dir / LOG_NAME).read_text(encoding="utf-8").splitlines()]


def med_tasks(owner) -> list[Task]:
    """Return every medication task the owner has."""
    return [t for t in owner.get_all_tasks() if t.medication is not None]


# --- form validation ---


def test_valid_entry_has_no_errors():
    assert validate_entry(entry(), make_world().owner) == []


def test_retyped_dose_must_match():
    errors = validate_entry(entry(dose_retyped="50 mg"), make_world().owner)
    assert any("re-typed dose does not match" in e for e in errors)


def test_retyped_dose_ignores_case_and_spaces():
    assert validate_entry(entry(dose_text="25 mg Tablet", dose_retyped=" 25MG  tablet "), make_world().owner) == []


def test_attestation_is_required():
    errors = validate_entry(entry(attested=False), make_world().owner)
    assert any("matches my veterinarian's prescription label" in e for e in errors)


def test_vet_info_without_consent_is_rejected_not_dropped():
    errors = validate_entry(entry(vet_phone="555-0100"), make_world().owner)
    assert errors == ["Vet contact details need your consent: check the consent box or clear the vet fields."]


def test_vet_info_with_consent_is_stored():
    tasks = build_dose_tasks(entry(vet_name="Dr. Example", vet_consent=True))
    assert tasks[0].medication.vet_name == "Dr. Example" and tasks[0].medication.vet_consent


@pytest.mark.parametrize("times_text, fragment", [
    ("", "at least one prescribed time"),
    ("8:00", "Invalid time"),
    ("08:00, 08:00", "must not repeat"),
    ("23:58", "runs past midnight"),
])
def test_bad_times_are_reported(times_text, fragment):
    errors = validate_entry(entry(times_text=times_text), make_world().owner)
    assert any(fragment in e for e in errors), errors


def test_all_errors_are_reported_together():
    errors = validate_entry(entry(drug="", dose_retyped="x", attested=False, vet_name="Dr. X"), make_world().owner)
    assert len(errors) == 4


def test_one_task_per_time_sharing_one_frozen_medication():
    tasks = build_dose_tasks(entry())
    assert [t.time for t in tasks] == ["08:00", "20:00"]
    assert tasks[0].medication is tasks[1].medication
    med = tasks[0].medication
    assert med.times == ("08:00", "20:00") and med.dose_text == "25 mg" and med.attested
    assert all(t.priority == "high" and t.frequency == "daily" and t.description == "Fakeprofen" for t in tasks)


def test_invalid_entry_adds_nothing(tmp_path):
    world = make_world()
    before = len(world.owner.get_all_tasks())
    result = register_medication(world.owner, entry(attested=False), None, None, tmp_path)
    assert result.errors and result.tasks == []
    assert len(world.owner.get_all_tasks()) == before
    assert not (tmp_path / LOG_NAME).exists()


def test_no_photo_saves_unverified(tmp_path):
    world = make_world()
    result = register_medication(world.owner, entry(), None, None, tmp_path)
    assert not result.errors and len(result.tasks) == 2
    assert all(t.pet_name == "Mochi" and not t.medication.label_verified for t in result.tasks)
    assert not (tmp_path / LOG_NAME).exists()  # no photo, no verification record


def test_photo_without_vision_client_saves_nothing(tmp_path):
    world = make_world()
    result = register_medication(world.owner, entry(), png_bytes(), None, tmp_path)
    assert "ANTHROPIC_API_KEY" in result.errors[0] and result.tasks == []
    assert len(med_tasks(world.owner)) == 1  # only the fixture's Carprofen dose


# --- dose amounts and frequency ---


@pytest.mark.parametrize("owner_dose, label_dose, expected", [
    ("1 tablet (25 mg)", "25 mg", True),
    ("0.5 ml", "5 ml", False),
    ("25 mg", "50 mg", False),
    (".5 ml", "0.5 ml", True),
    ("25 milligrams", "25mg", True),
    ("2 tablets", "1 tablet", False),
    ("25 mg", "1 tablet", False),  # strength on one side only -> not a match
    # Kyle's real-run gap: a count other than 1 on the label must be stated by the owner.
    ("25 mg", "25 mg (1 tablet)", True),  # a single tablet may be implied
    ("25 mg", "25 mg, give 2 tablets", False),
    ("2 tablets (25 mg)", "25 mg tablets, give 2 tablets", True),
    ("25 mg", "25 mg (1 ml)", False),  # 1 ml is never implied
    ("2 tablets (50 mg)", "50 mg", False),  # a count the label doesn't state can't be verified
    ("1 tablet", "25 mg (1 tablet)", False),  # label strength must be stated too
])
def test_dose_amount_matching(owner_dose, label_dose, expected):
    assert doses_match(owner_dose, label_dose) is expected


def test_strength_is_never_multiplied_by_count():
    """Fail safe: 2 x 25 mg tablets is NOT treated as 50 mg; strengths are compared exactly as written."""
    problems = dose_problems("2 tablets (50 mg)", "25 mg tablets, give 2 tablets")
    assert len(problems) == 1 and problems[0].startswith("strength: label says 25 mg, you entered 50 mg")


def test_missing_count_explains_what_to_type():
    problems = dose_problems("25 mg", "25 mg, give 2 tablets")
    assert problems == ['count: label says 2 tablet, you entered none (include the count, e.g. "25 mg, 2 tablets")']


@pytest.mark.parametrize("text, expected", [
    ("give twice daily (every 12 hours)", (2, "day")),
    ("BID", (2, "day")),
    ("once daily", (1, "day")),
    ("daily", (1, "day")),
    ("once a week", (1, "week")),
    ("as directed", None),
    ("once daily every 12 hours", None),  # contradictory -> needs review
])
def test_label_frequency(text, expected):
    assert label_frequency(text) == expected


# --- label comparison ---


def test_matching_label():
    assert compare_label(entry(), MATCHING).result == "match"


@pytest.mark.parametrize("field, value, fragment", [
    ("dose_text", "50 mg (2 tablets)", "dose strength:"),
    ("dose_text", "25 mg, give 2 tablets", "dose count:"),
    ("drug", "Otherprofen", "drug:"),
    ("frequency", "once daily", "frequency:"),
    ("frequency", "as directed", "could not match the label frequency"),
    ("pet_name", "Bean", "pet:"),
])
def test_mismatching_label(field, value, fragment):
    check = compare_label(entry(), replace(MATCHING, **{field: value}))
    assert check.result == "mismatch"
    assert any(fragment in m for m in check.mismatches)


@pytest.mark.parametrize("reading", [
    replace(MATCHING, readable=False),
    replace(MATCHING, dose_text=""),
    replace(MATCHING, dose_text="see bottle"),  # no amount on the label
])
def test_unreadable_label(reading):
    assert compare_label(entry(), reading).result == "unreadable"


# --- register with a stubbed vision client ---


def test_match_saves_label_verified_tasks(tmp_path):
    world = make_world()
    vision = FakeVisionClient(MATCHING)
    result = register_medication(world.owner, entry(), png_bytes(), vision, tmp_path)
    assert not result.errors and result.label_verified
    assert len(result.tasks) == 2 and all(t.medication.label_verified for t in result.tasks)
    assert result.tasks[0].medication.dose_text == "25 mg"  # the owner's text, not the label's "25 mg (1 tablet)"
    assert len(vision.calls) == 1


@pytest.mark.parametrize("vision_result, expected", [
    (replace(MATCHING, dose_text="50 mg (2 tablets)"), "mismatch"),
    (replace(MATCHING, readable=False), "unreadable"),
    (LLMError("Rate limited by the API (429) after retries."), "error"),
])
def test_failed_check_saves_nothing(tmp_path, vision_result, expected):
    world = make_world()
    before = len(world.owner.get_all_tasks())
    result = register_medication(world.owner, entry(), png_bytes(), FakeVisionClient(vision_result), tmp_path)
    assert result.check.result == expected and result.errors and result.tasks == []
    assert len(world.owner.get_all_tasks()) == before
    assert log_records(tmp_path)[0]["result"] == expected


def test_corrupt_image_is_unreadable_without_calling_the_model(tmp_path):
    vision = FakeVisionClient(MATCHING)
    result = register_medication(make_world().owner, entry(), b"not an image", vision, tmp_path)
    assert result.check.result == "unreadable" and vision.calls == []


def test_image_is_downscaled_before_sending():
    out = Image.open(io.BytesIO(prepare_image(png_bytes((4000, 3000)))))
    assert max(out.size) == MAX_IMAGE_EDGE and out.format == "JPEG"


# --- privacy of the verification log ---


def test_record_has_hash_values_consent_and_model(tmp_path):
    raw = png_bytes()
    cross_check_label(entry(attested=True), raw, FakeVisionClient(MATCHING), tmp_path, keep_label_copy=False)
    record = log_records(tmp_path)[0]
    assert record["image_sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["owner_entered"] == {"pet": "Mochi", "drug": "Fakeprofen", "dose_text": "25 mg",
                                       "frequency": "daily", "times": ["08:00", "20:00"]}
    assert record["model_extracted"]["dose_text"] == "25 mg (1 tablet)"
    assert record["result"] == "match" and record["model"] == "fake-vision"
    assert record["consent"] == {"attested": True, "vet_consent": False, "keep_label_copy": False}
    assert record["ts"]


def test_log_never_contains_image_bytes_or_vet_details(tmp_path):
    raw = png_bytes()
    e = entry(vet_name="Dr. Example", vet_phone="555-0199", vet_consent=True)
    cross_check_label(e, raw, FakeVisionClient(MATCHING), tmp_path)
    text = (tmp_path / LOG_NAME).read_text(encoding="utf-8")
    log_bytes = (tmp_path / LOG_NAME).read_bytes()
    assert raw[8:40] not in log_bytes  # PNG body
    assert base64.b64encode(raw).decode()[:40] not in text
    assert "Dr. Example" not in text and "555-0199" not in text
    assert list(tmp_path.iterdir()) == [tmp_path / LOG_NAME]  # nothing else written


def test_label_copy_is_written_only_on_request(tmp_path):
    raw = png_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    path = save_label_copy(raw, sha, tmp_path / "label_copies")
    assert path.name == f"{sha[:16]}.png" and path.read_bytes() == raw


def test_label_copies_folder_is_git_ignored():
    assert "label_copies/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


# --- real client request shape (stub SDK client, no network) ---


def stub_sdk(text: str, stop_reason: str = "end_turn"):
    """Return an object shaped like anthropic.Anthropic whose messages.create returns one text block."""
    sent: list[dict] = []

    def create(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason)

    return SimpleNamespace(messages=SimpleNamespace(create=create), sent=sent)


def test_vision_request_uses_image_block_and_strict_schema(monkeypatch):
    monkeypatch.setenv("PAWPAL_MODEL", "claude-haiku-5-5")
    client = AnthropicVisionClient(client=stub_sdk("{}"))
    request = client.build_request(b"jpeg-bytes")
    image_block, text_block = request["messages"][0]["content"]
    assert image_block["type"] == "image" and image_block["source"]["media_type"] == "image/jpeg"
    assert request["output_config"]["format"] == {"type": "json_schema", "schema": LABEL_SCHEMA}
    assert request["model"] == "claude-haiku-5-5" and "never instructions" in text_block["text"]


def test_vision_client_parses_reply_and_treats_refusal_as_unreadable():
    reply = json.dumps({"drug": "Fakeprofen", "dose_text": "25 mg", "frequency": "BID", "pet_name": "Mochi",
                        "readable": True})
    assert AnthropicVisionClient(client=stub_sdk(reply)).read_label(b"x").dose_text == "25 mg"
    assert AnthropicVisionClient(client=stub_sdk(reply, "refusal")).read_label(b"x").readable is False


@pytest.mark.parametrize("text", ["not json", "[]", '{"drug": "x"}'])
def test_malformed_label_json_is_unreadable(text):
    assert parse_label_json(text).readable is False


# --- safety regressions: medication stays human-only and immutable ---


def test_medication_is_still_frozen():
    med = build_dose_tasks(entry())[0].medication
    with pytest.raises(FrozenInstanceError):
        med.dose_text = "50 mg"
    with pytest.raises(FrozenInstanceError):
        med.label_verified = True


def test_no_agent_tool_can_create_or_edit_medication():
    for schema in TOOL_SCHEMAS:
        params = set(schema["input_schema"].get("properties", {}))
        assert not params & {"medication", "dose", "dose_text", "drug", "label_verified"}, schema["name"]
        assert "medication" not in schema["name"] and "dose" not in schema["name"]


@pytest.mark.parametrize("extra", [{"medication": {"drug": "X"}}, {"dose_text": "50 mg"}])
def test_change_parser_rejects_medication_fields(extra):
    add = {"kind": "add", "task_id": "abc12345", "pet_name": "Mochi", "description": "Walk", "time": "09:00",
           "duration_minutes": 10, "due_date": TODAY.isoformat(), "rationale": "x"}
    parse_change(add)  # valid without the extra field
    with pytest.raises(ValueError):
        parse_change({**add, **extra})


def test_medication_type_is_unchanged():
    assert Medication.__dataclass_params__.frozen
