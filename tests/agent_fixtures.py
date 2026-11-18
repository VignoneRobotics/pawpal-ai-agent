"""Shared builders for the agent tests (plain functions, imported by the test modules)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta

from pawpal_agent.llm import LLMResponse, text_reply
from pawpal_agent.proposal import Proposal
from pawpal_system import Medication, Owner, Pet, Task

TODAY = date(2026, 1, 5)
TOMORROW = TODAY + timedelta(days=1)
YESTERDAY = TODAY - timedelta(days=1)


@dataclass
class World:
    """An owner with two pets and a few known tasks."""

    owner: Owner
    mochi: Pet
    bean: Pet
    walk: Task  # Mochi 08:00-08:30 high
    feed: Task  # Bean 09:00-09:10 medium, daily
    play: Task  # Mochi 10:00-10:30 low
    dose: Task  # Mochi Carprofen 20:00-20:05 high, daily (prescribed 08:00 and 20:00)


def make_med() -> Medication:
    """Return a twice-daily Carprofen prescription with vet details (consented)."""
    return Medication(
        drug="Carprofen",
        dose_text="25 mg tablet",
        times=("08:00", "20:00"),
        vet_name="Dr. Lee",
        vet_phone="555-0100",
        vet_consent=True,
        attested=True,
    )


def make_world(budget: int = 480, windows: list[str] | None = None) -> World:
    """Build the standard test world for TODAY."""
    owner = Owner("Jordan", available_minutes=budget, availability_windows=windows or [])
    mochi, bean = Pet("Mochi", "dog"), Pet("Bean", "cat")
    owner.add_pet(mochi)
    owner.add_pet(bean)
    walk = Task("Morning walk", "08:00", 30, "high", due_date=TODAY)
    feed = Task("Feed Bean", "09:00", 10, "medium", "daily", due_date=TODAY)
    play = Task("Play", "10:00", 30, "low", due_date=TODAY)
    dose = Task.medication_dose(make_med(), "20:00", 5, TODAY)
    mochi.add_task(walk)
    bean.add_task(feed)
    mochi.add_task(play)
    mochi.add_task(dose)
    return World(owner, mochi, bean, walk, feed, play, dose)


def final(summary: str = "Done.", confidence: float = 0.8, notes: list[str] | None = None,
          refusal: str | None = None) -> LLMResponse:
    """Build the model's final strict-JSON reply."""
    return text_reply(json.dumps(
        {"summary": summary, "confidence": confidence, "notes": notes or [], "refusal": refusal}
    ))


def proposal_of(*changes) -> Proposal:
    """Wrap changes in a Proposal for TODAY."""
    return Proposal(request="test", day=TODAY, changes=list(changes))
