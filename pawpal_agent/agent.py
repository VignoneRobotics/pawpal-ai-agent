"""The plan -> act -> verify loop, and apply_proposal (the only path that mutates real state)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from pawpal_system import Owner, Scheduler

from .guardrails import check_request, redact_dosing, score_confidence, verify_proposal
from .llm import LLMClient
from .logging_utils import TraceLogger, truncate
from .prompts import BASELINE_PROMPT, PROMPTS, resolve_prompt_name
from .proposal import FinalReply, Proposal, apply_change, change_kind, parse_final_reply
from .tools import TOOL_SCHEMAS, ToolSession, tasks_for_day

MAX_TOOL_TURNS = 6  # per attempt
MAX_VERIFIER_RETRIES = 2  # so at most 3 attempts
FAILED_SUMMARY = "I couldn't find a schedule change that passes the safety checks. Nothing was changed."
MODEL_REFUSAL = "The assistant declined this request. Nothing was changed."

# Kept for existing imports; prompt variants live in prompts.py (each constant, so it can be prompt-cached).
SYSTEM_PROMPT = BASELINE_PROMPT


@dataclass
class Usage:
    """Per-request cost and effort counters (used by eval.py later)."""

    llm_calls: int = 0
    tool_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class AgentResult:
    """Everything a caller needs: the proposal to show the human, the trace, and usage counters."""

    proposal: Proposal
    trace: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class AttemptFailed(Exception):
    """An attempt ended without a usable final reply (bad JSON or too many tool turns)."""

    def __init__(self, errors: list[str]) -> None:
        """Store the errors to feed back to the model."""
        super().__init__("; ".join(errors))
        self.errors = errors


class ModelRefused(Exception):
    """The model's safety classifier declined the request (stop_reason "refusal")."""


def build_context(owner: Owner, day: date, today: date, request: str) -> str:
    """Return the first user message: only the planning day's tasks plus owner limits (no vet info)."""
    pets = ", ".join(f"{p.name} ({p.species})" for p in owner.pets) or "none"
    windows = ", ".join(owner.availability_windows) or "all day"
    tasks = json.dumps(tasks_for_day(owner, day))
    return (
        f"Today: {today.isoformat()}. Planning day: {day.isoformat()}.\n"
        f"Pets: {pets}.\n"
        f"Daily budget: {owner.available_minutes} min. Availability windows: {windows}.\n"
        f"Tasks on {day.isoformat()}: {tasks}\n\n"
        f"Owner request: {request}"
    )


def _add_user_text(messages: list[dict], text: str) -> None:
    """Append text as a user turn, merging into the last user turn so roles keep alternating."""
    if messages and messages[-1]["role"] == "user":
        last = messages[-1]
        if isinstance(last["content"], str):
            last["content"] = [{"type": "text", "text": last["content"]}]
        last["content"].append({"type": "text", "text": text})
    else:
        messages.append({"role": "user", "content": text})


def _feedback(errors: list[str]) -> str:
    """Return the retry message listing why the last attempt failed."""
    lines = "\n".join(f"- {e}" for e in errors)
    return (
        "Your proposal was rejected:\n"
        f"{lines}\n"
        "The schedule has been reset to the original. Propose again with the tools, "
        "then reply with only the final JSON."
    )


def _run_attempt(
    llm: LLMClient,
    session: ToolSession,
    messages: list[dict],
    trace: TraceLogger,
    usage: Usage,
    attempt: int,
    system_prompt: str = BASELINE_PROMPT,
) -> FinalReply:
    """Let the model call tools until it gives a final reply; raise AttemptFailed on bad JSON or too many turns."""
    tool_turns = 0
    while True:
        response = llm.complete(system_prompt, messages, TOOL_SCHEMAS)
        usage.llm_calls += 1
        usage.input_tokens += response.input_tokens
        usage.output_tokens += response.output_tokens
        usage.cache_read_tokens += response.cache_read_tokens
        usage.cache_write_tokens += response.cache_write_tokens
        trace.log(
            "llm_call",
            attempt=attempt,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cache_read_tokens=response.cache_read_tokens,
            cache_write_tokens=response.cache_write_tokens,
            stop_reason=response.stop_reason,
            tool_calls=[c.name for c in response.tool_calls],
        )

        if response.stop_reason == "refusal":
            raise ModelRefused()

        if response.tool_calls:
            if response.assistant_content:  # real model: send its blocks (incl. thinking) back unchanged
                assistant = list(response.assistant_content)
            else:
                assistant = [{"type": "text", "text": response.text}] if response.text else []
                assistant += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in response.tool_calls
                ]
            messages.append({"role": "assistant", "content": assistant})

            if tool_turns >= MAX_TOOL_TURNS:  # answer the calls (API needs it) but don't run them
                limit = json.dumps({"error": "tool turn limit reached"})
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": c.id, "content": limit, "is_error": True}
                        for c in response.tool_calls
                    ],
                })
                raise AttemptFailed([f"used more than {MAX_TOOL_TURNS} tool turns; plan with fewer calls"])

            tool_turns += 1
            usage.tool_turns += 1
            results = []
            for c in response.tool_calls:
                content, is_error = session.dispatch(c.name, c.arguments)
                trace.log("tool_result", attempt=attempt, name=c.name, args=c.arguments,
                          is_error=is_error, result=truncate(content))
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": content, "is_error": is_error})
            messages.append({"role": "user", "content": results})
            continue

        if response.assistant_content:
            messages.append({"role": "assistant", "content": list(response.assistant_content)})
        else:
            messages.append({"role": "assistant", "content": response.text or "(empty)"})
        if response.stop_reason == "max_tokens":
            raise AttemptFailed(["your reply was cut off (max_tokens); use fewer words"])
        try:
            return parse_final_reply(response.text, len(session.changes))
        except ValueError as err:
            raise AttemptFailed([f"final reply: {err}"]) from err


def run_agent(
    request: str,
    owner: Owner,
    today: date,
    llm: LLMClient,
    day: date | None = None,
    log_dir: str | Path | None = "logs",
    prompt: str | None = None,
) -> AgentResult:
    """Guard the request, run the tool loop on a sandbox, verify (with up to 2 retries), and return a Proposal.

    prompt picks the system prompt variant ("baseline" or "fewshot"); None uses PAWPAL_PROMPT, else baseline.
    """
    day = day or today
    prompt_name = resolve_prompt_name(prompt)
    system_prompt = PROMPTS[prompt_name]
    trace = TraceLogger(log_dir)
    usage = Usage()
    drug_names = tuple(sorted({t.medication.drug for t in owner.get_all_tasks() if t.medication}))
    trace.log("request", request=request, day=day.isoformat(), today=today.isoformat(), prompt=prompt_name)

    # Medication task descriptions are excluded so a drug-named task can never exempt a dosing question.
    task_names = tuple(sorted({t.description for t in owner.get_all_tasks() if t.medication is None}))
    guard = check_request(request, tuple(p.name for p in owner.pets), drug_names, task_names)
    trace.log("guard", allowed=guard.allowed, category=guard.category)
    if not guard.allowed:
        proposal = Proposal(request, day, refusal=guard.message, verified=True, model_confidence=1.0, confidence=1.0)
        trace.log("final", verified=True, refusal=guard.category, changes=[], **vars(usage))
        return AgentResult(proposal, trace.events, usage)

    session = ToolSession(owner, today)
    messages: list[dict] = [{"role": "user", "content": build_context(owner, day, today, request)}]
    errors: list[str] = []
    for attempt in range(MAX_VERIFIER_RETRIES + 1):
        try:
            reply = _run_attempt(llm, session, messages, trace, usage, attempt, system_prompt)
        except ModelRefused:
            proposal = Proposal(request, day, refusal=MODEL_REFUSAL, verified=True, retries=attempt)
            trace.log("final", verified=True, refusal="model", changes=[], **vars(usage))
            return AgentResult(proposal, trace.events, usage)
        except AttemptFailed as failed:
            errors = failed.errors
            trace.log("attempt_failed", attempt=attempt, errors=errors)
        else:
            reply, flagged = redact_dosing(reply, drug_names, has_changes=bool(session.changes))
            if flagged:  # originals kept only in the git-ignored trace, to tune the detector
                trace.log("output_guard", attempt=attempt, redacted=True, flagged=flagged)
            proposal = Proposal(
                request,
                day,
                changes=list(session.changes),
                summary=reply.summary,
                model_confidence=reply.confidence,
                notes=reply.notes,
                refusal=reply.refusal,
                retries=attempt,
            )
            result = verify_proposal(owner, proposal, today)
            trace.log("verification", attempt=attempt, ok=result.ok, errors=result.errors)
            if result.ok:
                proposal.verified = True
                proposal.confidence = score_confidence(reply.confidence, attempt, True)
                trace.log("final", verified=True, confidence=proposal.confidence, retries=attempt,
                          changes=[f"{change_kind(c)} {c.task_id}" for c in proposal.changes], **vars(usage))
                return AgentResult(proposal, trace.events, usage)
            errors = result.errors

        if attempt < MAX_VERIFIER_RETRIES:
            _add_user_text(messages, _feedback(errors))
            session.reset()

    proposal = Proposal(request, day, summary=FAILED_SUMMARY, errors=errors, retries=MAX_VERIFIER_RETRIES)
    trace.log("final", verified=False, confidence=0.0, retries=MAX_VERIFIER_RETRIES, errors=errors, **vars(usage))
    return AgentResult(proposal, trace.events, usage)


def apply_proposal(owner: Owner, proposal: Proposal, today: date) -> list[str]:
    """Apply an approved, verified proposal to the real owner; re-verify first and raise ValueError without changes if it fails."""
    if proposal.refusal:
        raise ValueError("A refusal has no changes to apply")
    if not proposal.verified:
        raise ValueError("Only verified proposals can be applied")
    result = verify_proposal(owner, proposal, today)  # the schedule may have changed since the proposal
    if not result.ok:
        raise ValueError("Proposal no longer passes verification: " + "; ".join(result.errors))
    scheduler = Scheduler(owner)
    return [apply_change(scheduler, change, today) for change in proposal.changes]
