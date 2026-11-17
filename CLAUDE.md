# CLAUDE.md — PawPal+ Agent (AI110 Project 4) context

This file is the shared plan for every AI session on this repo. Read it fully before making changes.
Sections below "Base system reference" describe the Module 2 code and still apply unless a Project 4 rule overrides them.

## Project
PawPal+ Agent extends the Module 2 PawPal+ scheduler (repo `ai110-module2show-pawpal-starter`) into an **applied AI system**:
a natural-language scheduling assistant. An LLM (Claude Haiku 5.5) plans and **proposes** schedule changes by calling the
existing `Scheduler` as tools; deterministic code **verifies** every proposal; the human **approves** before anything is saved.
Core idea: *the AI proposes, the scheduler verifies, the human decides.*

## How we work
- Kyle is lead architect. Propose, then implement; Kyle reviews every diff and makes every commit and push. **Never commit or push** unless Kyle explicitly asks.
- One phase per chat session. Stay inside the current phase's scope. Plan first (Plan mode), then edit.
- Do **not** write `model_card.md` reflection content in Kyle's voice unless he asks; then draft from facts only and he edits it.
- If a suggestion deviates from this design, say so explicitly so Kyle can accept or reject it (feeds the model card's "accepted / rejected suggestion").
- Prefer readable over clever. If a "Pythonic" version is harder to follow, offer both.
- **Never run destructive commands** (`rm`, deleting `data.json`, `logs/`, git resets) without asking first. (An earlier agent deleted `data.json` unasked.)
- Record notable AI interactions (accepted/rejected suggestions, agent mistakes) in the "Change log" at the bottom as they happen.

## Files the grader expects (Project 4)
| File | Purpose |
|---|---|
| `README.md` | Names the base project + 2-3 sentence summary; title/summary; architecture overview; setup; 2-3 sample interactions (input -> output in code blocks); design decisions; testing summary; short reflection |
| `model_card.md` | **Graded reflection**: limitations/biases, misuse + prevention, what surprised us in testing, one AI suggestion accepted + one rejected/revised, AI collaboration, evaluation results tables |
| `diagrams/architecture.mmd` | Mermaid system diagram: components, data flow (input -> process -> output), where human/tests check AI. Must match the code |
| `diagrams/uml.mmd` | Class diagram, kept in sync with code |
| `ai_interactions.md` | Stretch: agent reasoning traces (embedded or linked committed log) |
| `assets/` | Images (architecture PNG export, sample prescription label) |
| `agent_demo.py` | CLI end-to-end demo (2-3 inputs) |
| `eval.py` | Stretch: evaluation harness over predefined inputs, prints a pass/fail + confidence summary |
| `app.py` | Streamlit UI incl. agent chat + approval |
| `tests/` | pytest; must pass with **no network and no API key** |

## Architecture (Project 4)
```
User request (Streamlit / CLI)
  -> input_guard          (scope check, dosing-question refusal)
  -> Agent loop (Haiku)   (plans, calls tools on a SANDBOX copy of the Owner)
  -> Proposal             (list of structured changes + rationale + confidence)
  -> verifier             (deterministic checks; on failure: errors fed back, max 2 retries)
  -> Human approval       (before/after preview; Approve / Reject)
  -> apply_proposal       (only path that mutates real state) -> data.json
  -> audit/trace logs
```
Planned modules (names may be adjusted in each phase's plan):
- `pawpal_system.py` — logic layer (extended in Phase 1). No LLM code here.
- `pawpal_agent/llm.py` — `LLMClient` protocol; `AnthropicClient` (real) and `FakeLLM` (scripted, for tests).
- `pawpal_agent/tools.py` — tool JSON schemas + dispatch to `Scheduler` methods on the sandbox Owner.
- `pawpal_agent/agent.py` — the plan -> act -> verify loop; returns `Proposal` + trace.
- `pawpal_agent/guardrails.py` — input guard, proposal verifier, confidence scoring.
- `pawpal_agent/medication.py` — medication rules, label cross-check (vision), image hashing.
- `pawpal_agent/logging_utils.py` — JSONL trace + verification logs.

## Agent rules
- Model: `claude-haiku-5-5`, set via env var `PAWPAL_MODEL` (default Haiku). Sonnet 5.5 only if evals show Haiku failing.
- **The agent never mutates real state.** Tools run on a deep copy; only `apply_proposal()` (called after human approval) writes.
- Tools available to the agent (initial set): `list_tasks`, `find_conflicts`, `next_available_slot`, `build_daily_plan`, `propose_move_task`, `propose_add_task`, `propose_complete_task`. No delete tool in v1.
- Output is a **strict JSON proposal** (schema in `agent.py`): changes, one-line rationale per change, confidence 0-1, and any refusals/notes.
- Verifier rejects a proposal if any change: fails `Task` validation; creates a new conflict; exceeds the owner's budget/windows; touches a medication `dose`; breaks a medication's dosing interval; references an unknown task/pet.
- Out-of-scope requests (not pet-care scheduling) -> polite refusal, no tool calls.
- The agent **never contacts anyone** (no email/phone/web tools).

## Haiku optimization rules
- Few tools, short clear names and descriptions; small structured outputs; cap `max_tokens`; cap the loop (max 6 tool turns, max 2 verifier retries).
- Send only the relevant day's tasks as context, not the whole history.
- Keep the system prompt stable so it can use **prompt caching**.
- Downscale images before sending (long edge ~1568 px) to limit visual tokens.
- Log tokens in/out, tool turns and retries per request (used by `eval.py` to compare Haiku vs Sonnet if needed).

## Medication safety rules (decided with Kyle)
- The agent must **never calculate, convert, adjust, or suggest a dose**. Dosing questions get a scripted refusal ("I can't give dosing advice. Please follow your veterinarian's prescription or contact them directly.").
- Medication tasks are created only through the **human form**: drug name, dose text, frequency/times, optional vet name/clinic/phone.
- `dose` is **read-only to the agent** (no tool can set or change it); the verifier rejects any proposal that changes it.
- Form confirmation: owner **re-types the dose** (must match exactly) + checks the attestation box "This matches my veterinarian's prescription label." Vet contact info requires its own consent checkbox before it is stored.
- **Optional label photo cross-check** (Option B): vision model reads drug, dose, frequency and pet name from the label; compared to what the owner typed. Match -> "Label-verified" badge. Mismatch or unreadable -> task is flagged/blocked for human review. The photo is a **check, never the source** of the dose (fails safe). This verifies consistency, not authenticity.
- Moving a med task is allowed only if the dosing interval still holds (e.g., twice daily stays ~12h apart, within tolerance).

## Privacy rules
- `ANTHROPIC_API_KEY` lives in `.env` (git-ignored). Commit `.env.example` with placeholder only. Never print or log the key.
- Prescription photos are sent to the API for the cross-check but **never written to disk, data.json, or logs** by default.
- Verification log stores: timestamp, owner-entered values, model-extracted values, match result, model id, consent flags, and the **SHA-256 hash** of the image (proves which image was checked without keeping it).
- Optional opt-in "keep a local copy of the label" saves to a git-ignored folder.
- `data.json`, `logs/` (runtime) and `.env` are git-ignored. Only curated **sample** logs (fake data) are committed for grading.
- Only the minimum needed context is sent to the API (relevant day's tasks, pet names; no vet contact info).

## Testing rules
- `python -m pytest` must pass with **no network and no API key**: all agent tests use `FakeLLM` with scripted responses.
- Logic layer still never calls `date.today()`; callers pass `today`.
- `eval.py` uses the real API, runs a fixed list of requests (normal, conflict, overload, out-of-scope, dosing question, dose-change attempt, label match/mismatch), and prints pass/fail, confidence, tokens and retries. Output saved to a committed `.txt` log for the README.

## Project 4 phase plan
0. **Setup** — new repo, venv, assets/ ✅
1. **Scheduler upgrades (deterministic)** — overlap-aware daily plan (an overlapping lower-priority task is skipped with reason "overlaps X"; `build_daily_plan` returns reasons); owner availability windows used by both the plan and `next_available_slot`; `overdue_tasks(today)`; `move_task(task_id, new_time, new_date=None)` with validation; `Medication` data on `Task` (dose read-only to agent). Tests for each.
2. **Agent core (offline)** — `LLMClient`/`FakeLLM`, tools, proposal schema, verifier, retry loop, input guard, traces. Tests with FakeLLM only.
3. **Real Haiku wiring + CLI** — `AnthropicClient`, `.env`/`.env.example`, `agent_demo.py` with 3 sample requests; save outputs for README.
4. **Streamlit integration** — chat box, proposal preview (before/after), Approve/Reject, confidence + rationale shown, overdue view.
5. **Medication flow** — form with re-type + attestation + vet consent; optional label cross-check, hashing, verification log; fake `SAMPLE` label in `assets/`.
6. **Evaluation + specialization** — `eval.py` summary; few-shot vs baseline prompt comparison recorded in `model_card.md`.
7. **Docs** — README rewrite, `diagrams/architecture.mmd`, `uml.mmd` sync, `model_card.md`, traces in `ai_interactions.md`.
8. *(Optional)* **RAG** — pet-care guideline lookup tool; only if time remains.

## Change log (Project 4 — facts for model_card.md)
- (add entries as decisions/AI suggestions are accepted or rejected)

---

# Base system reference (Module 2 — still binding unless overridden above)

## Conventions
- Python 3.14. `@dataclass` for `Task` and `Pet` (Owner may be a dataclass too).
- 1-line docstrings on every method; type hints everywhere.
- Times are zero-padded `"HH:MM"` strings; convert with a helper (`_to_minutes`) for math. Dates are `datetime.date`.
- Priority is a string: `"low" | "medium" | "high"`, ranked via `PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}`.
- Frequency is a string: `"once" | "daily" | "weekly"`, mapped via `FREQUENCY_DAYS = {"daily": 1, "weekly": 7}`.
- Conflict detection returns warning strings; it never raises.
- **ASCII-only text in `pawpal_system.py`** (warnings, explanations). Emoji/colors (SF4) live only in the CLI/Streamlit display layer, never in the logic layer. Redirected Windows output garbles non-ASCII.
- No new dependencies without asking. Approved: `tabulate[widechars]` (SF4; `wcwidth` aligns emoji columns).
- `main.py` calls `sys.stdout.reconfigure(encoding="utf-8")` because piped Windows stdout is cp1252 and cannot encode emoji.

- Project 4 additions: `anthropic` and `python-dotenv` approved as new dependencies; Pillow (already installed via Streamlit) for image downscaling.

## Design (approved decisions A–L; final API below)
**Task** (dataclass): `task_id` (uuid4 hex[:8], default factory), `description`, `time`, `duration_minutes`, `priority`, `frequency`, `due_date`, `completed=False`, `pet_name=""`.
Methods: `__post_init__` validation (zero-padded HH:MM 00:00-23:59, known priority/frequency, positive duration, no running past midnight), `mark_complete()`, `next_occurrence(today=None) -> Task | None` (see decision Q).

**Pet** (dataclass): `name`, `species`, `age`, `tasks: list[Task]`.
Methods: `add_task(task)` (stamps `task.pet_name = self.name`), `remove_task(task_id)`, `get_tasks()`.

**Owner**: `name`, `available_minutes` (daily time budget), `pets: list[Pet]`.
Methods: `add_pet(pet)` (raises `ValueError` on a duplicate pet name, case-insensitive), `get_pet(name) -> Pet | None`, `get_all_tasks()` (every task across every pet), `find_task(task_id) -> tuple[Pet, Task] | None` (task plus its owning pet), `save_to_json(path)`, `@classmethod load_from_json(path) -> Owner` (SF2; dates as ISO strings; rebuilds via `Task(**d)` so validation runs).

**Scheduler**: holds `owner`. **Always reads tasks via `owner.get_all_tasks()`**, never from a single pet (rubric: multi-pet).
Every method that takes `tasks` treats it as optional: `None` means "all owner tasks", so calls compose, e.g. `sort_by_time(filter_tasks(pet_name="Mochi"))`.
- `sort_by_time(tasks)` — `sorted(..., key=lambda t: (t.due_date, t.time, PRIORITY_RANK[t.priority], t.description))` — date first so recurring next-day copies sort chronologically; deterministic tie-break. Same-day order is unchanged.
- `sort_by_priority(tasks)` — key `(due_date, PRIORITY_RANK, time, description)`: priority first within each day (SF3).
- `filter_tasks(tasks, pet_name=None, completed=None)`
- `find_conflict_pairs(tasks)` — `(earlier, later)` overlapping pairs via `itertools.combinations`; single source of the overlap rule (used by `detect_conflicts` and `app.py`).
- `detect_conflicts(tasks)` — warning strings built from `find_conflict_pairs`; labels same-pet vs cross-pet.
- `next_available_slot(duration_minutes, day, window_start="07:00", window_end="22:00") -> str | None` — first gap between pending tasks on `day` where the duration fits (SF1).
- `mark_task_complete(task_id, today=None)` — looks up via `owner.find_task`; marks complete, **keeps the completed task** (history), and if recurring adds `next_occurrence()` to the same pet; returns the new task or None. If the task is already completed (or not found), do nothing and return None (no duplicate next occurrence).
- `build_daily_plan(day)` — only tasks due on `day` and not completed; greedy first-fit: order by priority, then time; if a task doesn't fit the remaining budget, **skip it and continue** (never `break`). Returns `(planned, skipped)`. It does **not** drop overlapping tasks — conflicts are warnings only (callers run `detect_conflicts(planned)`).
- `explain_plan(planned, skipped)` — human-readable reasons. Recomputes from `self.owner.available_minutes` by walking `planned` in order. The only skip reason is "didn't fit the remaining budget" (not-due tasks are never in `skipped`), so no extra parameters are needed.

Relationships: Owner 1 → * Pet (aggregation); Pet 1 → * Task (composition); Scheduler → Owner (uses).

Known tradeoffs (for reflection): plan and conflict checks are separate (an overlapping pair can both be planned, with a warning); greedy plan can miss a better combination; `pet_name` duplicated on Task for simple filtering; time stored as text, not datetime (rejected a single `datetime` field to follow the assignment's HH:MM sorting); task list grows because completed tasks are kept.

## Design change log (facts for reflection 1b — Kyle writes the reflection)
- After the Phase 1 skeleton review (Sonnet 5.5): **accepted** adding `Owner.find_task` (scheduler needed the owning pet); **accepted** skip-and-continue for the budget; **chose** warn-only conflicts; **partly rejected** the suggestion that `explain_plan` needs the day or that `build_daily_plan` must return reasons — with warn-only conflicts the only skip reason is the budget, so it can be recomputed and signatures stay unchanged.

## Module 2 algorithm decisions (approved M–S + review items)
- **M Conflict algorithm:** simple pairwise double loop (`for i ... for j in range(i+1, n)`), each pair checked once, no duplicate warnings. A sort-and-sweep version is evaluated in Step 5 as the "simplify/performance" comparison; keep whichever Kyle chooses.
- **N Overlap rule:** `a_start < b_end and b_start < a_end` (minutes via `_to_minutes`). Back-to-back (end == start) is **not** a conflict.
- **O Scope:** only compare tasks with the **same `due_date`** that are **not completed**.
- **P Warning format:** `Conflict (same pet: Mochi): Walk 08:00-08:30 overlaps Bath 08:15-08:45` / `Conflict (cross-pet: Mochi, Bean): ...`. Plain ASCII hyphen only — an en dash was garbled (`û`) when output was redirected in Windows PowerShell.
- **Q Recurrence (revised after SF5 model comparison):** `next_occurrence(today: date | None = None)` uses `dataclasses.replace(self, task_id=<new uuid>, completed=False, due_date=next_due)`. Must override `task_id`. Returns None for "once". `next_due = due_date + timedelta(days=FREQUENCY_DAYS[freq])`; if `today` is given, `while next_due <= today: next_due += step` (skip missed occurrences, keep the original cadence/weekday, strictly after today). **Never call `date.today()` in pawpal_system.py** — callers pass it (`app.py` passes `date.today()`, `main.py` its demo day, tests a fixed date). Omitting `today` keeps the original behavior. `Scheduler.mark_task_complete(task_id, today=None)` passes `today` through.
- **R Filtering:** `None` = don't filter on that field; filters combine; `pet_name` match is case-insensitive (same as `Owner.get_pet`).
- **S Scope:** don't touch `app.py` in Phase 4. `main.py` uses `sort_by_time` (remove the inline `sorted()`), shows out-of-order tasks, a filtered view, a completed daily task with its next-day copy, and same-pet + cross-pet conflict warnings.
- **Display order:** `build_daily_plan` still *selects* by priority, but callers *display* `sort_by_time(planned)`; `explain_plan` output stays in selection order. No signature change.
- **Conflicts on the plan:** callers run `detect_conflicts(planned)` (CLI in Phase 4, `st.warning` in Phase 6).
- **Sort-key fix (after Step 3):** recurring copies exposed that a time-only key put tomorrow's 07:00 before today's 08:00, so `due_date` became the first key. CLI `format_task` shows due date and done/pending so same-time tasks on different days are distinguishable.
- **Rejected:** "overdue" filter (scope creep, signature change) — candidate for reflection 4b "next steps". Optimal/knapsack planning — greedy is the documented tradeoff.


## Commands
```
py -3.14 -m venv .venv          # once (python is not on PATH; use the py launcher)
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
python main.py                  # Module 2 CLI demo
python agent_demo.py            # Project 4 agent demo (Phase 3+)
python eval.py                  # evaluation harness (Phase 6+, needs API key)
python -m pytest                # offline tests
streamlit run app.py
```
