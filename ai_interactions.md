# AI Interactions Log

> Project 4 (PawPal+ Agent): agent reasoning traces are below. The Module 2 stretch-feature sections (SF7 agent workflow, SF11 prompt comparison) follow unchanged.

---

## Agent reasoning traces (Project 4)

Every request writes one JSONL event per step (`request`, `guard`, `llm_call`, `tool_result`, `verification`, `output_guard`, `final`, `human_decision`) to `logs/agent_trace.jsonl`, which is git-ignored. The curated traces committed for grading come from the evaluation runs on a fake household: [`assets/sample_traces/baseline/`](assets/sample_traces/baseline/) and [`assets/sample_traces/fewshot/`](assets/sample_traces/fewshot/), one file per case and repeat (`<nn>_<case>_r<i>.jsonl`).

### Trace 1: the verifier rejects the first plan, and the agent retries (live UI run)

A conflict-fix request for Mochi's bath in the Streamlit app. The raw events are in the git-ignored `logs/`, so this is the Developer-mode trace view, also shown in [screenshot 05](assets/screenshots/05_agent_trace.png):

```
Guard: allowed
[attempt 0] find_conflicts(day='2026-11-17')
[attempt 0] list_tasks(day='2026-11-17', pet_name='Mochi')
[attempt 0] next_available_slot(day='2026-11-17', duration_minutes=30, ...)
[attempt 0] propose_move_task(task_id='17c68f5b', new_time='16:00')
[attempt 0] verification failed: Laser play would be pushed out of the plan (skipped: budget)
[attempt 1] build_daily_plan(day='2026-11-17')
[attempt 1] find_conflicts(day='2026-11-17')
[attempt 1] propose_move_task(task_id='17c68f5b', new_date='2026-11-18', new_time='16:00')
[attempt 1] verification passed
Usage: 7 LLM calls, 5 tool turns, tokens in 31898 (cache read 25411), out 2417
```

**What it shows:** Haiku's first fix cleared the overlap but used up the time budget, so lower-priority Laser play would have been silently skipped. The deterministic verifier caught that and fed the error back. On attempt 1 the model changed strategy: it called `build_daily_plan` first to see the budget, then moved the bath to tomorrow. The final confidence dropped from the model's 0.70 to 0.59 (−15% for one retry), and the approved schedule shows the bath on 2026-11-18 ([screenshot 06b](assets/screenshots/06b_approved_schedule.png)). Prompt caching covered 25,411 of the 31,898 input tokens.

### Trace 2: an out-of-tolerance medication move ends in a verified no-change (eval, fewshot)

Raw file: [`assets/sample_traces/fewshot/04_med_move_too_far_r1.jsonl`](assets/sample_traces/fewshot/04_med_move_too_far_r1.jsonl)

```jsonl
{"ts": "2026-11-18T06:10:59+00:00", "request_id": "4c02b94deca3", "event": "request", "request": "Move Mochi's evening Carprofen dose 2 hours later.", "day": "2026-03-02", "today": "2026-03-02", "prompt": "fewshot"}
{"ts": "2026-11-18T06:10:59+00:00", "request_id": "4c02b94deca3", "event": "guard", "allowed": true, "category": "ok"}
{"ts": "2026-11-18T06:11:02+00:00", "request_id": "4c02b94deca3", "event": "llm_call", "attempt": 0, "input_tokens": 3607, "output_tokens": 415, "cache_read_tokens": 2599, "cache_write_tokens": 1004, "stop_reason": "end_turn", "tool_calls": []}
{"ts": "2026-11-18T06:11:02+00:00", "request_id": "4c02b94deca3", "event": "verification", "attempt": 0, "ok": true, "errors": []}
{"ts": "2026-11-18T06:11:02+00:00", "request_id": "4c02b94deca3", "event": "final", "verified": true, "confidence": 0.9, "retries": 0, "changes": [], "llm_calls": 1, "tool_turns": 0, "input_tokens": 3607, "output_tokens": 415, "cache_read_tokens": 2599, "cache_write_tokens": 1004}
```

**What it shows:** the request asks to move the 20:00 Carprofen dose 2 hours later, outside the ±60-minute dosing window. The model answered in a single call with no tool calls and no change. The verifier confirmed an empty proposal (`"changes": []`), so nothing could reach the schedule. With the baseline prompt, the same case ended in a model refusal in 2 of 3 runs. Few-shot gave this verified no-change in 3 of 3 runs, which is one reason it became the default.

---

## Agent Workflow (SF7)

> Document your experience using an AI agent (e.g., Cursor Agent, Claude, Copilot) to make multi-step changes autonomously.

**What task did you give the agent?**

Add a "next available slot" feature (SF1) in agent mode: a `Scheduler.next_available_slot(duration, day, window_start, window_end)` method that walks the gaps between a day's pending tasks and returns the first start time where the task fits, wired into the Streamlit UI, the CLI demo, and the test suite. I approved its plan in auto mode so it could edit several files without stopping.

**What did the agent do?**

- **Files changed:** `pawpal_system.py` (new method plus a time-formatting helper), `app.py` (a "Find a free slot" input and a 💡 suggestion under each conflict warning), `main.py` (a demo section), and `tests/test_pawpal.py` (8 new tests: empty day, a gap between tasks, an exact back-to-back fit, no room, ignoring completed tasks and other days, and the window end).
- **Commands run:** `python main.py`, `python -m pytest` (73 passed), a headless Streamlit check, `git status`, and `rm -f data.json`.

**What did you have to verify or fix manually?**

- I reviewed every diff in the Source Control tab, re-ran the tests and the demo myself, and checked the free-slot suggestions in the browser.
- The agent deleted my `data.json` with `rm -f` without checking what was in it. It only held test data, but it showed me that auto mode can take destructive actions I didn't ask for, so I review its command list as well as the diffs.
- The agent duplicated the overlap logic inside `app.py` to find which tasks conflict. I had it move that into the logic layer instead (`find_conflict_pairs`), so the UI and the scheduler can't drift apart.

---

## Prompt Comparison (SF11)

> Compare two different prompts (or two different models) on the same task.

| | Option A | Option B |
|-|----------|----------|
| **Model / tool used** | Claude Opus 5.5 (VS Code extension, Plan mode) | Claude Sonnet 5.5 (VS Code extension, Plan mode) |
| **Prompt** | Identical for both: "Read CLAUDE.md and pawpal_system.py. Problem: `Task.next_occurrence` sets the next due date to `due_date + FREQUENCY_DAYS`. If a daily or weekly task is completed weeks late, the new copy is already in the past. Propose a fix (policy + code for `next_occurrence`, plus any tests). Explain the tradeoffs. Don't edit files." | (same) |
| **Response summary** | Policy: skip missed dates and keep the original schedule. Added an optional `today` parameter; when it's left out, the old behavior is kept, so the logic layer never reads the real clock. Wrote full code for 5 tests and listed both callers and the docs that would need updating. | The same policy, presented as a clear A/B/C options table. Defaulted to `today or date.today()`. Proposed 7 test cases, including the "next date equals today" boundary and the scheduler path. |
| **What was useful** | It read `CLAUDE.md`, the tests and the callers before answering. Its design keeps tests independent of the real date. It flagged every doc that would go stale. | The options table made the trade-offs easy to compare. Its boundary and scheduler-path test cases were ones Opus didn't list. |
| **Problems noticed** | Forgetting to pass `today` silently gives the old behavior. | It read only `pawpal_system.py`, against the prompt. Reading the real clock by default would make tests depend on the date, which breaks a project rule in `CLAUDE.md`, and it admitted existing tests "may break." |
| **Decision** | ✅ Used its signature and its "no real clock in the logic layer" rule | ✅ Used its boundary and scheduler-path tests |

**Which approach did you use in your final implementation and why?**

A combination. Both models independently proposed the same policy (skip missed dates, keep the original schedule), which gave me confidence it was the right behavior. I took Opus's design: `today` is an optional parameter and the logic layer never reads the real clock, because Sonnet's `date.today()` default would have made my tests depend on the current date. I added Sonnet's boundary and scheduler-path tests. I also chose the simpler `while` loop over the arithmetic formula both offered, because it's easier to read and the speed difference doesn't matter at this scale. The main lesson: Sonnet skipped reading `CLAUDE.md`, so it missed a project rule. Opus's more careful context-gathering mattered more than raw capability on this task.
