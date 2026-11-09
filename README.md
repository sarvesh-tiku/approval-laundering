# When Every Step Is Approved: Compositional Failures in Multi-Agent Oversight

Apart Research Swarm Alignment Hackathon, November 2026, Track 4 (Multi-Agent Oversight and Control).

Can individually approved agent actions combine into a state that breaks a global rule? This repository tests that on a fully
synthetic ledger, and separates four causes: **specification** (rule never given), **information** (state not visible),
**isolation** (state changed between review and commit), and **decision** (rule and state available, wrong call).

Report: [`paper/report.pdf`](paper/report.pdf).

## Status: what ran and what did not

| Part | Status |
|---|---|
| Fixed-action replay with deterministic reference reviewers | Ran: 880 scenarios x 27 arms = 23,760 episodes |
| Random workloads (in-flight limit, demand, abandonment, revocation) | Ran: 180 cells x 200 replicates x 6 arms = 216,000 episodes |
| Real-thread race on file-backed SQLite | Ran: 2,400 trials |
| Real-thread race on PostgreSQL 15 (9 modes, 4 workloads) | Ran: 43,200 trials, plus a sequential control (1,440), a 50-retry check (3,200) and staggered arrivals (1,600) |
| Recorded predictions, scored by code | `configs/predictions.md`; scored in `results/table_predictions.tex`. Not a formal preregistration |
| Rollouts with scripted (handcrafted) agents | Ran: 1,536 rollouts |
| LLM replay, `openai/gpt-4.1-mini` and `google/gemini-2.5-flash` | Ran: 220 scenarios x 9 arms x 2 models = 3,960 episodes |
| LLM rollouts, same models | Ran: 32 tasks x 32 conditions x 3 samples x 2 models = 6,144 rollouts, plus 32 excluded pilot rollouts |

LLM spend: $7.05 over 18,561 calls, from stored usage and matched against the provider's counter. Results cover two inexpensive models
on synthetic tasks and are not frontier-agent results. Several LLM arms were added after seeing earlier results, and `configs/predictions.md` records which.
The C6 reservation arm, random workloads and thread races use reference reviewers or scripted checks, with no LLM. Every record has a `mode` field, and each mode has its own results file.

The reference reviewer is a measuring device: it applies the rules correctly to what it can see. Many of its results follow from
the semantics, and the guard's zero violations hold by construction. Pooled rates depend on the designed scenario mix and are not
estimates of any real system.

## Reproduce

```bash
uv venv --python 3.11 ~/.venvs/approval-laundering && ln -s ~/.venvs/approval-laundering .venv
uv pip install --python .venv/bin/python -e ".[dev]"
source .venv/bin/activate

pytest -q
python -m experiments.run_workloads --reps 200
# Needs PostgreSQL binaries and `uv pip install -e ".[postgres]"`. Uses a throwaway cluster in ~/.cache and a unix socket.
python -m experiments.run_postgres_race
python -m experiments.run_postgres_race --trials 20 --threads 8 --sequential --out results/postgres_control.jsonl
python -m experiments.run_postgres_race --trials 200 --threads 8 --modes guard_repeatable_read guard_serializable --max-retries 50 --out results/postgres_retry50.jsonl
for J in 10 100; do python -m experiments.run_postgres_race --trials 200 --threads 2 8 --latency 0 --modes guard_read_committed guard_repeatable_read --workloads cross_account --jitter-ms $J --out results/postgres_jitter$J.jsonl; done
python -m experiments.run_baselines                  # deterministic replay and thread race
python -m experiments.run_llm_trials --scripted-only # scripted-agent rollouts; no key needed
python -m experiments.run_llm_trials --phase fixed --workers 16    # needs a key; stored results are skipped
python -m experiments.run_llm_trials --phase rollouts --workers 12 # needs a key; stored results are skipped
python -m experiments.analyze                        # analysis.json, numbers.tex, tables, figures
(cd paper && latexmk -pdf report.tex)
```

Runs append to `results/*.jsonl` and skip stored observations, so re-running does not duplicate them. Fixed replay is deterministic;
the thread race is not, so delete `results/thread_race.jsonl` to resample it.

### Running the LLM trials

```bash
cp .env.example .env        # set LLM_API_KEY, LLM_MODELS, LLM_BUDGET_USD
python -m experiments.run_llm_trials --phase fixed --limit 30   # pilot; check cost
python -m experiments.run_llm_trials                            # full run, resumable, stops at the budget
python -m experiments.analyze
```

Any OpenAI-compatible endpoint works. `.env` overrides shell exports. The budget applies per invocation. Responses are cached in
`~/.cache/approval-laundering/llm_cache`. Invalid model output fails closed and is counted. `analyze.py` writes LLM results to
`analysis.json`, numbers and tables for the paper. Reproducing the reported rollouts uses `configs/experiments.yaml` as stored; `--budget` sets a hard per-run stop.

## Layout

| Path | Purpose |
|---|---|
| `src/models.py` | Typed records; integer cents, timezone-aware times, immutable observations |
| `src/database.py` | Authoritative SQLite ledger and state hashing |
| `src/policy_engine.py` | Independent verifier: rules in pure Python over a snapshot |
| `src/executor.py` | Only writer. Unchecked commit, and the atomic guard (rules in SQL under `BEGIN IMMEDIATE`) |
| `src/monitors.py` | Visibility functions C0-C3 and ablations; ideal reviewer |
| `src/agents.py`, `src/llm_provider.py` | LLM and scripted agents; provider client with cache and budget |
| `src/scenarios.py` | Seeded generator: 27 families, four categories |
| `src/runner.py`, `src/rollouts.py` | Evaluation A (fixed actions) and Evaluation B (adaptive) |
| `src/replay.py` | Counterfactual replay and differential reports |
| `src/metrics.py` | GIVR, UAR, FRR, LTCR, gap, severity, cluster bootstrap, attribution ladder |
| `experiments/` | CLI entry points |
| `results/` | Traces, `analysis.json`, `numbers.tex`, tables, `differential_C0.jsonl`, `first_violations_C0.jsonl` |

## Preexisting versus sprint work

Everything here was written during the sprint, on 2026-11-08, by Claude Code (Claude Opus 5.5) at the author's direction. No code
was reused from earlier projects. External dependencies only: Python, SQLite, Pydantic, NumPy, SciPy, Matplotlib, PyYAML, pytest.

## Ethics

All agents, accounts, vendors and money are fictional. Nothing connects to real financial systems.
