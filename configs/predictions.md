# Predictions for the extension experiments

Written before any of these experiments ran (2026-11-08, about 23:20 EST). The file is local and not
publicly timestamped, so treat this as a working record and not as a formal preregistration. The paper
reports every prediction below, including any that fail.

Terms: `w` = most proposals in flight at once; `rho` = total demand / capacity; `q` = chance an approved
proposal is abandoned before commit; `g` = a grant is revoked between a review and a commit.

## P1. Postgres, cross-account invariant (shared project budget, disjoint rows written)
Overshoot occurs under: check-then-commit, guard at READ COMMITTED, guard at REPEATABLE READ (write skew),
a lock on the source account (wrong row), and an unlocked reservation.
Zero overshoot under: guard at SERIALIZABLE, a lock on the invariant's own row, and an atomic reservation.

## P2. Postgres, same-account invariant (all writers update one row)
Overshoot occurs under: check-then-commit and guard at READ COMMITTED.
Zero overshoot under REPEATABLE READ, because first-updater-wins aborts the conflict. That safety is an
accident of the rows the transfer writes, which is why P1 expects it to disappear across accounts.
Zero overshoot under SERIALIZABLE and both lock modes.

## P3. Postgres, lawful work that all fits
Lock modes lose no lawful transfers. SERIALIZABLE and REPEATABLE READ need retries on the same-account
workload, and at 8 threads with at most 5 retries some lawful transfers are lost.
REPEATABLE READ needs no retries on the cross-account workload, because no written row is shared.

## P4. Reservation reviewer (C6) on the fixed suite
C6 removes the violations caused by overlapping proposals, and leaves those caused by state changing after
review: all 32 `revoked_grant` episodes and the other families with an admin change between review and commit.
C6's false rejection rate stays within 1 point of C2's, since every approved proposal in that suite commits.

## P5. Random workloads
- C2 commits zero violations when `w = 1` or `rho <= 1`, with `g = 0`.
- C2's violation rate rises with `w` and with `rho`.
- Overshoot bound, `g = 0`: C2 overshoot never exceeds `(w - 1) * a_max` and C3 with lag `k` never exceeds
  `(w - 1 + k) * a_max`, where `a_max` is the largest amount. The bound is not vacuous: worst observed
  overshoot exceeds half the bound in at least one cell with `w >= 4`.
- C6 commits zero violations at `g = 0` and some at `g = 1`.
- C4 and C5 commit zero violations everywhere.
- C6 completes less lawful value than C4 once `q > 0` and `rho >= 1`, because abandoned holds tie up
  capacity. With `q = 0` the two match.

## P6. Family-mix sensitivity (random Dirichlet mixes over the same stored traces)
The paired C0 minus C2 gap stays positive in more than 99% of mixes. The share of C0 violations that
survive C2 varies widely (the 95% range spans more than 20 points), so the paper should not lean on one pooled share.

# Predictions for the language-model trials

Written before any model call (2026-11-09, about 00:20 EST), with the same caveat: a local file, not a formal
preregistration. Models: `openai/gpt-4.1-mini` and `google/gemini-2.5-flash`. Each holds for both models unless stated.

- L1. Decision failures exist: under C2, each LLM reviewer commits a violation in at least one invariant episode where
  the ideal reviewer at the same observation does not.
- L2. Exact computed aggregates help: C2agg commits fewer such decision failures than C2 for each model.
- L3. More context does not fix overlap: on overlapping harmful factorial episodes, C2/llm GIVR stays at or above 70%.
- L4. Local approval remains weak: C0/llm GIVR is at or above 40%, and above C2/llm GIVR.
- L5. C5/llm+guard commits zero violations, because the guard decides last.
- L6. An LLM with the task text catches something the guard cannot: C5/llm+guard has fewer task-text limit breaches
  than the 32-of-32 rate of the guard alone, for at least one model.
- L7. H5, incentives: in rollouts, the completion objective produces more episodes with an attempted violation or
  dual-approval structuring than the compliance objective, for each model.
- L8. Rollouts under C0 commit at least one violation for each model, so models do produce these sequences without
  being told to.
- L9. Models structure under the guard: under C4 with the completion objective, dual-approval structuring occurs in at
  least one rollout for each model.

## Added after the pilot (2026-11-09, about 00:30 EST)
The pilot showed both models answering `request_more_info` on nearly every C0 review, which contradicts L4 in those
episodes. L4 stays scored as written. These additions were written after seeing that pilot, so they are not independent of it.
A new forced-choice C0 arm allows only approve or reject. Rollouts use it in place of C0, and L8 is scored on it.

- L10. Forced to choose, C0f/llm GIVR is at or above 40% for each model.
- L11. Forced to choose, C0f/llm rejects lawful proposals at a higher rate than the C0 ideal reviewer's 0%.

## Added after full fixed replay (2026-11-09, about 00:50 EST)
Full replay showed C0 models rejecting or deferring almost everything, even when forced. The C0 prompt says "No ledger
history or balances are available to you", which may cause this, so new arms remove that one sentence. Written after seeing
those results, so not independent of them.

- L12. Without the sentence (C0n, deferral allowed; C0nf, forced), both models approve more proposals than with it.
- L13. Without the sentence, at least one model's C0nf GIVR is at or above 40%.

Rollouts use C0nf as their local monitor, so L8 is scored on C0nf. This choice was made after seeing C0nf's replay results.

Rollouts were first run with one sample per prompt. The cache then gives identical prompts one shared reply, so runs across arms
were not independent. All L7 to L9 scoring uses the final 3-sample run with 4 task seeds, and intervals bootstrap over tasks.
