"""Reservation review, attribution that resists lucky rejections, the exposure bound, and the Postgres race."""
import os
import shutil

import pytest

from experiments.run_workloads import ARMS as WL_ARMS, POLICY as WL_POLICY, make_workload, run_cell
from src import metrics as M
from src.models import AddGrantEvent, CommitEvent, Grant, Policy, ReleaseEvent, ReviewEvent, RevokeGrantEvent
from src.runner import deterministic_arms, run_episode

from .conftest import K, T0, overlap_events, prop, seq_events, world

ARMS = {a.name: a for a in deterministic_arms()}
POL = Policy()
T = [T0.replace(minute=m) for m in range(20)]
AP = ("a1", "a2")  # $6,000 is above the dual-approval threshold, so capacity is the only rule left to bind


def test_c6_counts_approved_in_flight_proposals():
    sc = world(proposals=[prop("p1", 6 * K, approvals=AP), prop("p2", 6 * K, who="w2", approvals=AP)], events=overlap_events("p1", "p2"))
    assert run_episode(sc, ARMS["C2/ideal"], POL)["violated"]
    t = run_episode(sc, ARMS["C6/ideal"], POL)
    assert not t["violated"] and [s["committed"] for s in t["steps"]] == [True, False]


def test_release_frees_a_hold_only_after_it_happens():
    ev = (ReviewEvent(at=T[0], proposal_id="p1"), ReleaseEvent(at=T[1], proposal_id="p1"), ReviewEvent(at=T[2], proposal_id="p2"), CommitEvent(at=T[3], proposal_id="p2"))
    assert run_episode(world(proposals=[prop("p1", 6 * K, approvals=AP), prop("p2", 6 * K, who="w2", approvals=AP)], events=ev), ARMS["C6/ideal"], POL)["steps"][-1]["committed"]
    late = (ReviewEvent(at=T[0], proposal_id="p1"), ReviewEvent(at=T[1], proposal_id="p2"), ReleaseEvent(at=T[2], proposal_id="p1"), CommitEvent(at=T[3], proposal_id="p2"))
    assert not run_episode(world(proposals=[prop("p1", 6 * K, approvals=AP), prop("p2", 6 * K, who="w2", approvals=AP)], events=late), ARMS["C6/ideal"], POL)["steps"][-1]["committed"]


def test_c6_does_not_fix_state_that_changes_after_review():
    g = Grant(grant_id="g", account_id="A", extra_ceiling_cents=10 * K, valid_from=T0.replace(hour=1), expires_at=T0.replace(day=20))
    ev = (ReviewEvent(at=T[0], proposal_id="p1"), RevokeGrantEvent(at=T[1], grant_id="g"), CommitEvent(at=T[2], proposal_id="p1"))
    sc = world(grants=[g], proposals=[prop("p1", 15 * K, approvals=("a1", "a2"))], events=ev)
    assert run_episode(sc, ARMS["C6/ideal"], POL)["violated"]
    assert not run_episode(sc, ARMS["C4/guard"], POL)["violated"]


def test_a_hold_is_never_written_to_the_ledger():
    """A hold is the reviewer's knowledge and is never written to the ledger."""
    sc = world(proposals=[prop("p1", 3 * K)], events=(ReviewEvent(at=T[0], proposal_id="p1"), ReleaseEvent(at=T[1], proposal_id="p1")))
    t = run_episode(sc, ARMS["C6/ideal"], POL)
    assert t["steps"] == [] and t["initial_hash"] == t["final_hash"]


def _fake(sid, fam, **arms):
    return {a: {"scenario_id": sid, "family": fam, "violated": v} for a, v in arms.items()}


def test_ladder_never_credits_c1_for_a_lucky_rejection():
    # C1 removes it, but C2 with full information does not: that is an isolation failure.
    rows = [_fake("s", "f", **{"C0/ideal": True, "C1/ideal": False, "C2/ideal": True, "C6/ideal": True, "C4/guard": False})]
    by = {a: [r[a] for r in rows] for a in rows[0]}
    lad = M.attribution_ladder(by)
    assert lad["by_cause"] == {"isolation_post_review": 1} and lad["removed_by_false_rejection_at_c1"] == 1


@pytest.mark.parametrize("c1,c2,c6,c4,cause", [(False, False, False, False, "information_history"), (True, False, False, False, "information_global"),
                                               (True, True, False, False, "isolation_in_flight"), (True, True, True, False, "isolation_post_review"), (True, True, True, True, "unresolved")])
def test_ladder_causes(c1, c2, c6, c4, cause):
    r = _fake("s", "f", **{"C0/ideal": True, "C1/ideal": c1, "C2/ideal": c2, "C6/ideal": c6, "C4/guard": c4})
    assert M.attribution_ladder({a: [v] for a, v in r.items()})["by_cause"] == {cause: 1}


def test_workload_is_deterministic_and_identical_across_arms():
    a, _ = make_workload(4, 1.5, 0.1, 1, 3)
    b, _ = make_workload(4, 1.5, 0.1, 1, 3)
    assert a == b
    assert len({run_episode(a, arm, WL_POLICY)["initial_hash"] for arm in WL_ARMS}) == 1


@pytest.mark.parametrize("w", [1, 2, 4, 8])
@pytest.mark.parametrize("rho", [0.9, 1.1, 2.0])
@pytest.mark.parametrize("q", [0.0, 0.3])
def test_exposure_bound_and_guards(w, rho, q):
    for rep in range(6):
        rows = {r["arm"]: r for r in run_cell((w, rho, q, 0, rep))}
        assert rows["C2/ideal"]["overshoot_cents"] <= (w - 1) * rows["C2/ideal"]["a_max"]
        assert rows["C3k1/ideal"]["overshoot_cents"] <= w * rows["C3k1/ideal"]["a_max"]
        for arm in ("C6/ideal", "C4/guard", "C5/ideal+guard"):
            assert not rows[arm]["violated"] and rows[arm]["overshoot_cents"] == 0
        if w == 1 or rho <= 1:
            assert not rows["C2/ideal"]["violated"]


def test_workload_demand_matches_rho_and_stays_below_dual_threshold():
    sc, _ = make_workload(8, 1.5, 0.0, 0, 0)
    assert sum(p.amount_cents for p in sc.proposals) == round(1.5 * WL_POLICY.transfer_ceiling_cents)
    assert max(p.amount_cents for p in sc.proposals) <= WL_POLICY.dual_approval_threshold_cents


@pytest.mark.skipif(not os.environ.get("PG_RACE_TEST") or not (shutil.which("pg_ctl") or os.path.exists("/opt/homebrew/opt/postgresql@15/bin/pg_ctl")),
                    reason="set PG_RACE_TEST=1 with PostgreSQL installed; uses a throwaway cluster")
def test_postgres_race_smoke():
    from experiments import run_postgres_race as R
    R.start_cluster()
    try:
        admin = R.connect()
        admin.execute("DROP TABLE IF EXISTS transfers, holds, accounts, scopes")
        admin.execute(R.SCHEMA)
        conns = [R.connect() for _ in range(4)]
        run = lambda m, w: [R.trial(conns, admin, m, w, 4, 2, i) for i in range(5)]
        assert any(t["overshoot"] for t in run("guard_read_committed", "cross_account"))
        assert any(t["overshoot"] for t in run("guard_repeatable_read", "cross_account"))
        for m in ("guard_serializable", "lock_invariant", "reserve_atomic_fixed"):
            ts = run(m, "cross_account")
            assert not any(t["overshoot"] for t in ts) and all(t["conserved"] for t in ts)
        assert all(t["lost_lawful"] == 0 for t in run("lock_invariant", "cross_account_lawful"))
        for c in conns:
            c.close()
        admin.close()
    finally:
        R.stop_cluster()
