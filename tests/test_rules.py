"""Boundary behaviour of every invariant, on both the pure-Python verifier and the SQL guard."""
from datetime import timedelta

import pytest

from src.database import Database
from src.executor import Executor
from src.models import Grant, Policy
from src.policy_engine import evaluate
from tests.conftest import K, T0, prop, world

POL = Policy()


def both(db, ex, p, at):
    py = {v.rule for v in evaluate(db.snapshot(), p, at, POL)}
    sql = {v.rule for v in ex.check_sql(p, at)}
    assert py == sql, (py, sql)
    return py


def test_exact_ceiling_allowed_and_one_cent_over_rejected(ledger):
    db, ex = ledger
    ok = prop(amt=10 * K, approvals=("a1", "a2"))
    assert "A" not in both(db, ex, ok, T0)
    assert "A" in both(db, ex, prop(amt=10 * K + 1, approvals=("a1", "a2")), T0)


def test_exact_dual_threshold_needs_no_approval(ledger):
    db, ex = ledger
    assert "B" not in both(db, ex, prop(amt=5 * K), T0)
    assert "B" in both(db, ex, prop(amt=5 * K + 1), T0)


def test_two_6k_transfers_not_inherently_prohibited_outside_window(ledger):
    db, ex = ledger
    first = prop("p1", 6 * K, approvals=("a1", "a2"))
    assert ex.commit_guarded(first, T0)[0]
    second = prop("p2", 6 * K, approvals=("a1", "a2"))
    assert "A" in both(db, ex, second, T0 + timedelta(hours=23))
    assert "A" not in both(db, ex, second, T0 + timedelta(hours=25))


def test_rolling_window_boundary_is_half_open(ledger):
    db, ex = ledger
    assert ex.commit_guarded(prop("p1", 6 * K, approvals=("a1", "a2")), T0)[0]
    p2 = prop("p2", 6 * K, approvals=("a1", "a2"))
    assert "A" in both(db, ex, p2, T0 + timedelta(seconds=86_399))
    assert "A" not in both(db, ex, p2, T0 + timedelta(seconds=86_400))  # exactly 24h earlier has left the window


def test_ceiling_is_per_account(ledger):
    db, ex = ledger
    assert ex.commit_guarded(prop("p1", 6 * K, approvals=("a1", "a2")), T0)[0]
    assert "A" not in both(db, ex, prop("p2", 6 * K, src="B", approvals=("a1", "a2")), T0 + timedelta(minutes=1))


def test_grant_raises_ceiling_only_for_its_account_and_time(ledger):
    db, ex = ledger
    db.insert_grant(Grant(grant_id="g", account_id="A", extra_ceiling_cents=2 * K, valid_from=T0, expires_at=T0 + timedelta(hours=1)))
    p = prop(amt=12 * K, approvals=("a1", "a2"))
    assert "A" not in both(db, ex, p, T0)
    assert "A" in both(db, ex, p, T0 - timedelta(seconds=1))
    assert "A" in both(db, ex, p, T0 + timedelta(hours=1))  # expiry is exclusive
    assert "A" in both(db, ex, prop(amt=12 * K, src="B", approvals=("a1", "a2")), T0)


def test_duplicate_label_counts_once(ledger):
    db, ex = ledger
    assert "B" in both(db, ex, prop(amt=6 * K, approvals=("a1", "a1")), T0)


def test_alias_of_same_identity_counts_once(ledger):
    db, ex = ledger
    assert "B" in both(db, ex, prop(amt=6 * K, approvals=("a1", "a1-alt")), T0)


def test_non_approver_identity_does_not_count(ledger):
    db, ex = ledger
    assert "B" in both(db, ex, prop(amt=6 * K, approvals=("a1", "nobody")), T0)


def test_deactivated_approver_does_not_count(ledger):
    db, ex = ledger
    db.conn.execute("UPDATE identities SET active=0 WHERE identity_id='a2'")
    assert "B" in both(db, ex, prop(amt=6 * K, approvals=("a1", "a2")), T0)


def test_separation_of_duties_through_alias(ledger):
    db, ex = ledger
    assert "E" in both(db, ex, prop(amt=K, who="w1", approvals=("alias-w1",)), T0)
    assert "E" not in both(db, ex, prop(amt=K, who="w1", approvals=("a1",)), T0)


def test_restricted_destination(ledger):
    db, ex = ledger
    assert "D" in both(db, ex, prop(dst="X"), T0)


def test_insufficient_balance_exact_boundary():
    db = Database.from_scenario(world(balance=3 * K))
    ex = Executor(db, POL)
    assert "F" not in both(db, ex, prop(amt=3 * K), T0)
    assert "F" in both(db, ex, prop(amt=3 * K + 1), T0)


def test_non_positive_and_unknown_account(ledger):
    db, ex = ledger
    assert "F" in both(db, ex, prop(amt=0), T0)
    assert "F" in both(db, ex, prop(amt=-5), T0)
    assert "F" in both(db, ex, prop(dst="NOPE"), T0)


def test_budget_exhaustion_exact_and_cumulative():
    db = Database.from_scenario(world(budget=3 * K))
    ex = Executor(db, POL)
    assert "C" not in both(db, ex, prop(amt=3 * K), T0)
    assert ex.commit_guarded(prop("p1", 2 * K), T0)[0]
    assert "C" not in both(db, ex, prop("p2", K), T0 + timedelta(minutes=1))
    assert "C" in both(db, ex, prop("p3", K + 1, src="B"), T0 + timedelta(minutes=1))  # budget spans source accounts


def test_naive_datetime_rejected():
    from datetime import datetime
    from src.database import to_us
    with pytest.raises(ValueError):
        to_us(datetime(2026, 1, 1))
