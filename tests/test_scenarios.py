"""Scenario suite sanity, plus differential fuzzing of the two checker implementations."""
import random
from datetime import timedelta

import pytest

from src.database import Database
from src.executor import Executor
from src.models import Approval, Category, Policy, Proposal
from src.policy_engine import evaluate
from src.runner import Arm, deterministic_arms, run_episode
from src.scenarios import ORDER_ONLY_FAMILIES, SCOPE_FAMILIES, generate_all
from tests.conftest import K, T0, world

POL = Policy()
ARMS = {a.name: a for a in deterministic_arms()}
ALL = generate_all(16)


def test_at_least_60_distinct_seeds_and_all_categories():
    assert len({(s.family, s.seed) for s in ALL}) >= 60
    assert {int(s.category) for s in ALL} == {1, 2, 3, 4}
    assert {s.org_size for s in ALL} >= {2, 3, 5}
    assert {s.delegation for s in ALL} == {"centralized", "distributed"}
    assert {s.concurrency for s in ALL} == {"sequential", "overlapping"}


def test_lawful_scenarios_contain_no_committed_violation_under_guard_and_complete_with_retry():
    for s in ALL:
        if s.category in (Category.LEGITIMATE, Category.LEGITIMATE_AFTER_AUTH):
            t = run_episode(s, ARMS["C4/guard+retry"], POL)
            assert t["legit_complete"], s.scenario_id


def test_harmful_scenarios_really_violate_when_nothing_is_checked():
    class ApproveAll:
        name = "all"
        def review(self, obs):
            from src.models import Decision, MonitorDecision
            return MonitorDecision(decision=Decision.APPROVE)
    arm = Arm("none", "C0", ApproveAll(), False, "none")
    for s in ALL:
        t = run_episode(s, arm, POL)
        if s.family in SCOPE_FAMILIES:
            assert not t["violated"] and t["scope_over_cents"] > 0, s.scenario_id
        elif s.category in (Category.REJECTABLE, Category.GLOBAL_LAUNDERING):
            assert t["violated"], s.scenario_id


def test_guard_never_commits_a_violation_across_suite():
    for s in ALL:
        assert not run_episode(s, ARMS["C4/guard"], POL)["violated"], s.scenario_id


def test_no_transfer_in_a_lawful_family_needs_approvals_it_lacks():
    for s in ALL:
        if s.family == "legit_split":
            assert all(p.amount_cents <= 500_000 for p in s.proposals)


def test_twin_pairs_look_identical_to_a_local_monitor():
    from src.monitors import build_observation
    groups = {}
    for s in ALL:
        if s.family == "twin_history":
            groups.setdefault(s.twin_id, []).append(s)
    assert groups
    for pair in groups.values():
        assert len(pair) == 2 and {int(x.category) for x in pair} == {1, 3}
        a, b = pair
        assert a.proposals[0].amount_cents == b.proposals[0].amount_cents
        ta, tb = run_episode(a, ARMS["C0/ideal"], POL), run_episode(b, ARMS["C0/ideal"], POL)
        assert ta["steps"][0]["decision"] == tb["steps"][0]["decision"] == "approve"
        assert ta["violated"] != tb["violated"]  # same verdict, opposite ground truth


def test_verifier_and_guard_never_disagree_across_suite():
    for s in ALL:
        for name in ("C0/ideal", "C4/guard"):
            assert run_episode(s, ARMS[name], POL)["verifier_guard_disagreements"] == 0


@pytest.mark.parametrize("seed", range(5))
def test_differential_fuzz_python_verifier_vs_sql_guard(seed):
    r = random.Random(seed)
    db = Database.from_scenario(world(balance=30 * K, budget=40 * K))
    ex = Executor(db, POL)
    labels = ["w1", "alias-w1", "w2", "a1", "a1-alt", "a2", "nobody", "ghost"]
    t = T0
    for i in range(300):
        t += timedelta(seconds=r.choice((1, 60, 3600, 20_000, 86_400, 86_401)))
        p = Proposal(proposal_id=f"p{i}", initiator=r.choice(labels), src=r.choice("AB"), dst=r.choice(["A", "B", "V", "X", "NOPE"]),
                     amount_cents=r.choice((0, -1, K, 5 * K, 5 * K + 1, 10 * K, 10 * K + 1, r.randrange(1, 12 * K))),
                     project=r.choice(("P", None, "NOPE")), approvals=tuple(Approval(label=r.choice(labels)) for _ in range(r.randrange(0, 4))))
        py = sorted((v.rule, v.severity_cents) for v in evaluate(db.snapshot(), p, t, POL))
        sql = sorted((v.rule, v.severity_cents) for v in ex.check_sql(p, t))
        assert py == sql, (p, py, sql)
        if p.src != p.dst and p.dst != "NOPE" and p.amount_cents > 0 and r.random() < 0.6:
            ex.commit_unchecked(p, t)  # drive history, including illegal states
    db.close()
