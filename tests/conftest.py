from datetime import datetime, timedelta, timezone

import pytest

from src.database import Database
from src.executor import Executor
from src.models import (Account, Approval, Category, CommitEvent, Grant, Identity, Policy, Project, Proposal, ReviewEvent, Scenario)

T0 = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)
K = 100_000  # $1,000 in cents


def world(balance=100 * K, budget=500 * K, grants=(), events=(), proposals=(), prior=()):
    return Scenario(
        scenario_id="t", family="t", seed=0, category=Category.LEGITIMATE, twin_id="t", delegation="distributed", concurrency="sequential",
        org_size=3, task="t", lawful_outcome="t",
        accounts=(Account(account_id="A", owner="o", balance_cents=balance), Account(account_id="B", owner="o", balance_cents=balance),
                  Account(account_id="V", owner="v", kind="external"), Account(account_id="X", owner="x", kind="external", restricted=True)),
        projects=(Project(project_id="P", budget_cents=budget),),
        identities=(Identity(identity_id="w1", labels=("alias-w1",)), Identity(identity_id="w2"),
                    Identity(identity_id="a1", can_approve=True, labels=("a1-alt",)), Identity(identity_id="a2", can_approve=True),
                    Identity(identity_id="nobody")),
        grants=tuple(grants), prior_transfers=tuple(prior), proposals=tuple(proposals), events=tuple(events))


def prop(pid="p1", amt=K, who="w1", src="A", dst="V", project="P", approvals=()):
    return Proposal(proposal_id=pid, initiator=who, src=src, dst=dst, amount_cents=amt, project=project,
                    approvals=tuple(Approval(label=x) for x in approvals))


def seq_events(*pids, gap=60, start=T0):
    ev, t = [], start
    for pid in pids:
        ev.append(ReviewEvent(at=t, proposal_id=pid)); t += timedelta(seconds=gap)
        ev.append(CommitEvent(at=t, proposal_id=pid)); t += timedelta(seconds=gap)
    return tuple(ev)


def overlap_events(*pids, gap=60, start=T0):
    ev, t = [], start
    for pid in pids:
        ev.append(ReviewEvent(at=t, proposal_id=pid)); t += timedelta(seconds=gap)
    for pid in pids:
        ev.append(CommitEvent(at=t, proposal_id=pid)); t += timedelta(seconds=gap)
    return tuple(ev)


@pytest.fixture
def ledger():
    db = Database.from_scenario(world())
    yield db, Executor(db, Policy())
    db.close()
