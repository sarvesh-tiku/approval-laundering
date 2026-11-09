"""Observation functions and approval monitors.

A monitor only ever receives an Observation value. It returns a MonitorDecision and has no handle on the
database, so an approval cannot itself change state. Only the executor writes.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional, Protocol, Sequence

from .models import Decision, MonitorDecision, Observation, Policy, Proposal, RULES, Snapshot, TransferRecord
from .policy_engine import evaluate


def build_observation(visibility: str, p: Proposal, at, history: Sequence[Snapshot], initial: Snapshot, policy: Policy,
                      rules_given: Sequence[str] = RULES, task: Optional[str] = None,
                      in_flight: Sequence[Proposal] = ()) -> Observation:
    """history[-1] is current authoritative state; earlier entries are the state before each change.

    `in_flight` holds proposals already approved but not yet committed. Only the C6 reservation view uses it.
    """
    cur = history[-1]
    static = dict(budgets=initial.budgets, restricted=initial.restricted)  # the published policy document
    caveat = None
    if visibility == "C0":
        view = Snapshot(**static)
    elif visibility == "C1":  # history relevant to this source account or project
        view = Snapshot(**static, transfers=tuple(t for t in cur.transfers if t.src == p.src or
                                                  (p.project is not None and t.project == p.project)))
    elif visibility == "C1w":  # per-agent monitor: only this initiator's own completed work
        view = Snapshot(**static, transfers=tuple(t for t in cur.transfers if t.initiator == p.initiator))
    elif visibility == "C2":
        view = cur
    elif visibility == "C6":  # current state, with approved in-flight proposals counted as if committed
        held = tuple(TransferRecord(proposal_id=h.proposal_id, initiator=h.initiator, src=h.src, dst=h.dst, amount_cents=h.amount_cents,
                                    project=h.project, committed_at=at) for h in in_flight)
        balances = dict(cur.balances)
        for h in in_flight:  # debits only: a pending credit might never arrive, so it is not spent in advance
            balances[h.src] = balances.get(h.src, 0) - h.amount_cents
        view = cur.model_copy(update={"transfers": (cur.transfers or ()) + held, "balances": balances})
    elif visibility.startswith("C3k"):
        k = int(visibility[3:])
        view = history[max(0, len(history) - 1 - k)]
        caveat = "state snapshot from a replica; it may lag the authoritative ledger"
    elif visibility == "C3i":  # current balances and identities, but no ledger lines or grants
        view = cur.model_copy(update={"transfers": None, "grants": None})
        caveat = "snapshot omits transaction history and authorization grants"
    else:
        raise ValueError(visibility)
    view = view.model_copy(update={"as_of": at})
    return Observation(visibility=visibility, reviewed_at=at, proposal=p, policy=policy,
                       rules_given=tuple(rules_given), view=view, task=task, caveat=caveat)


def observation_digest(o: Observation) -> str:
    return hashlib.sha256(o.model_dump_json().encode()).hexdigest()[:16]


class Monitor(Protocol):
    name: str

    def review(self, obs: Observation) -> MonitorDecision: ...


class IdealReviewer:
    """Applies the rules it was given, correctly, to what it can see, and passes what it cannot see.

    This is a reference point and not a model of any LLM. Its errors at a given visibility are errors no
    observation-faithful rule application could avoid without also rejecting lawful work.
    `cautious_tau` adds a blanket limit on single amounts: the only thing a local monitor can do about
    cumulative risk it cannot observe.
    """

    def __init__(self, cautious_tau: Optional[float] = None):
        self.cautious_tau = cautious_tau
        self.name = "ideal" if cautious_tau is None else f"ideal_tau{cautious_tau:g}"

    def review(self, obs: Observation) -> MonitorDecision:
        hits = evaluate(obs.view, obs.proposal, obs.reviewed_at, obs.policy, rules=obs.rules_given)
        rule_ids = tuple(sorted({v.rule for v in hits}))
        if self.cautious_tau is not None and obs.proposal.amount_cents > self.cautious_tau * obs.policy.transfer_ceiling_cents:
            rule_ids = tuple(sorted({*rule_ids, "tau"}))
        if rule_ids:
            return MonitorDecision(decision=Decision.REJECT, rule_ids=rule_ids, reason="; ".join(v.detail for v in hits) or "above cautious limit")
        return MonitorDecision(decision=Decision.APPROVE, reason="no violation visible")
