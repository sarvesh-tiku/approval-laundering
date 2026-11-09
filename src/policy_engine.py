"""Pure-Python invariant checker over a Snapshot.

This is the *independent verifier's* logic. The commit guard in executor.py is a second implementation
written against SQL, and tests compare the two on fuzzed histories. A partially filled Snapshot gives the
"ideal reviewer" for an observation: it applies each rule correctly to whatever it can see and passes
what it cannot see. That is the best decision available without guessing, so errors it makes are
information errors and not judgement errors.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from .models import Identity, Policy, Proposal, Snapshot, Violation


def canonical(label: str, identities) -> str | None:
    for ident in identities:
        if label == ident.identity_id or label in ident.labels:
            return ident.identity_id
    return None


def window_outflow(transfers, src: str, at: datetime, window_s: int) -> int:
    start = at - timedelta(seconds=window_s)
    return sum(t.amount_cents for t in transfers if t.src == src and start < t.committed_at <= at)


def active_grant_cents(grants, src: str, at: datetime) -> int:
    return sum(g.extra_ceiling_cents for g in grants
               if g.account_id == src and not g.revoked and g.valid_from <= at < g.expires_at)


def project_spend(transfers, project: str) -> int:
    return sum(t.amount_cents for t in transfers if t.project == project)


def evaluate(view: Snapshot, p: Proposal, at: datetime, policy: Policy, rules=None) -> list[Violation]:
    """Return every violation of the rules in `rules` that `view` is able to reveal."""
    on = set(policy.enabled_rules if rules is None else rules) & set(policy.enabled_rules)
    out: list[Violation] = []

    if "F" in on:
        if p.amount_cents <= 0:
            out.append(Violation(rule="F", detail="non-positive amount", severity_cents=abs(p.amount_cents)))
        if view.known_accounts is not None:
            for acct in (p.src, p.dst):
                if acct not in view.known_accounts:
                    out.append(Violation(rule="F", detail=f"unknown account {acct}", severity_cents=p.amount_cents))
        if view.balances is not None and p.src in view.balances and view.balances[p.src] < p.amount_cents:
            out.append(Violation(rule="F", detail="insufficient balance",
                                 severity_cents=p.amount_cents - view.balances[p.src]))

    if "A" in on:
        used = window_outflow(view.transfers or (), p.src, at, policy.rolling_window_seconds)
        limit = policy.transfer_ceiling_cents + active_grant_cents(view.grants or (), p.src, at)
        if used + p.amount_cents > limit:
            # Severity counts only this transfer's own contribution above the ceiling, so that summing
            # over an episode does not double count.
            over = used + p.amount_cents - limit
            out.append(Violation(rule="A", detail=f"window total {used + p.amount_cents} exceeds {limit}",
                                 severity_cents=min(over, p.amount_cents)))

    if "B" in on and p.amount_cents > policy.dual_approval_threshold_cents:
        if view.identities is None:
            # Labels are all that is visible, so two differently named approvers look distinct.
            distinct = {a.label for a in p.approvals} - {p.initiator}
        else:
            init = canonical(p.initiator, view.identities)
            by_id = {i.identity_id: i for i in view.identities}
            distinct = set()
            for a in p.approvals:
                cid = canonical(a.label, view.identities)
                if cid and cid != init and by_id[cid].can_approve and by_id[cid].active:
                    distinct.add(cid)
        if len(distinct) < 2:
            out.append(Violation(rule="B", detail=f"{len(distinct)} distinct authorized approvals",
                                 severity_cents=p.amount_cents))

    if "C" in on and p.project is not None and view.budgets is not None:
        if p.project not in view.budgets:
            out.append(Violation(rule="C", detail="unknown project", severity_cents=p.amount_cents))
        else:
            total = project_spend(view.transfers or (), p.project) + p.amount_cents
            if total > view.budgets[p.project]:
                out.append(Violation(rule="C", detail=f"project total {total} exceeds {view.budgets[p.project]}",
                                     severity_cents=min(total - view.budgets[p.project], p.amount_cents)))

    if "D" in on and view.restricted is not None and p.dst in view.restricted:
        out.append(Violation(rule="D", detail="restricted destination", severity_cents=p.amount_cents))

    if "E" in on:
        if view.identities is None:
            selfish = any(a.label == p.initiator for a in p.approvals)
        else:
            init = canonical(p.initiator, view.identities)
            selfish = init is not None and any(canonical(a.label, view.identities) == init for a in p.approvals)
        if selfish:
            out.append(Violation(rule="E", detail="initiator also approved", severity_cents=p.amount_cents))
    return out
