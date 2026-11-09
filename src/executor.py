"""The only writer of the ledger. Two commit paths:

* commit_unchecked: applies whatever the approval step passed. Ledger mechanics stay atomic, but no
  policy check happens at commit, so an approved overdraft really does overdraw.
* commit_guarded: the deterministic guard. It checks every invariant with SQL inside a BEGIN IMMEDIATE
  transaction, so the read and the write cannot interleave with another writer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from .database import Database, to_us
from .models import (AddGrantEvent, BudgetChangeEvent, DeactivateIdentityEvent, MarkRestrictedEvent, Policy,
                     Proposal, RevokeGrantEvent, Violation)


class InjectedFailure(RuntimeError):
    pass


class Executor:
    def __init__(self, db: Database, policy: Policy):
        self.db, self.policy = db, policy

    # Administrative state changes -----------------------------------------------------------------
    def apply_admin(self, ev) -> None:
        c = self.db.conn
        if isinstance(ev, AddGrantEvent):
            self.db.insert_grant(ev.grant)
        elif isinstance(ev, RevokeGrantEvent):
            c.execute("UPDATE grants SET revoked=1 WHERE grant_id=?", (ev.grant_id,))
        elif isinstance(ev, DeactivateIdentityEvent):
            c.execute("UPDATE identities SET active=0 WHERE identity_id=?", (ev.identity_id,))
        elif isinstance(ev, BudgetChangeEvent):
            c.execute("UPDATE projects SET budget_cents=? WHERE project_id=?", (ev.new_budget_cents, ev.project_id))
        elif isinstance(ev, MarkRestrictedEvent):
            c.execute("UPDATE accounts SET restricted=1 WHERE account_id=?", (ev.account_id,))
        else:
            raise TypeError(type(ev))

    # Commit paths ---------------------------------------------------------------------------------
    def _apply(self, p: Proposal, at: datetime, fail_after_debit: bool) -> None:
        c = self.db.conn
        c.execute("UPDATE accounts SET balance_cents=balance_cents-? WHERE account_id=?", (p.amount_cents, p.src))
        if fail_after_debit:
            raise InjectedFailure("simulated crash between debit and credit")
        c.execute("UPDATE accounts SET balance_cents=balance_cents+? WHERE account_id=?", (p.amount_cents, p.dst))
        c.execute("INSERT INTO transfers(proposal_id,initiator,src,dst,amount_cents,project,committed_at_us)"
                  " VALUES(?,?,?,?,?,?,?)", (p.proposal_id, p.initiator, p.src, p.dst, p.amount_cents, p.project, to_us(at)))

    def commit_unchecked(self, p: Proposal, at: datetime, fail_after_debit: bool = False) -> bool:
        c = self.db.conn
        c.execute("BEGIN IMMEDIATE")
        try:
            self._apply(p, at, fail_after_debit)
            c.execute("COMMIT")
            return True
        except BaseException:
            c.execute("ROLLBACK")
            raise

    def commit_guarded(self, p: Proposal, at: datetime, fail_after_debit: bool = False) -> tuple[bool, list[Violation]]:
        c = self.db.conn
        c.execute("BEGIN IMMEDIATE")
        try:
            before = c.execute("SELECT SUM(balance_cents) FROM accounts").fetchone()[0]
            violations = self.check_sql(p, at)
            if violations:
                c.execute("ROLLBACK")
                return False, violations
            self._apply(p, at, fail_after_debit)
            after = c.execute("SELECT SUM(balance_cents) FROM accounts").fetchone()[0]
            if before != after:  # conservation is re-verified, not assumed
                c.execute("ROLLBACK")
                return False, [Violation(rule="F", detail="conservation failure", severity_cents=abs(after - before))]
            c.execute("COMMIT")
            return True, []
        except BaseException:
            c.execute("ROLLBACK")
            raise

    def check_sql(self, p: Proposal, at: datetime) -> list[Violation]:
        """Second, independent implementation of the invariants, in SQL against live tables."""
        c, pol = self.db.conn, self.policy
        on = set(pol.enabled_rules)
        t, out = to_us(at), []
        src = c.execute("SELECT balance_cents FROM accounts WHERE account_id=?", (p.src,)).fetchone()
        dst = c.execute("SELECT restricted FROM accounts WHERE account_id=?", (p.dst,)).fetchone()

        if "F" in on:
            if p.amount_cents <= 0:
                out.append(Violation(rule="F", detail="non-positive amount", severity_cents=abs(p.amount_cents)))
            if src is None:
                out.append(Violation(rule="F", detail=f"unknown account {p.src}", severity_cents=p.amount_cents))
            if dst is None:
                out.append(Violation(rule="F", detail=f"unknown account {p.dst}", severity_cents=p.amount_cents))
            if src is not None and src[0] < p.amount_cents:
                out.append(Violation(rule="F", detail="insufficient balance", severity_cents=p.amount_cents - src[0]))

        if "A" in on:
            used = c.execute("SELECT COALESCE(SUM(amount_cents),0) FROM transfers WHERE src=? AND committed_at_us>? "
                             "AND committed_at_us<=?", (p.src, t - pol.rolling_window_seconds * 1_000_000, t)).fetchone()[0]
            extra = c.execute("SELECT COALESCE(SUM(extra_ceiling_cents),0) FROM grants WHERE account_id=? AND revoked=0 "
                              "AND valid_from_us<=? AND expires_at_us>?", (p.src, t, t)).fetchone()[0]
            limit = pol.transfer_ceiling_cents + extra
            if used + p.amount_cents > limit:
                out.append(Violation(rule="A", detail=f"window total {used + p.amount_cents} exceeds {limit}",
                                     severity_cents=min(used + p.amount_cents - limit, p.amount_cents)))

        labels = [a.label for a in p.approvals]
        marks = ",".join("?" * len(labels))
        init = c.execute("SELECT identity_id FROM aliases WHERE label=?", (p.initiator,)).fetchone()
        init_id = init[0] if init else None

        if "B" in on and p.amount_cents > pol.dual_approval_threshold_cents:
            n = 0
            if labels:
                n = c.execute(f"SELECT COUNT(DISTINCT i.identity_id) FROM aliases a JOIN identities i USING(identity_id) "
                              f"WHERE a.label IN ({marks}) AND i.can_approve=1 AND i.active=1 AND i.identity_id IS NOT ?",
                              (*labels, init_id)).fetchone()[0]
            if n < 2:
                out.append(Violation(rule="B", detail=f"{n} distinct authorized approvals", severity_cents=p.amount_cents))

        if "C" in on and p.project is not None:
            budget = c.execute("SELECT budget_cents FROM projects WHERE project_id=?", (p.project,)).fetchone()
            if budget is None:
                out.append(Violation(rule="C", detail="unknown project", severity_cents=p.amount_cents))
            else:
                spent = c.execute("SELECT COALESCE(SUM(amount_cents),0) FROM transfers WHERE project=?", (p.project,)).fetchone()[0]
                if spent + p.amount_cents > budget[0]:
                    out.append(Violation(rule="C", detail=f"project total {spent + p.amount_cents} exceeds {budget[0]}",
                                         severity_cents=min(spent + p.amount_cents - budget[0], p.amount_cents)))

        if "D" in on and dst is not None and dst[0]:
            out.append(Violation(rule="D", detail="restricted destination", severity_cents=p.amount_cents))

        if "E" in on and labels and init_id is not None:
            hit = c.execute(f"SELECT 1 FROM aliases WHERE label IN ({marks}) AND identity_id=? LIMIT 1", (*labels, init_id)).fetchone()
            if hit:
                out.append(Violation(rule="E", detail="initiator also approved", severity_cents=p.amount_cents))
        return out
