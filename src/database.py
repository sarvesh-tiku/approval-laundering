"""Authoritative SQLite ledger. Agents and monitors never hold a Database; they get value-object snapshots."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

from .models import Account, Grant, Identity, Project, Scenario, Snapshot, TransferRecord

SCHEMA = """
CREATE TABLE accounts(account_id TEXT PRIMARY KEY, owner TEXT NOT NULL, kind TEXT NOT NULL,
                      balance_cents INTEGER NOT NULL, restricted INTEGER NOT NULL);
CREATE TABLE projects(project_id TEXT PRIMARY KEY, budget_cents INTEGER NOT NULL);
CREATE TABLE identities(identity_id TEXT PRIMARY KEY, can_approve INTEGER NOT NULL, active INTEGER NOT NULL);
CREATE TABLE aliases(label TEXT PRIMARY KEY, identity_id TEXT NOT NULL REFERENCES identities(identity_id));
CREATE TABLE grants(grant_id TEXT PRIMARY KEY, account_id TEXT NOT NULL, extra_ceiling_cents INTEGER NOT NULL,
                    valid_from_us INTEGER NOT NULL, expires_at_us INTEGER NOT NULL, revoked INTEGER NOT NULL);
CREATE TABLE transfers(seq INTEGER PRIMARY KEY AUTOINCREMENT, proposal_id TEXT NOT NULL, initiator TEXT NOT NULL,
                       src TEXT NOT NULL, dst TEXT NOT NULL, amount_cents INTEGER NOT NULL, project TEXT,
                       committed_at_us INTEGER NOT NULL);
CREATE INDEX transfers_src ON transfers(src, committed_at_us);
"""
UTC = timezone.utc


def to_us(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise ValueError("naive datetime")
    return round(dt.timestamp() * 1_000_000)


def from_us(us: int) -> datetime:
    return datetime.fromtimestamp(us / 1_000_000, tz=UTC)


class Database:
    def __init__(self, path: str = ":memory:", timeout: float = 30.0):
        self.path = path
        # isolation_level=None: transactions are explicit, so BEGIN IMMEDIATE means what it says.
        self.conn = sqlite3.connect(path, isolation_level=None, timeout=timeout, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys=ON")

    @classmethod
    def from_scenario(cls, s: Scenario, path: str = ":memory:") -> "Database":
        db = cls(path)
        db.conn.executescript(SCHEMA)
        db.conn.execute("BEGIN")
        for a in s.accounts:
            db.conn.execute("INSERT INTO accounts VALUES(?,?,?,?,?)",
                            (a.account_id, a.owner, a.kind, a.balance_cents, int(a.restricted)))
        for p in s.projects:
            db.conn.execute("INSERT INTO projects VALUES(?,?)", (p.project_id, p.budget_cents))
        for i in s.identities:
            db.conn.execute("INSERT INTO identities VALUES(?,?,?)", (i.identity_id, int(i.can_approve), int(i.active)))
            for lab in {i.identity_id, *i.labels}:
                db.conn.execute("INSERT INTO aliases VALUES(?,?)", (lab, i.identity_id))
        for g in s.grants:
            db.insert_grant(g)
        for t in s.prior_transfers:  # history predating the episode; balances are already net of it
            db.conn.execute("INSERT INTO transfers(proposal_id,initiator,src,dst,amount_cents,project,committed_at_us)"
                            " VALUES(?,?,?,?,?,?,?)", (t.proposal_id, t.initiator, t.src, t.dst, t.amount_cents,
                                                       t.project, to_us(t.committed_at)))
        db.conn.execute("COMMIT")
        return db

    def close(self):
        self.conn.close()

    def insert_grant(self, g: Grant):
        self.conn.execute("INSERT INTO grants VALUES(?,?,?,?,?,?)", (g.grant_id, g.account_id, g.extra_ceiling_cents,
                                                                     to_us(g.valid_from), to_us(g.expires_at), int(g.revoked)))

    def snapshot(self) -> Snapshot:
        """Full authoritative read inside one read transaction, returned as an immutable value."""
        c = self.conn
        c.execute("BEGIN")
        try:
            accts = c.execute("SELECT account_id,balance_cents,restricted FROM accounts").fetchall()
            budgets = dict(c.execute("SELECT project_id,budget_cents FROM projects").fetchall())
            transfers = tuple(TransferRecord(proposal_id=r[0], initiator=r[1], src=r[2], dst=r[3], amount_cents=r[4],
                                             project=r[5], committed_at=from_us(r[6])) for r in c.execute(
                "SELECT proposal_id,initiator,src,dst,amount_cents,project,committed_at_us FROM transfers ORDER BY seq"))
            grants = tuple(Grant(grant_id=r[0], account_id=r[1], extra_ceiling_cents=r[2], valid_from=from_us(r[3]),
                                 expires_at=from_us(r[4]), revoked=bool(r[5])) for r in c.execute(
                "SELECT grant_id,account_id,extra_ceiling_cents,valid_from_us,expires_at_us,revoked FROM grants ORDER BY grant_id"))
            labels: dict[str, list[str]] = {}
            for lab, iid in c.execute("SELECT label,identity_id FROM aliases ORDER BY label"):
                labels.setdefault(iid, []).append(lab)
            identities = tuple(Identity(identity_id=r[0], can_approve=bool(r[1]), active=bool(r[2]),
                                        labels=tuple(labels.get(r[0], ()))) for r in c.execute(
                "SELECT identity_id,can_approve,active FROM identities ORDER BY identity_id"))
        finally:
            c.execute("COMMIT")
        return Snapshot(balances={a[0]: a[1] for a in accts}, known_accounts=tuple(sorted(a[0] for a in accts)),
                        budgets=budgets, restricted=tuple(sorted(a[0] for a in accts if a[2])),
                        transfers=transfers, grants=grants, identities=identities)

    def state_hash(self) -> str:
        """Canonical digest of authoritative state, used to prove arms start from identical worlds."""
        rows = {}
        for table, order in [("accounts", "account_id"), ("projects", "project_id"), ("identities", "identity_id"),
                             ("aliases", "label"), ("grants", "grant_id"), ("transfers", "seq")]:
            rows[table] = self.conn.execute(f"SELECT * FROM {table} ORDER BY {order}").fetchall()
        return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()

    def total_balance(self) -> int:
        return self.conn.execute("SELECT COALESCE(SUM(balance_cents),0) FROM accounts").fetchone()[0]


def read_only_connection(path: str) -> sqlite3.Connection:
    """A connection that cannot write, for external inspection of a file-backed ledger."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn
