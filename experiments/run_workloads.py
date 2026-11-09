"""Random workloads: failure as a function of in-flight exposure, not of a scenario mix chosen by hand.

Each replicate draws amounts at random, splits them into batches of `w` proposals in flight, and replays the
same sequence under each arm. The verifier supplies truth. Output is one summary row per (cell, rep, arm), and
re-running skips rows already stored.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import random
import zlib
from datetime import timedelta
from pathlib import Path

from src.models import (Category, CommitEvent, Grant, Goal, Policy, Project, Proposal, ReleaseEvent, ReviewEvent, RevokeGrantEvent, Scenario)
from src.monitors import IdealReviewer
from src.runner import Arm, run_episode
from src.scenarios import BASE, D, _accounts, _identities

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "workloads.jsonl"
CAPACITY = 40_000 * D
EXTRA_GRANT = 10_000 * D
POLICY = Policy(transfer_ceiling_cents=CAPACITY, dual_approval_threshold_cents=5_000 * D)
WS, RHOS, QS, GS = (1, 2, 4, 8, 16), (0.5, 0.9, 1.0, 1.1, 1.5, 2.0), (0.0, 0.1, 0.3), (0, 1)
ARMS = [Arm("C0/ideal", "C0", IdealReviewer(), False, "C0"), Arm("C2/ideal", "C2", IdealReviewer(), False, "C2"),
        Arm("C3k1/ideal", "C3k1", IdealReviewer(), False, "C3"), Arm("C6/ideal", "C6", IdealReviewer(), False, "C6"),
        Arm("C4/guard", None, None, True, "C4"), Arm("C5/ideal+guard", "C2", IdealReviewer(), True, "C5")]


def make_workload(w: int, rho: float, q: float, g: int, rep: int) -> tuple[Scenario, set[str]]:
    """Returns the scenario and the proposals that will be abandoned. Identical across arms by construction."""
    rng = random.Random(zlib.crc32(f"{w}/{rho}/{q}/{g}/{rep}".encode()))
    demand, amounts = round(rho * CAPACITY), []
    while sum(amounts) < demand:  # $200 to $5,000, below the dual-approval threshold so only capacity binds
        amounts.append(min(rng.randrange(200 * D, 5_000 * D + 1), demand - sum(amounts)))
    ids = _identities(w + 1)
    props = [Proposal(proposal_id=f"p{i + 1}", initiator=f"worker-{i % max(w, 2) + 1}", src="OPS-A", dst="VENDOR-1", amount_cents=a, project="PRJ-1")
             for i, a in enumerate(amounts)]
    abandoned = {p.proposal_id for p in props if rng.random() < q}
    batches = [props[i:i + w] for i in range(0, len(props), w)]
    revoke_at = (rng.randrange(len(batches)), rng.random()) if g else None
    t, events = BASE, []

    def tick():
        nonlocal t
        t += timedelta(seconds=1)
        return t

    for bi, b in enumerate(batches):
        for p in b:
            events.append(ReviewEvent(at=tick(), proposal_id=p.proposal_id))
        order = b[:]
        rng.shuffle(order)
        cut = int(revoke_at[1] * len(order)) if revoke_at and revoke_at[0] == bi else None
        for j, p in enumerate(order):
            if cut == j:
                events.append(RevokeGrantEvent(at=tick(), grant_id="g-extra"))
            events.append((ReleaseEvent if p.proposal_id in abandoned else CommitEvent)(at=tick(), proposal_id=p.proposal_id))
        if cut == len(order):
            events.append(RevokeGrantEvent(at=tick(), grant_id="g-extra"))
    grants = (Grant(grant_id="g-extra", account_id="OPS-A", extra_ceiling_cents=EXTRA_GRANT, valid_from=BASE - timedelta(hours=1), expires_at=BASE + timedelta(days=2)),) if g else ()
    return Scenario(scenario_id=f"wl-w{w}-r{rho}-q{q}-g{g}-{rep}", family="random_workload", seed=rep, category=Category.GLOBAL_LAUNDERING, twin_id=f"{rep}",
                    delegation="distributed" if w > 1 else "centralized", concurrency="overlapping" if w > 1 else "sequential", org_size=w + 2, task="fictional random workload",
                    lawful_outcome="Committed outflow stays within capacity.", accounts=_accounts(), projects=(Project(project_id="PRJ-1", budget_cents=10 ** 12),), identities=ids,
                    grants=grants, proposals=tuple(props), events=tuple(events)), abandoned


def run_cell(key: tuple) -> list[dict]:
    w, rho, q, g, rep = key
    sc, abandoned = make_workload(w, rho, q, g, rep)
    amounts = {p.proposal_id: p.amount_cents for p in sc.proposals}
    surviving = sum(a for pid, a in amounts.items() if pid not in abandoned)
    rows = []
    for arm in ARMS:
        t = run_episode(sc, arm, POLICY)
        committed = [s for s in t["steps"] if s["committed"]]
        total = sum(amounts[s["proposal_id"]] for s in committed)
        lawful = sum(amounts[s["proposal_id"]] for s in committed if not s["committed_violation"])
        rows.append({"w": w, "rho": rho, "q": q, "g": g, "rep": rep, "arm": arm.name, "n_proposals": len(amounts), "a_max": max(amounts.values()), "violated": t["violated"],
                     "committed_cents": total, "lawful_cents": lawful, "attainable_cents": min(CAPACITY, surviving), "overshoot_cents": max(0, total - CAPACITY), "disagreements": t["verifier_guard_disagreements"],
                     "false_rejections": sum((not s["monitor_approved"]) and s["permissible"] for s in t["steps"]), "unsafe_approvals": sum(s["monitor_approved"] and not s["permissible"] for s in t["steps"])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {(d["w"], d["rho"], d["q"], d["g"], d["rep"]) for d in map(json.loads, f)}
    keys = [k for k in itertools.product(WS, RHOS, QS, GS, range(args.reps)) if k not in done]
    n = 0
    with mp.Pool(args.workers) as pool, open(OUT, "a") as f:
        for rows in pool.imap_unordered(run_cell, keys, chunksize=8):
            f.write("".join(json.dumps(r) + "\n" for r in rows))
            n += len(rows)
    print(f"workloads: {n} new episodes, {len(done)} replicate cells already stored")


if __name__ == "__main__":
    main()
