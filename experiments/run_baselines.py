"""Evaluation A with deterministic reference reviewers, plus a real-thread race experiment.

Output is append-only JSONL. Re-running skips (scenario, arm) pairs and (cell, trial) pairs already stored.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import yaml

from src.database import Database
from src.executor import Executor
from src.models import Policy, Scenario
from src.policy_engine import evaluate
from src.replay import differential, first_violating_transition
from src.runner import deterministic_arms, run_episode
from src.scenarios import generate_all
from src.scenarios import race_world

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def load_policy() -> Policy:
    cfg = yaml.safe_load((ROOT / "configs/policies.yaml").read_text())["default"]
    cfg["enabled_rules"] = tuple(cfg["enabled_rules"])
    return Policy(**cfg)


def load_config() -> dict:
    return yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())


def run_fixed(n_seeds: int, base_seed: int, policy: Policy) -> int:
    RES.mkdir(exist_ok=True)
    scenarios = generate_all(n_seeds, base_seed)
    with open(RES / "scenarios.jsonl", "w") as f:
        for s in scenarios:
            f.write(s.model_dump_json() + "\n")
    out = RES / "fixed_replay_deterministic.jsonl"
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(d["scenario_id"], d["arm"]) for d in map(json.loads, f)}
    new = 0
    with open(out, "a") as f:
        for arm in deterministic_arms():
            for sc in scenarios:
                if (sc.scenario_id, arm.name) in done:
                    continue
                f.write(json.dumps(run_episode(sc, arm, policy)) + "\n")
                new += 1
    print(f"fixed replay: {len(scenarios)} scenarios x {len(deterministic_arms())} arms; {new} new episodes, {len(done)} already stored")
    return new


def differential_report(policy: Policy) -> None:
    by_arm: dict[str, dict] = {}
    with open(RES / "fixed_replay_deterministic.jsonl") as f:
        for d in map(json.loads, f):
            by_arm.setdefault(d["arm"], {})[d["scenario_id"]] = d
    base = by_arm["C0/ideal"]
    with open(RES / "differential_C0.jsonl", "w") as f:
        for arm in ("C1/ideal", "C2/ideal", "C4/guard", "C5/ideal+guard"):
            for sid, t in base.items():
                d = differential(t, by_arm[arm][sid])
                if d["changes"]:
                    f.write(json.dumps(d) + "\n")
    with open(RES / "first_violations_C0.jsonl", "w") as f:
        for sid, t in base.items():
            fv = first_violating_transition(t)
            if fv:
                f.write(json.dumps({"scenario_id": sid, **fv}) + "\n")


def _race_trial(mode: str, n_threads: int, latency_ms: float, trial: int, policy: Policy, tmp: str) -> dict:
    """n workers each try to move a $6,000 transfer out of one account; the ceiling admits only one."""
    path = os.path.join(tmp, f"{mode}-{n_threads}-{latency_ms}-{trial}.db")
    sc, proposals, at = race_world(n_threads)
    Database.from_scenario(sc, path).close()
    barrier = threading.Barrier(n_threads)
    wins: list[int] = []
    lock = threading.Lock()

    def worker(p):
        db = Database(path)
        ex = Executor(db, policy)
        barrier.wait()
        if mode == "snapshot_check_then_commit":
            ok = not evaluate(db.snapshot(), p, at, policy)  # correct rules on current state, read before the write
            if latency_ms:
                time.sleep(latency_ms / 1000)  # the review step takes time; here it is short and optimistic
            if ok:
                ex.commit_unchecked(p, at)
        else:
            ok, _ = ex.commit_guarded(p, at)
        with lock:
            wins.append(int(ok))
        db.close()

    threads = [threading.Thread(target=worker, args=(p,)) for p in proposals]
    [t.start() for t in threads]
    [t.join() for t in threads]
    db = Database(path)
    n = db.conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0]
    db.close()
    os.remove(path)
    for suffix in ("-wal", "-shm", "-journal"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)
    return {"mode": mode, "threads": n_threads, "latency_ms": latency_ms, "trial": trial, "commits": n, "violated": n > 1}


def run_race(cfg: dict, policy: Policy) -> None:
    out = RES / "thread_race.jsonl"
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(d["mode"], d["threads"], d["latency_ms"], d["trial"]) for d in map(json.loads, f)}
    # A real file ledger, outside iCloud-synced folders.
    cache = Path.home() / ".cache" / "approval-laundering"
    cache.mkdir(parents=True, exist_ok=True)
    new = 0
    with tempfile.TemporaryDirectory(dir=cache) as tmp, open(out, "a") as f:
        for mode in ("snapshot_check_then_commit", "atomic_guard"):
            for nt in cfg["thread_counts"]:
                for lat in cfg["review_latency_ms"]:
                    for tr in range(cfg["trials_per_cell"]):
                        if (mode, nt, lat, tr) in done:
                            continue
                        f.write(json.dumps(_race_trial(mode, nt, lat, tr, policy, tmp)) + "\n")
                        new += 1
    print(f"thread race: {new} new trials, {len(done)} already stored")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--skip-race", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    policy = load_policy()
    run_fixed(args.seeds or cfg["baselines"]["n_seeds"], cfg["baselines"]["base_seed"], policy)
    differential_report(policy)
    if not args.skip_race:
        run_race(cfg["thread_race"], policy)


if __name__ == "__main__":
    main()
