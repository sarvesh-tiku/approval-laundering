"""Real-thread race on PostgreSQL, under real multi-version concurrency control.

Asks whether an "atomic guard" stays atomic once it leaves SQLite's single-writer lock. Runs on a throwaway
cluster inside ~/.cache, on a unix socket, and touches no other database. This schema encodes only the capacity
rules under test, not all six rules, and truth is read from the final table contents after all threads finish.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "postgres_race.jsonl"
PGDIR = Path.home() / ".cache" / "approval-laundering" / "pg"
PORT = 54329
AMOUNT, ONE_FITS = 600_000, 1_000_000  # $6,000 against a $10,000 capacity
MODES = ("check_then_commit", "guard_read_committed", "guard_repeatable_read", "guard_serializable", "lock_source", "lock_invariant", "reserve_unlocked", "reserve_atomic", "reserve_atomic_fixed")
WORKLOADS = ("same_account", "cross_account", "same_account_lawful", "cross_account_lawful")
MAX_RETRIES = 5
SCHEMA = """
CREATE TABLE accounts(id TEXT PRIMARY KEY, balance BIGINT NOT NULL);
CREATE TABLE scopes(id TEXT PRIMARY KEY, capacity BIGINT NOT NULL);
CREATE TABLE transfers(seq BIGSERIAL PRIMARY KEY, worker INT, src TEXT, dst TEXT, amount BIGINT, scope TEXT);
CREATE INDEX ON transfers(scope);
CREATE TABLE holds(worker INT PRIMARY KEY, amount BIGINT, scope TEXT);
"""


def pg_bin(name: str) -> str:
    found = shutil.which(name) or (f"/opt/homebrew/opt/postgresql@15/bin/{name}" if Path(f"/opt/homebrew/opt/postgresql@15/bin/{name}").exists() else None)
    if not found:
        raise SystemExit(f"PostgreSQL binary '{name}' not found: the Postgres race did not run and the paper must not claim it.")
    return found


def start_cluster() -> None:
    PGDIR.parent.mkdir(parents=True, exist_ok=True)
    if not (PGDIR / "PG_VERSION").exists():
        subprocess.run([pg_bin("initdb"), "-D", str(PGDIR), "-U", "lab", "--auth=trust", "-E", "UTF8"], check=True, capture_output=True)
    opts = f"-p {PORT} -k {PGDIR} -c listen_addresses='' -c max_connections=60 -c fsync=off -c synchronous_commit=off -c full_page_writes=off"
    subprocess.run([pg_bin("pg_ctl"), "-D", str(PGDIR), "-w", "-o", opts, "-l", str(PGDIR / "server.log"), "start"], check=True, capture_output=True)


def stop_cluster() -> None:
    subprocess.run([pg_bin("pg_ctl"), "-D", str(PGDIR), "-m", "fast", "stop"], capture_output=True)


def connect(autocommit=True):
    import psycopg
    return psycopg.connect(host=str(PGDIR), port=PORT, user="lab", dbname="postgres", autocommit=autocommit)


def reset(c, workload: str, n: int) -> tuple[int, str]:
    """Same amounts everywhere. `*_lawful` raises capacity so all n fit, so every lost transfer is a false loss."""
    c.execute("TRUNCATE transfers, holds, accounts, scopes RESTART IDENTITY")
    capacity = ONE_FITS if not workload.endswith("lawful") else AMOUNT * n
    for i in range(n):
        c.execute("INSERT INTO accounts VALUES(%s,%s)", (f"SRC-{i}", 10 ** 9))
        c.execute("INSERT INTO accounts VALUES(%s,0)", (f"VEN-{i}",))
    c.execute("INSERT INTO scopes VALUES(%s,%s)", ("LIMIT", capacity))
    return capacity, "LIMIT"


def attempt(c, mode: str, src: str, dst: str, worker: int, latency_s: float) -> bool:
    """One attempt at one transfer. Returns whether it committed. Serialization errors propagate to the retry loop."""
    def used(cur, with_holds=False):
        s = cur.execute("SELECT COALESCE(SUM(amount),0)::bigint FROM transfers WHERE scope='LIMIT'").fetchone()[0]
        if with_holds:
            s += cur.execute("SELECT COALESCE(SUM(amount),0)::bigint FROM holds WHERE scope='LIMIT'").fetchone()[0]
        return s

    def used_one_snapshot(cur):
        """Transfers and holds read in one statement. Two statements at READ COMMITTED can miss a hold that
        becomes a transfer between them, since the commit step moves it without taking the scope lock."""
        return cur.execute("SELECT (SELECT COALESCE(SUM(amount),0) FROM transfers WHERE scope='LIMIT')::bigint + "
                           "(SELECT COALESCE(SUM(amount),0) FROM holds WHERE scope='LIMIT')::bigint").fetchone()[0]

    def cap(cur):
        return cur.execute("SELECT capacity FROM scopes WHERE id='LIMIT'").fetchone()[0]

    def write(cur):
        cur.execute("UPDATE accounts SET balance=balance-%s WHERE id=%s", (AMOUNT, src))
        cur.execute("UPDATE accounts SET balance=balance+%s WHERE id=%s", (AMOUNT, dst))
        cur.execute("INSERT INTO transfers(worker,src,dst,amount,scope) VALUES(%s,%s,%s,%s,'LIMIT')", (worker, src, dst, AMOUNT))

    def nap():
        if latency_s:
            time.sleep(latency_s)

    if mode == "check_then_commit":  # correct rules on a fresh read, outside any write transaction
        c.autocommit = True
        ok = used(c) + AMOUNT <= cap(c)
        nap()
        if not ok:
            return False
        with c.transaction():
            write(c)
        return True
    if mode in ("reserve_unlocked", "reserve_atomic", "reserve_atomic_fixed"):
        c.autocommit = True
        with c.transaction():
            if mode != "reserve_unlocked":
                c.execute("SELECT 1 FROM scopes WHERE id='LIMIT' FOR UPDATE")
            ok = (used_one_snapshot(c) if mode == "reserve_atomic_fixed" else used(c, True)) + AMOUNT <= cap(c)
            if ok:
                c.execute("INSERT INTO holds VALUES(%s,%s,'LIMIT')", (worker, AMOUNT))
        nap()  # the review happens here, after capacity is held
        if not ok:
            return False
        with c.transaction():
            write(c)
            c.execute("DELETE FROM holds WHERE worker=%s", (worker,))
        return True
    level = {"guard_read_committed": "READ COMMITTED", "guard_repeatable_read": "REPEATABLE READ", "guard_serializable": "SERIALIZABLE", "lock_source": "READ COMMITTED", "lock_invariant": "READ COMMITTED"}[mode]
    c.autocommit = True
    with c.transaction():
        c.execute(f"SET TRANSACTION ISOLATION LEVEL {level}")
        if mode == "lock_source":
            c.execute("SELECT 1 FROM accounts WHERE id=%s FOR UPDATE", (src,))
        elif mode == "lock_invariant":
            c.execute("SELECT 1 FROM scopes WHERE id='LIMIT' FOR UPDATE")
        ok = used(c) + AMOUNT <= cap(c)
        nap()
        if ok:
            write(c)
    return ok


def trial(conns, admin, mode: str, workload: str, n: int, latency_ms: float, tr: int, sequential: bool = False, jitter_ms: float = 0.0) -> dict:
    """`sequential` is the validity control: one thread at a time, which must never overshoot.
    `jitter_ms` staggers arrivals uniformly, because the barrier alone is a worst case for overlap."""
    import psycopg
    capacity, _ = reset(admin, workload, n)
    barrier = threading.Barrier(1 if sequential else n)
    rng = random.Random(f"{mode}/{workload}/{n}/{latency_ms}/{tr}/{jitter_ms}")
    delays = [rng.uniform(0, jitter_ms / 1000) for _ in range(n)]
    same = workload.startswith("same_account")
    results: list[dict] = [{} for _ in range(n)]

    def worker(i: int):
        c, retries, ok, gave_up = conns[i], 0, False, False
        src, dst = ("SRC-0" if same else f"SRC-{i}"), f"VEN-{i}"  # distinct vendors, so no row is shared by accident
        barrier.wait()
        if delays[i]:
            time.sleep(delays[i])
        t0 = time.perf_counter()
        while True:
            try:
                ok = attempt(c, mode, src, dst, i, latency_ms / 1000)
                break
            except (psycopg.errors.SerializationFailure, psycopg.errors.DeadlockDetected):
                retries += 1
                if retries > MAX_RETRIES:
                    gave_up = True
                    break
                time.sleep(0.001 * retries)
        results[i] = {"ok": ok, "retries": retries, "gave_up": gave_up, "seconds": time.perf_counter() - t0}

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    if sequential:
        for t in threads:
            t.start()
            t.join()
    else:
        [t.start() for t in threads]
        [t.join() for t in threads]
    commits, total = admin.execute("SELECT COUNT(*), COALESCE(SUM(amount),0)::bigint FROM transfers").fetchone()
    conserved = admin.execute("SELECT COALESCE(SUM(balance),0)::bigint FROM accounts").fetchone()[0]
    expected = (10 ** 9) * n
    return {"mode": mode, "workload": workload, "threads": n, "latency_ms": latency_ms, "trial": tr, "sequential": sequential, "jitter_ms": jitter_ms, "commits": commits, "overshoot": total > capacity, "overshoot_cents": max(0, total - capacity),
            "lost_lawful": (n - commits) if workload.endswith("lawful") else 0, "retries": sum(r["retries"] for r in results), "gave_up": sum(r["gave_up"] for r in results),
            "max_seconds": max(r["seconds"] for r in results), "conserved": conserved == expected}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--threads", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--latency", type=float, nargs="+", default=[0, 2])
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sequential", action="store_true", help="validity control: one thread at a time")
    ap.add_argument("--jitter-ms", type=float, default=0.0)
    args = ap.parse_args()
    global MAX_RETRIES
    MAX_RETRIES = args.max_retries
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(d["mode"], d["workload"], d["threads"], d["latency_ms"], d["trial"]) for d in map(json.loads, f)}
    start_cluster()
    new = 0
    try:
        admin = connect()
        admin.execute("DROP TABLE IF EXISTS transfers, holds, accounts, scopes")
        admin.execute(SCHEMA)
        conns = [connect() for _ in range(max(args.threads))]
        with open(out, "a") as f:
            for mode, wl, n, lat in itertools.product(args.modes, args.workloads, args.threads, args.latency):
                for tr in range(args.trials):
                    if (mode, wl, n, float(lat), tr) in done or (mode, wl, n, lat, tr) in done:
                        continue
                    f.write(json.dumps(trial(conns, admin, mode, wl, n, lat, tr, args.sequential, args.jitter_ms)) + "\n")
                    new += 1
                f.flush()
        version = admin.execute("SHOW server_version").fetchone()[0]
        (ROOT / "results" / "postgres_env.json").write_text(json.dumps({"server_version": version, "fsync": "off", "synchronous_commit": "off"}))
        for c in conns:
            c.close()
        admin.close()
    finally:
        stop_cluster()
    print(f"postgres race: {new} new trials, {len(done)} already stored")


if __name__ == "__main__":
    main()
