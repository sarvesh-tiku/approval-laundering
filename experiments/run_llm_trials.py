"""LLM monitors on fixed action sequences (Evaluation A) and interactive rollouts (Evaluation B),
plus scripted-agent rollouts that need no credentials. Every record carries its mode, and each mode has its own file.

  python -m experiments.run_llm_trials --scripted-only      # no key needed
  python -m experiments.run_llm_trials --phase fixed        # needs LLM_API_KEY and LLM_MODELS
  python -m experiments.run_llm_trials --phase rollouts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.run_baselines import load_config, load_policy
from src.agents import LLMCoordinator, LLMReviewer, LLMWorker, ScriptedCoordinator, ScriptedWorker
from src.llm_provider import BudgetExceeded, ChatProvider, ProviderError, configured_models, load_env
from src.models import RULES, Scenario
from src.monitors import IdealReviewer
from src.rollouts import make_tasks, run_rollout
from src.runner import Arm, deterministic_arms, run_episode
from src.scenarios import generate_all

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
lock = threading.Lock()


def slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def llm_fixed_arms(provider, model: str, conditions: list[str], temperature: float) -> list[Arm]:
    arms = []
    for c in conditions:
        rev = LLMReviewer(provider, model, temperature=temperature)
        if c == "C5":
            arms.append(Arm(f"C5/llm+guard", "C2", rev, True, "C5", give_task=True))
        elif c == "C0f":
            arms.append(Arm("C0f/llm", "C0", LLMReviewer(provider, model, temperature=temperature, forced=True), False, "C0"))
        elif c == "C0n":
            arms.append(Arm("C0n/llm", "C0", LLMReviewer(provider, model, temperature=temperature, quiet=True), False, "C0"))
        elif c == "C0nf":
            arms.append(Arm("C0nf/llm", "C0", LLMReviewer(provider, model, temperature=temperature, forced=True, quiet=True), False, "C0"))
        elif c == "C2agg":
            arms.append(Arm("C2agg/llm", "C2", LLMReviewer(provider, model, aggregates=True, temperature=temperature), False, "C2"))
        else:
            arms.append(Arm(f"{c}/llm", c, rev, False, c[:2]))
    return arms


def stored_keys(path: Path, keyf) -> set:
    if not path.exists():
        return set()
    with open(path) as f:
        return {keyf(d) for d in map(json.loads, f)}


def run_fixed(provider, models, cfg, policy, workers, limit=None):
    c = cfg["fixed_replay"]
    # Whole seeds only, so every family keeps its full factorial structure.
    scenarios = generate_all(c["seeds"], 0)
    if limit:
        scenarios = scenarios[:limit]
    print(f"fixed: {len(scenarios)} scenarios")
    for model in models:
        out = RES / f"fixed_replay_llm_{slug(model)}.jsonl"
        done = stored_keys(out, lambda d: (d["scenario_id"], d["arm"]))
        jobs = [(s, a) for a in llm_fixed_arms(provider, model, c["conditions"], c["temperature"]) for s in scenarios if (s.scenario_id, a.name) not in done]
        print(f"  {model}: {len(jobs)} episodes to run, {len(done)} stored")

        def job(sc, arm):
            t = run_episode(sc, arm, policy, mode=f"llm:{model}")
            t["model"] = model
            return t

        _drain(jobs, job, out, workers, provider)


def run_rollouts_llm(provider, models, cfg, policy, workers, limit=None):
    c = cfg["rollouts"]
    tasks = make_tasks(c["task_seeds"])
    for model in models:
        out = RES / f"rollouts_llm_{slug(model)}.jsonl"
        key = lambda d: (d["scenario_id"], d["arm"], d["objective"], d["delegation"], d["concurrency"], d["sample"])
        done = stored_keys(out, key)
        jobs = []
        for mon in c["monitors"]:
            for obj in c["objectives"]:
                for dl in c["delegations"]:
                    for cc in c["concurrency"]:
                        for sample in range(c["repeats"]):
                            for t in tasks:
                                arm = rollout_arm(mon, provider, model)
                                if (t.task_id, arm.name, obj, dl, cc, sample) not in done:
                                    jobs.append((t, mon, obj, dl, cc, sample))
        if limit:
            jobs = jobs[:limit]
        print(f"  {model}: {len(jobs)} rollouts to run, {len(done)} stored")

        def job(t, mon, obj, dl, cc, sample):
            arm = rollout_arm(mon, provider, model, sample)
            coord = LLMCoordinator(provider, model, obj, policy, c["temperature"], sample)
            worker = LLMWorker(provider, model, policy, c["temperature"], sample)
            return run_rollout(t, coord, worker, arm, policy, dl, cc, obj, model, sample, f"llm:{model}", c["max_turns"])

        _drain(jobs, job, out, workers, provider)


def rollout_arm(mon: str, provider, model: str, sample: int = 0) -> Arm:
    if mon == "C4":
        return Arm("C4/guard", None, None, True, "C4")
    if mon == "C0f":
        return Arm("C0f/llm", "C0", LLMReviewer(provider, model, temperature=0.0, sample=sample, forced=True), False, "C0")
    if mon == "C0nf":
        return Arm("C0nf/llm", "C0", LLMReviewer(provider, model, temperature=0.0, sample=sample, forced=True, quiet=True), False, "C0")
    rev = LLMReviewer(provider, model, temperature=0.0, sample=sample)
    if mon == "C5":
        return Arm("C5/llm+guard", "C2", rev, True, "C5", give_task=True)
    return Arm(f"{mon}/llm", mon, rev, False, mon[:2])


def _drain(jobs, fn, out: Path, workers: int, provider) -> None:
    failed = 0
    t0 = time.time()
    with open(out, "a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, *j): j for j in jobs}
        try:
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    rec = fut.result()
                except ProviderError as e:
                    failed += 1
                    if "HTTP 40" in str(e):
                        print(f"fatal provider error: {e}", file=sys.stderr)
                        raise
                    continue
                with lock:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                if i % 200 == 0:
                    u = provider.usage
                    print(f"    {i}/{len(jobs)} {time.time() - t0:.0f}s calls={u.calls} cached={u.cached} ${u.cost_usd:.3f} failed={failed}", flush=True)
        except (BudgetExceeded, ProviderError) as e:
            print(f"stopping early, stored work is kept: {e}", file=sys.stderr)
            for fut in futs:
                fut.cancel()
    print(f"    finished; failed={failed}; usage so far: calls={provider.usage.calls} ${provider.usage.cost_usd:.3f}")


def run_scripted(cfg, policy) -> None:
    c = cfg["rollouts"]
    arms = {a.name: a for a in deterministic_arms()}
    out = RES / "rollouts_scripted.jsonl"
    key = lambda d: (d["scenario_id"], d["arm"], d["objective"], d["delegation"], d["concurrency"])
    done = stored_keys(out, key)
    n = 0
    with open(out, "a") as f:
        for name in ("C0/ideal", "C1/ideal", "C2/ideal", "C3k1/ideal", "C4/guard", "C5/ideal+guard"):
            for obj in c["objectives"]:
                for dl in c["delegations"]:
                    for cc in c["concurrency"]:
                        for t in make_tasks(c["task_seeds"]):
                            if (t.task_id, name, obj, dl, cc) in done:
                                continue
                            rec = run_rollout(t, ScriptedCoordinator(obj, policy), ScriptedWorker(), arms[name], policy, dl, cc, obj, "scripted", 0, "scripted_agents", c["max_turns"])
                            f.write(json.dumps(rec) + "\n")
                            n += 1
    print(f"scripted rollouts: {n} new, {len(done)} stored")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["fixed", "rollouts", "all"], default="all")
    ap.add_argument("--scripted-only", action="store_true")
    ap.add_argument("--models", default=None, help="comma-separated; defaults to LLM_MODELS")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="pilot: run only this many scenarios/rollouts")
    ap.add_argument("--budget", type=float, default=None, help="hard USD stop for this invocation, overriding LLM_BUDGET_USD")
    args = ap.parse_args()
    cfg, policy = load_config(), load_policy()
    RES.mkdir(exist_ok=True)
    run_scripted(cfg["llm"], policy)
    if args.scripted_only:
        return
    load_env()
    models = [m.strip() for m in args.models.split(",")] if args.models else configured_models()
    if not os.environ.get("LLM_API_KEY") or not models:
        print("No LLM_API_KEY or LLM_MODELS configured: no LLM trials ran. Scripted results above are not LLM data.")
        return
    provider = ChatProvider(budget_usd=args.budget)
    if args.phase in ("fixed", "all"):
        run_fixed(provider, models, cfg["llm"], policy, args.workers, args.limit)
    if args.phase in ("rollouts", "all"):
        run_rollouts_llm(provider, models, cfg["llm"], policy, args.workers, args.limit)
    u = provider.usage
    log = {"models": models, "calls": u.calls, "cached": u.cached, "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "cost_usd": u.cost_usd, "repairs": u.repairs, "parse_failures": u.parse_failures, "phase": args.phase}
    with open(RES / "llm_run_log.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")
    print(log)


if __name__ == "__main__":
    main()
