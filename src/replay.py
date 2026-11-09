"""Counterfactual replay: same initial state and same proposal sequence, different oversight.

Replaying holds the actions fixed. For a trace from an adaptive rollout it answers "what would this
oversight have done to these exact actions", not "what would the agents have done under this oversight".
Every replayed trace records its origin in `replayed_from` so the two are never pooled.
"""
from __future__ import annotations

from .models import Policy, Scenario
from .runner import Arm, run_episode


class InitialStateMismatch(RuntimeError):
    pass


def replay(trace: dict, scenario: Scenario, arm: Arm, policy: Policy) -> dict:
    origin = f"{trace.get('mode', 'unknown')}:{trace['arm']}"
    new = run_episode(scenario, arm, policy, mode="counterfactual_replay", replayed_from=origin)
    if new["initial_hash"] != trace["initial_hash"]:
        raise InitialStateMismatch(f"{trace['scenario_id']}: replay does not start from the recorded state")
    return new


def first_violating_transition(trace: dict) -> dict | None:
    for s in trace["steps"]:
        if s["committed_violation"]:
            return {"proposal_id": s["proposal_id"], "commit_at": s["commit_at"], "rules": sorted({v["rule"] for v in s["truth"]}),
                    "post_hash": s["post_hash"]}
    return None


def differential(base: dict, alt: dict) -> dict:
    """Which decisions and commits changed between two runs of the same action sequence?"""
    if base["initial_hash"] != alt["initial_hash"]:
        raise InitialStateMismatch(base["scenario_id"])
    a = {(s["proposal_id"], s["attempt"]): s for s in base["steps"]}
    b = {(s["proposal_id"], s["attempt"]): s for s in alt["steps"]}
    changes = []
    for key in a.keys() & b.keys():
        x, y = a[key], b[key]
        # "approve" and "forwarded" both let a proposal proceed; only a real change of outcome is reported.
        passed = lambda st: st["decision"] in ("approve", "forwarded")
        if passed(x) != passed(y) or x["committed"] != y["committed"]:
            changes.append({"proposal_id": key[0], "attempt": key[1], "from_decision": x["decision"], "to_decision": y["decision"],
                            "from_committed": x["committed"], "to_committed": y["committed"], "to_guard": y["guard"],
                            "truth_under_alt": [v["rule"] for v in y["truth"]], "reason_alt": y["reason"]})
    return {"scenario_id": base["scenario_id"], "from_arm": base["arm"], "to_arm": alt["arm"], "changes": sorted(changes, key=lambda c: c["proposal_id"]),
            "violated_from": base["violated"], "violated_to": alt["violated"], "first_violation_from": first_violating_transition(base),
            "first_violation_to": first_violating_transition(alt)}
