"""
Phase 2a of the scaled-up cost-substitution verification: GPT-4o, n=20,
all three CostBench pairings, gate_version="v3" held FIXED, only
cost_substitution True/False varying. Tests whether the substitution
opportunity Phase 1 found (gpt-4o-mini reaching for costlier composite
"Full_Planning"/"Refine_StepA_to_StepB" tools over cheaper equivalent-
output atomic ones) is specific to a weaker model, or a general property
of the tool registry any model's free-choice batching would trigger.

Same invariant-check caveat established in Phase 1: the live round-count/
cost checks below are confounded by cross-call LLM stochasticity at
temp=0 (two separate live episodes, not a controlled replay) and WILL
show some violation rate regardless of code correctness -- that's now
understood, not a red flag by itself. The real safety evidence is the
offline replay (experiments/noise_checks/offline_replay_cost_substitution_invariants.py),
which should be re-pointed at this run's own output file afterward.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep_conditions

CONDITIONS = {
    "v3_no_cost_sub": dict(allow_batch=True, safety_mode="gated", gate_version="v3", cost_substitution=False),
    "v3_cost_aware": dict(allow_batch=True, safety_mode="gated", gate_version="v3", cost_substitution=True),
}

PAIRINGS = [
    ("location", ["location", "transportation"], "location+transportation"),
    ("location", ["location", "accommodation"], "location+accommodation"),
    ("travel", ["location", "transportation", "dining"], "location+transportation+dining"),
]

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    for task, categories, label in PAIRINGS:
        print(f"\n########## {label} ##########")
        seed_results = run_llm_sweep_conditions(
            task=task, categories=categories,
            label=f"{label} (v3 with vs without cost-substitution, n=20, gpt-4o)",
            conditions=CONDITIONS, reference="v3_no_cost_sub",
            model="gpt-4o", seeds=range(1, 21))

        print(f"\n=== [{label}] ROUND-COUNT INVARIANT CHECK (live, confounded by cross-call stochasticity) ===")
        mismatches = [r["seed"] for r in seed_results
                      if r["v3_no_cost_sub_calls"] != r["v3_cost_aware_calls"]]
        if mismatches:
            print(f"differed on seed(s): {mismatches} (expected some divergence -- see offline replay for the "
                  f"confound-free check)")
        else:
            print(f"OK -- llm_calls identical between conditions on all {len(seed_results)} seeds.")

        print(f"=== [{label}] COST INVARIANT CHECK (live, confounded by cross-call stochasticity) ===")
        worse = [r["seed"] for r in seed_results
                 if r["v3_cost_aware_cost"] > r["v3_no_cost_sub_cost"]]
        if worse:
            print(f"cost-aware was MORE expensive on seed(s): {worse} (expected some divergence -- see offline "
                  f"replay for the confound-free check)")
        else:
            print(f"OK -- cost-aware tool cost never exceeded the no-substitution condition on any of "
                  f"{len(seed_results)} seeds.")
