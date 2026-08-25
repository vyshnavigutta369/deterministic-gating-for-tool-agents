"""
Real n=5 verification of apply_cost_substitution: gpt-4o-mini,
location+transportation (the exact pairing that showed the cost premium
in Phase 1 of the gate comparison), gate_version="v3" (the current best
gate) held fixed in both conditions -- the only thing that differs is
cost_substitution True/False. Per the design constraint, round count must
come out IDENTICAL between conditions on every seed (apply_batch's
new_types computation is unaffected by substitution); cost must come out
lower or equal, never higher.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep_conditions

CONDITIONS = {
    "v3_no_cost_sub": dict(allow_batch=True, safety_mode="gated", gate_version="v3", cost_substitution=False),
    "v3_cost_aware": dict(allow_batch=True, safety_mode="gated", gate_version="v3", cost_substitution=True),
}

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    seed_results = run_llm_sweep_conditions(
        task="location", categories=["location", "transportation"],
        label="location+transportation (v3 with vs without cost-substitution, n=5, gpt-4o-mini)",
        conditions=CONDITIONS, reference="v3_no_cost_sub",
        model="gpt-4o-mini", seeds=range(1, 6),
        input_price_per_million=0.15, output_price_per_million=0.60)

    print("\n=== ROUND-COUNT INVARIANT CHECK (must be identical every seed, by construction) ===")
    mismatches = [r["seed"] for r in seed_results
                  if r["v3_no_cost_sub_calls"] != r["v3_cost_aware_calls"]]
    if mismatches:
        print(f"VIOLATION -- round/call count differed on seed(s): {mismatches}. "
              f"This should be structurally impossible; investigate immediately.")
    else:
        print("OK -- llm_calls identical between conditions on every seed, as guaranteed by construction.")

    print("\n=== COST INVARIANT CHECK (cost-aware must never be MORE expensive) ===")
    worse = [r["seed"] for r in seed_results
             if r["v3_cost_aware_cost"] > r["v3_no_cost_sub_cost"]]
    if worse:
        print(f"VIOLATION -- cost-aware was MORE expensive on seed(s): {worse}. Investigate immediately.")
    else:
        print("OK -- cost-aware tool cost never exceeded the no-substitution condition on any seed.")
