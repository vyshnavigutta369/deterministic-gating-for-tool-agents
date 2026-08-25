"""
Stage 3 of the redundancy-elimination verification: real n=5 test,
gpt-4o-mini, location+transportation (same flagship pairing as cost-
substitution's and state-caching's own n=5 checks), gate_version="v3"
fixed in both conditions -- the only difference is redundancy_elimination
True/False. Per the design constraint, round count must come out
IDENTICAL between conditions on every seed (apply_batch's new_types
computation is unaffected by dropping same-output duplicates); cost must
come out lower or equal, never higher.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep_conditions

CONDITIONS = {
    "v3_no_redundancy_elim": dict(allow_batch=True, safety_mode="gated", gate_version="v3"),
    "v3_redundancy_elim": dict(allow_batch=True, safety_mode="gated", gate_version="v3", redundancy_elimination=True),
}

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    seed_results = run_llm_sweep_conditions(
        task="location", categories=["location", "transportation"],
        label="location+transportation (v3 with vs without redundancy elimination, n=5, gpt-4o-mini)",
        conditions=CONDITIONS, reference="v3_no_redundancy_elim",
        model="gpt-4o-mini", seeds=range(1, 6),
        input_price_per_million=0.15, output_price_per_million=0.60)

    print("\n=== ROUND-COUNT INVARIANT CHECK (must be identical every seed, by construction) ===")
    mismatches = [r["seed"] for r in seed_results
                  if r["v3_no_redundancy_elim_calls"] != r["v3_redundancy_elim_calls"]]
    if mismatches:
        print(f"VIOLATION -- round/call count differed on seed(s): {mismatches}. "
              f"This should be structurally impossible; investigate immediately.")
    else:
        print("OK -- llm_calls identical between conditions on every seed, as guaranteed by construction.")

    print("\n=== COST INVARIANT CHECK (redundancy elimination must never be MORE expensive) ===")
    worse = [r["seed"] for r in seed_results
             if r["v3_redundancy_elim_cost"] > r["v3_no_redundancy_elim_cost"]]
    if worse:
        print(f"VIOLATION -- redundancy elimination was MORE expensive on seed(s): {worse}. Investigate immediately.")
    else:
        print("OK -- redundancy-elimination tool cost never exceeded the baseline on any seed.")

    print("\n=== GOAL-COMPLETION CHECK ===")
    failed = [r["seed"] for r in seed_results if r["v3_no_redundancy_elim_ok"] and not r["v3_redundancy_elim_ok"]]
    if failed:
        print(f"VIOLATION -- redundancy elimination failed where baseline succeeded on seed(s): {failed}")
    else:
        print("OK -- redundancy elimination reached the goal on every seed the baseline did.")
