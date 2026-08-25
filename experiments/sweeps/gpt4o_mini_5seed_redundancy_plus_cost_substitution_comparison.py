"""
Stage 4 real test: does redundancy_elimination genuinely STACK with
cost_substitution (dedup first, then substitute -- the exact order
run_episode already wires) rather than interfere with it? Three
conditions: baseline (neither), cost_substitution alone (already verified
independently), and both composed. gate_version="v3" fixed throughout.

Per the cost-substitution and redundancy-elimination investigations
already done, the LIVE round-count/cost "invariant" checks below are
confounded by cross-call LLM stochasticity between separate live episodes
and will show some divergence regardless of code correctness -- that's
expected. The real safety evidence is an offline replay of the real
recorded proposals from this run (zero new API cost), same discipline as
both prior mechanisms.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep_conditions

CONDITIONS = {
    "v3_baseline": dict(allow_batch=True, safety_mode="gated", gate_version="v3"),
    "v3_cost_sub_only": dict(allow_batch=True, safety_mode="gated", gate_version="v3", cost_substitution=True),
    "v3_composed": dict(allow_batch=True, safety_mode="gated", gate_version="v3",
                         redundancy_elimination=True, cost_substitution=True),
}

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    seed_results = run_llm_sweep_conditions(
        task="location", categories=["location", "transportation"],
        label="location+transportation (baseline vs cost-sub-only vs composed dedup+cost-sub, n=5, gpt-4o-mini)",
        conditions=CONDITIONS, reference="v3_baseline",
        model="gpt-4o-mini", seeds=range(1, 6),
        input_price_per_million=0.15, output_price_per_million=0.60)

    print("\n=== COMPOSED-NEVER-WORSE-THAN-COST-SUB-ALONE CHECK ===")
    worse = [r["seed"] for r in seed_results if r["v3_composed_cost"] > r["v3_cost_sub_only_cost"]]
    if worse:
        print(f"seed(s) where composed cost exceeded cost-sub-alone: {worse} (live-comparison confound expected -- "
              f"see offline replay for the confound-free check)")
    else:
        print("OK -- composed never exceeded cost-sub-alone on any seed live.")
