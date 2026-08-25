"""
Phase 1 of the scaled-up cost-substitution verification: gpt-4o-mini,
n=20, all three CostBench pairings, gate_version="v3" held FIXED in both
conditions -- the only thing that varies is cost_substitution True/False.
Matches the rigor level of the earlier gate v3 comparison (same model,
same n, same three pairings, same paired-significance apparatus), staged
the same way: report real results before deciding whether Phase 2
(GPT-4o, Claude Haiku) is warranted.

Beyond what run_llm_sweep_conditions already reports, this script adds an
explicit per-pairing, per-seed invariant check -- the actual safety claim
behind apply_cost_substitution is that it can NEVER change round count and
can NEVER raise cost, guaranteed by construction (apply_batch only cares
about output_type, which substitution never changes). n=5 confirmed this
held; this reruns the same checks at n=20 across all three pairings, since
"guaranteed by construction" should still be verified empirically, not
just asserted.
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
            label=f"{label} (v3 with vs without cost-substitution, n=20, gpt-4o-mini)",
            conditions=CONDITIONS, reference="v3_no_cost_sub",
            model="gpt-4o-mini", seeds=range(1, 21),
            input_price_per_million=0.15, output_price_per_million=0.60)

        print(f"\n=== [{label}] ROUND-COUNT INVARIANT CHECK (must be identical every seed) ===")
        mismatches = [r["seed"] for r in seed_results
                      if r["v3_no_cost_sub_calls"] != r["v3_cost_aware_calls"]]
        if mismatches:
            print(f"VIOLATION -- round/call count differed on seed(s): {mismatches}. "
                  f"This should be structurally impossible; investigate immediately.")
        else:
            print(f"OK -- llm_calls identical between conditions on all {len(seed_results)} seeds.")

        print(f"=== [{label}] COST INVARIANT CHECK (cost-aware must never be MORE expensive) ===")
        worse = [r["seed"] for r in seed_results
                 if r["v3_cost_aware_cost"] > r["v3_no_cost_sub_cost"]]
        if worse:
            print(f"VIOLATION -- cost-aware was MORE expensive on seed(s): {worse}. Investigate immediately.")
        else:
            print(f"OK -- cost-aware tool cost never exceeded the no-substitution condition on any of "
                  f"{len(seed_results)} seeds.")
