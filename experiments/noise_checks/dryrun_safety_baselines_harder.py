"""Minimal live smoke test (seed 1 only) for the two new harder safety-gate
ablation configs -- gpt-4o 3-category and gpt-4o-mini 2-category -- before
spending real money on the full 5-seed runs. gpt-4o-mini is the higher-risk
combination here: it's already shown real hallucinated-tool-name/ordering
mistakes elsewhere in this project, so this checks the blind/self_judged
paths don't crash on that kind of malformed output."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep_conditions

CONDITIONS = {
    "gated": dict(allow_batch=True, safety_mode="gated"),
    "blind": dict(allow_batch=True, safety_mode="blind"),
    "self_judged": dict(allow_batch=True, safety_mode="self_judged"),
}

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    print("### gpt-4o, location+transportation+dining, seed=1 ###")
    run_llm_sweep_conditions(task="travel", categories=["location", "transportation", "dining"],
                              label="dry run 3-category", conditions=CONDITIONS, reference="gated",
                              model="gpt-4o", seeds=range(1, 2))

    print("\n### gpt-4o-mini, location+transportation, seed=1 ###")
    run_llm_sweep_conditions(task="location", categories=["location", "transportation"],
                              label="dry run gpt-4o-mini", conditions=CONDITIONS, reference="gated",
                              model="gpt-4o-mini", seeds=range(1, 2),
                              input_price_per_million=0.15, output_price_per_million=0.60)
