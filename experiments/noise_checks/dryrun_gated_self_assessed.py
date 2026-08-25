"""Minimal live smoke test (seed 1 only) for the new gated_self_assessed
condition on gpt-4o-mini/location+transportation, alongside the other three,
before spending real money on the full n=20/4-condition run."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep_conditions

CONDITIONS = {
    "gated": dict(allow_batch=True, safety_mode="gated"),
    "blind": dict(allow_batch=True, safety_mode="blind"),
    "self_judged": dict(allow_batch=True, safety_mode="self_judged"),
    "gated_self_assessed": dict(allow_batch=True, safety_mode="gated_self_assessed"),
}

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)
    run_llm_sweep_conditions(task="location", categories=["location", "transportation"],
                              label="dry run 4-condition", conditions=CONDITIONS, reference="gated",
                              model="gpt-4o-mini", seeds=range(1, 2),
                              input_price_per_million=0.15, output_price_per_million=0.60)
