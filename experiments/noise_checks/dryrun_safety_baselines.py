"""Minimal live smoke test for the new blind/self_judged safety-baseline
conditions: just seed 1, location+transportation, all 3 conditions, to
confirm the self_judged prompt renders correctly, apply_batch_blind executes
real proposals without crashing, and the would-have-been-rejected diagnostic
fires on real (not simulated) unsafe proposals -- before spending real money
on the full 5-seed sweep."""
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
    run_llm_sweep_conditions(task="location", categories=["location", "transportation"],
                              label="location+transportation (safety baselines dry run)",
                              conditions=CONDITIONS, reference="gated",
                              model="gpt-4o", seeds=range(1, 2))
