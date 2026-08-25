"""
Sanity-check dry run for the n=50 scale-up: IDENTICAL to
gpt4o_mini_20seed_location_transportation_safety_baselines_4cond.py in
every respect except the seed range (1 seed instead of 20/50) -- confirms
the existing harness still runs correctly end-to-end with zero code
modification needed, before spending anything at real scale.
"""
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
    run_llm_sweep_conditions(task="location", categories=["location", "transportation"],
                              label="location+transportation (4-condition safety baselines, gpt-4o-mini, n=1 DRY RUN)",
                              conditions=CONDITIONS, reference="gated",
                              model="gpt-4o-mini", seeds=range(1, 2),
                              input_price_per_million=0.15, output_price_per_million=0.60)
