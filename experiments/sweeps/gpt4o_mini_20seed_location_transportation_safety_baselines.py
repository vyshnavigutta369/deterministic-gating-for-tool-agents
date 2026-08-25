"""
gpt-4o-mini, location+transportation, seeds 1-20 -- same safety-gate
ablation as gpt4o_mini_5seed_location_transportation_safety_baselines.py,
scaled from n=5 to n=20 to test whether the single traced unsafe-execution
case (blind, seed=3, round=3: Location_Refinement_Step2 executed with an
unmet precondition) represents a real, stable rate or was closer to a
one-off, and to give the paired significance tests real power.
"""
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
    run_llm_sweep_conditions(task="location", categories=["location", "transportation"],
                              label="location+transportation (safety baselines, gpt-4o-mini, n=20)",
                              conditions=CONDITIONS, reference="gated",
                              model="gpt-4o-mini", seeds=range(1, 21),
                              input_price_per_million=0.15, output_price_per_million=0.60)
