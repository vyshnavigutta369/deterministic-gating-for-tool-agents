"""
Scale-up of the GPT-4o factorial ablation from n=50 to n=100 on
location+transportation: IDENTICAL to
gpt4o_50seed_location_transportation_safety_baselines_4cond.py in every
respect except the seed range. Directly tests whether the self_judged
unsafe-execution signal found at n=50 (2/562 executed calls, 0.36%, both
at round 5 proposing the same tool pair) strengthens with more exposure
or stays a rare, low-count event, while blind/gated/gated_self_assessed
are expected to replicate their clean n=50 nulls.
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
                              label="location+transportation (4-condition safety baselines, GPT-4o, n=100)",
                              conditions=CONDITIONS, reference="gated",
                              model="gpt-4o", seeds=range(1, 101))
