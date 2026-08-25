"""
Cross-pairing replication of the central factorial ablation on
location+accommodation, n=50: IDENTICAL to
gpt4o_mini_50seed_location_transportation_safety_baselines_4cond.py in
every respect except the pairing (categories=["location","accommodation"]
instead of ["location","transportation"]). Same four conditions (gated
reference, blind, self_judged, gated_self_assessed), same harness, same
real validate_batch gate run as a silent diagnostic under every condition.
Tests whether the location+transportation unsafe-execution finding (0
unsafe executions across 1431 real executed calls in blind/self_judged)
is a general property of this model/harness or specific to that pairing.
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
    run_llm_sweep_conditions(task="location", categories=["location", "accommodation"],
                              label="location+accommodation (4-condition safety baselines, gpt-4o-mini, n=50)",
                              conditions=CONDITIONS, reference="gated",
                              model="gpt-4o-mini", seeds=range(1, 51),
                              input_price_per_million=0.15, output_price_per_million=0.60)
