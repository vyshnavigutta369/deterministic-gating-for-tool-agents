"""
Cross-model replication of the central factorial ablation on GPT-4o,
location+transportation, n=50: IDENTICAL to
gpt4o_mini_50seed_location_transportation_safety_baselines_4cond.py in
every respect except the model (gpt-4o instead of gpt-4o-mini, default
pricing already matches GPT-4o's real per-token cost). Same four
conditions (gated reference, blind, self_judged, gated_self_assessed),
same harness, same real validate_batch gate run as a silent diagnostic
under every condition. Tests whether the gpt-4o-mini unsafe-execution
finding (0 unsafe executions, both location+transportation and
location+accommodation pairings) is specific to gpt-4o-mini or holds for
a stronger model too.
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
                              label="location+transportation (4-condition safety baselines, GPT-4o, n=50)",
                              conditions=CONDITIONS, reference="gated",
                              model="gpt-4o", seeds=range(1, 51))
