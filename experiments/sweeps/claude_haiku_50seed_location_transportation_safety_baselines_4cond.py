"""
Cross-model replication of the central factorial ablation on Claude
Haiku, location+transportation, n=50: IDENTICAL to
gpt4o_50seed_location_transportation_safety_baselines_4cond.py in every
respect except the model (claude-haiku-4-5-20251001, pricing already
established for this model elsewhere in this project). Same four
conditions (gated reference, blind, self_judged, gated_self_assessed),
same harness, same real validate_batch gate run as a silent diagnostic
under every condition. Third model in the unsafe-execution comparison
alongside gpt-4o-mini (clean null both pairings) and GPT-4o (rare,
~0.1-0.4% real signal, present in both blind and self_judged at n=100).
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
                              label="location+transportation (4-condition safety baselines, claude-haiku, n=50)",
                              conditions=CONDITIONS, reference="gated",
                              model="claude-haiku-4-5-20251001", seeds=range(1, 51),
                              input_price_per_million=1.00, output_price_per_million=5.00)
