"""
Dry run for Claude Haiku's factorial ablation: IDENTICAL to
gpt4o_50seed_location_transportation_safety_baselines_4cond.py in every
respect except the model (claude-haiku-4-5-20251001, using the same
pricing already established for this model elsewhere in this project)
and the seed range (1 instead of 50). Confirms the harness handles
Claude Haiku correctly under all four conditions (gated/blind/
self_judged/gated_self_assessed) before spending anything at real scale.
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
                              label="location+transportation (4-condition safety baselines, claude-haiku, n=1 DRY RUN)",
                              conditions=CONDITIONS, reference="gated",
                              model="claude-haiku-4-5-20251001", seeds=range(1, 2),
                              input_price_per_million=1.00, output_price_per_million=5.00)
