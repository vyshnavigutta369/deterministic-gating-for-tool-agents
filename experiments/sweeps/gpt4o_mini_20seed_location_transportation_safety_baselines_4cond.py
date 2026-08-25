"""
gpt-4o-mini, location+transportation, seeds 1-20, FOUR safety-gate
conditions: gated (reference), blind, self_judged, and gated_self_assessed
(the self-judged prompt PLUS the real validate_batch execution gate --
tests whether self-assessment improves the gated condition's efficiency,
not just whether removing the gate is risky). Matches the n=20 scale
already established for gated/blind/self_judged so this has real
statistical power from the first run.
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
                              label="location+transportation (4-condition safety baselines, gpt-4o-mini, n=20)",
                              conditions=CONDITIONS, reference="gated",
                              model="gpt-4o-mini", seeds=range(1, 21),
                              input_price_per_million=0.15, output_price_per_million=0.60)
