"""
gpt-4o-mini, location+transportation, seeds 1-5 -- same safety-gate
ablation as gpt4o_5seed_location_transportation_safety_baselines.py, but on
a weaker model already shown elsewhere in this project (the original
gpt4o_mini_5seed_location_transportation.py run) to make real ordering-
adjacent mistakes (that run's docstring documents a hallucinated/malformed
tool name repeated for 27 rounds with no recovery). If the safety gate has
practical value anywhere, a weaker model on the same pairing is a more
likely place for it to actually show up than GPT-4o did.
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
                              label="location+transportation (safety baselines, gpt-4o-mini)",
                              conditions=CONDITIONS, reference="gated",
                              model="gpt-4o-mini", seeds=range(1, 6),
                              input_price_per_million=0.15, output_price_per_million=0.60)
