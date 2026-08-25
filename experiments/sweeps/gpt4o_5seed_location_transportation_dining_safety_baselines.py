"""
gpt-4o, location+transportation+dining (3-category pairing), seeds 1-5 --
same safety-gate ablation as gpt4o_5seed_location_transportation_safety_baselines.py
(gated / blind / self_judged via run_llm_sweep_conditions), but on the
harder 3-category pairing: more candidate tools visible per round than the
2-category case, so a genuine ordering mistake is more likely to actually
occur, which is exactly what the would-have-been-rejected diagnostic is
meant to detect if it happens.
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
    run_llm_sweep_conditions(task="travel", categories=["location", "transportation", "dining"],
                              label="location+transportation+dining (safety baselines)",
                              conditions=CONDITIONS, reference="gated",
                              model="gpt-4o", seeds=range(1, 6))
