"""
Sanity-check dry run for the GPT-4o n=50 factorial ablation: IDENTICAL to
gpt4o_mini_50seed_location_transportation_safety_baselines_4cond.py in
every respect except the model (gpt-4o instead of gpt-4o-mini, using the
default pricing since that's already GPT-4o's real per-token cost -- same
convention the earlier GPT-4o cost-substitution/gate-comparison scripts
this session used) and the seed range (1 instead of 50). Confirms the
harness handles GPT-4o correctly before spending anything at real scale.
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
                              label="location+transportation (4-condition safety baselines, GPT-4o, n=1 DRY RUN)",
                              conditions=CONDITIONS, reference="gated",
                              model="gpt-4o", seeds=range(1, 2))
