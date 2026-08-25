"""
gpt-4o-mini, location+transportation, seeds 1-5 -- three conditions:
  - v1_baseline: gate_version="v1", full_rejection_feedback=False (no fix at all)
  - full_rejection_fix: gate_version="v1", full_rejection_feedback=True (last turn's fix)
  - v3_unified: gate_version="v3" (chain-aware resolution + universal rejection
    reasoning for ANY rejection, partial or full)
Tests whether combining both capabilities into validate_batch_v3 genuinely
covers more ground than the full-rejection-only fix alone.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep_conditions

CONDITIONS = {
    "v1_baseline": dict(allow_batch=True, safety_mode="gated", gate_version="v1", full_rejection_feedback=False),
    "full_rejection_fix": dict(allow_batch=True, safety_mode="gated", gate_version="v1", full_rejection_feedback=True),
    "v3_unified": dict(allow_batch=True, safety_mode="gated", gate_version="v3"),
}

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)
    run_llm_sweep_conditions(task="location", categories=["location", "transportation"],
                              label="location+transportation (v1 vs full-rejection-fix vs v3-unified)",
                              conditions=CONDITIONS, reference="v1_baseline",
                              model="gpt-4o-mini", seeds=range(1, 6),
                              input_price_per_million=0.15, output_price_per_million=0.60)
