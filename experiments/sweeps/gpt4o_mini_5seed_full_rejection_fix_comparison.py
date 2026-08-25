"""
gpt-4o-mini, location+transportation, seeds 1-5 -- WITH vs WITHOUT the new
full_rejection_feedback branch (run_brts_minimal.py), otherwise identical
(safety_mode="gated", allow_batch=True). Uses run_llm_sweep_conditions
unchanged. The metric that matters most here is goal-completion rate: the
real log analysis found 54% of affected episodes never recovered under the
old (without) behavior.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep_conditions

CONDITIONS = {
    "with_fix": dict(allow_batch=True, safety_mode="gated", full_rejection_feedback=True),
    "without_fix": dict(allow_batch=True, safety_mode="gated", full_rejection_feedback=False),
}

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)
    run_llm_sweep_conditions(task="location", categories=["location", "transportation"],
                              label="location+transportation (full-rejection-feedback fix comparison)",
                              conditions=CONDITIONS, reference="without_fix",
                              model="gpt-4o-mini", seeds=range(1, 6),
                              input_price_per_million=0.15, output_price_per_million=0.60)
