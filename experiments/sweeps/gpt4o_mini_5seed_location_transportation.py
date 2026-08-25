"""gpt-4o-mini, location+transportation, seeds 1-5. See experiments/llm_sweep.py.

Historical note: this pairing showed a real failure (1/5 batched episodes
never reached the goal -- gpt-4o-mini proposed a hallucinated/malformed tool
name and repeated it for 27 rounds with no recovery, since nothing in the
prompt informs the model when a proposal was rejected). run_llm_sweep's
explicit failure flagging exists specifically because of this run.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep

if __name__ == "__main__":
    run_llm_sweep(task="location", categories=["location", "transportation"],
                  label="location+transportation", model="gpt-4o-mini",
                  input_price_per_million=0.15, output_price_per_million=0.60)
