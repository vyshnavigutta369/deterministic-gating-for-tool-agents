"""gpt-4o-mini, location+transportation+dining, seeds 1-5. See experiments/llm_sweep.py."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep

if __name__ == "__main__":
    run_llm_sweep(task="travel", categories=["location", "transportation", "dining"],
                  label="location+transportation+dining", model="gpt-4o-mini",
                  input_price_per_million=0.15, output_price_per_million=0.60)
