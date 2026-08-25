"""claude-haiku-4-5, location+transportation+dining, seeds 1-20. See experiments/llm_sweep.py.
Mirrors gpt4o_20seed_location_transportation_dining.py exactly (same seed range),
swapping model + pricing."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep

if __name__ == "__main__":
    run_llm_sweep(task="travel", categories=["location", "transportation", "dining"],
                  label="location+transportation+dining", model="claude-haiku-4-5-20251001", seeds=range(1, 21),
                  input_price_per_million=1.00, output_price_per_million=5.00)
