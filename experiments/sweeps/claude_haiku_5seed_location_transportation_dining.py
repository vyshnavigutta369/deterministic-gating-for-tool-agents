"""claude-haiku-4-5, location+transportation+dining, seeds 1-5. See experiments/llm_sweep.py.

Pricing verified 2026-07-27: claude-haiku-4-5 is $1.00/1M input, $5.00/1M output.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)
    run_llm_sweep(task="travel", categories=["location", "transportation", "dining"],
                  label="location+transportation+dining", model="claude-haiku-4-5-20251001",
                  input_price_per_million=1.00, output_price_per_million=5.00)
