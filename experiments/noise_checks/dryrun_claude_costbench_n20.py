"""Minimal live smoke test before the full n=20 x 3-pairing CostBench sweep:
just seed 1, location+transportation, to confirm the new entry scripts and
the new _print_significance code path in llm_sweep.py both work end-to-end
(the significance math itself was already unit-tested against synthetic
data at zero cost; this only needs to confirm real seed_results flow through
it without error)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)
    run_llm_sweep(task="location", categories=["location", "transportation"],
                  label="location+transportation (dry run)", model="claude-haiku-4-5-20251001",
                  seeds=range(1, 3), input_price_per_million=1.00, output_price_per_million=5.00)
