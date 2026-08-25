"""Minimal live smoke test (seed 1 only) for gate v2 (chain-aware batch
resolution): gated_v1 vs gated_v2 on gpt-4o, location+transportation+dining
(the 3-category pairing, most likely to surface real same-round dependency
chains), before spending real money on the full n=5/3-pairing comparison."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep_conditions

CONDITIONS = {
    "gated_v1": dict(allow_batch=True, safety_mode="gated", gate_version="v1"),
    "gated_v2": dict(allow_batch=True, safety_mode="gated", gate_version="v2"),
}

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)
    run_llm_sweep_conditions(task="travel", categories=["location", "transportation", "dining"],
                              label="dry run gate v2, 3-category", conditions=CONDITIONS,
                              reference="gated_v1", model="gpt-4o", seeds=range(1, 2))
