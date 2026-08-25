"""
gpt-4o, all three pairings, seeds 1-5 -- gate v1 (validate_batch) vs gate v2
(validate_batch_v2, chain-aware batch resolution) under otherwise-identical
free-choice batching conditions (safety_mode="gated" for both -- the only
difference is gate_version). Uses run_llm_sweep_conditions unchanged, same
as the other safety-baseline ablations, so results are directly comparable
in format.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep_conditions

CONDITIONS = {
    "gated_v1": dict(allow_batch=True, safety_mode="gated", gate_version="v1"),
    "gated_v2": dict(allow_batch=True, safety_mode="gated", gate_version="v2"),
}

PAIRINGS = [
    ("location", ["location", "transportation"], "location+transportation"),
    ("location", ["location", "accommodation"], "location+accommodation"),
    ("travel", ["location", "transportation", "dining"], "location+transportation+dining"),
]

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    for task, categories, label in PAIRINGS:
        print(f"\n########## {label} ##########")
        run_llm_sweep_conditions(task=task, categories=categories,
                                  label=f"{label} (gate v1 vs v2)",
                                  conditions=CONDITIONS, reference="gated_v1",
                                  model="gpt-4o", seeds=range(1, 6))
