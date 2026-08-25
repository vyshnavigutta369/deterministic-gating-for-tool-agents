"""
Phase 3 of the scaled-up gate comparison: claude-haiku-4-5, n=20, all three
CostBench pairings, same three conditions as Phases 1-2 (v1_baseline /
full_rejection_fix / v3_unified). Cross-provider depth check, run because
Phases 1-2 together showed a coherent, mechanistically-grounded picture: a
real effect on the model with the targeted failure mode (gpt-4o-mini), a
clean expected null on the model without it (GPT-4o). Claude Haiku has its
own separately-documented empty-response stall pattern earlier in this
project (fixed by build_empty_response_feedback) but has not been tested
against the full/partial-rejection failure mode specifically.
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

PAIRINGS = [
    ("location", ["location", "transportation"], "location+transportation"),
    ("location", ["location", "accommodation"], "location+accommodation"),
    ("travel", ["location", "transportation", "dining"], "location+transportation+dining"),
]

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    for task, categories, label in PAIRINGS:
        print(f"\n########## {label} ##########")
        run_llm_sweep_conditions(task=task, categories=categories,
                                  label=f"{label} (v1 vs full-rejection-fix vs v3-unified, n=20, claude-haiku)",
                                  conditions=CONDITIONS, reference="v1_baseline",
                                  model="claude-haiku-4-5-20251001", seeds=range(1, 21),
                                  input_price_per_million=1.00, output_price_per_million=5.00)
