"""
Phase 2 of the scaled-up gate comparison: GPT-4o, n=20, all three CostBench
pairings, same three conditions as Phase 1 (v1_baseline / full_rejection_fix
/ v3_unified). Consistency check -- GPT-4o rarely hits the hallucination
failure mode v3 targets (per the safety-gate ablation and gate v1-vs-v2
experiments earlier in this project), so expect little difference, but
worth confirming rather than assuming given Phase 1's real, significant
gpt-4o-mini effect.
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
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    for task, categories, label in PAIRINGS:
        print(f"\n########## {label} ##########")
        run_llm_sweep_conditions(task=task, categories=categories,
                                  label=f"{label} (v1 vs full-rejection-fix vs v3-unified, n=20, gpt-4o)",
                                  conditions=CONDITIONS, reference="v1_baseline",
                                  model="gpt-4o", seeds=range(1, 21))
