"""
gpt-4o-mini, all three pairings, seeds 1-5 -- gate v1 (validate_batch) vs
gate v2 (validate_batch_v2, chain-aware batch resolution), same setup as
gpt4o_5seed_gate_v1_vs_v2_all_pairings.py, swapping model + pricing.
gpt-4o-mini is a more plausible candidate for actually proposing a genuine
same-round dependency chain than GPT-4o's more conservative behavior --
this project has already documented real ordering-adjacent mistakes from
this model elsewhere (the hallucinated-tool-name stuck-loop in the
safety-gate ablation).
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
                                  label=f"{label} (gate v1 vs v2, gpt-4o-mini)",
                                  conditions=CONDITIONS, reference="gated_v1",
                                  model="gpt-4o-mini", seeds=range(1, 6),
                                  input_price_per_million=0.15, output_price_per_million=0.60)
