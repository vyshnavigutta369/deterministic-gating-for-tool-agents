"""
gpt-4o, all three pairings, seeds 1-5 -- gate v1 vs v2 under the NEW
chain-aware prompt (chain_aware_prompt=True), which explicitly PERMITS
proposing a tool that depends on another proposed tool's same-round output,
telling the model the system resolves execution order automatically. This
removes the "never a tool that needs another proposed tool's output"
restriction that (per gpt4o_5seed_gate_v1_vs_v2_all_pairings.py's real
result: zero divergence across 60 live episodes on two models) was
apparently why v1 and v2 never disagreed under the default prompt -- the
model simply never proposed a same-round chain in the first place. This is
the real test: v1 should now show real rejections it can't resolve, v2
should show real extra-accepted chains it can.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep_conditions

CONDITIONS = {
    "gated_v1_chain_prompt": dict(allow_batch=True, safety_mode="gated", gate_version="v1", chain_aware_prompt=True),
    "gated_v2_chain_prompt": dict(allow_batch=True, safety_mode="gated", gate_version="v2", chain_aware_prompt=True),
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
                                  label=f"{label} (gate v1 vs v2, chain-aware prompt)",
                                  conditions=CONDITIONS, reference="gated_v1_chain_prompt",
                                  model="gpt-4o", seeds=range(1, 6))
