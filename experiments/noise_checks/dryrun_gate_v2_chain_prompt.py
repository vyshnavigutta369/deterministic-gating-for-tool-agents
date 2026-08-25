"""Minimal live smoke test (seed 1 only) for gate v1 vs v2 under the NEW
chain-aware prompt (chain_aware_prompt=True), gpt-4o, location+transportation
+dining -- confirms the new prompt actually gets the model to propose
same-round dependencies in practice, and that v1 rejects while v2 chains
them, before spending real money on the full n=5/3-pairing comparison."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep_conditions

CONDITIONS = {
    "gated_v1_chain_prompt": dict(allow_batch=True, safety_mode="gated", gate_version="v1", chain_aware_prompt=True),
    "gated_v2_chain_prompt": dict(allow_batch=True, safety_mode="gated", gate_version="v2", chain_aware_prompt=True),
}

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)
    run_llm_sweep_conditions(task="travel", categories=["location", "transportation", "dining"],
                              label="dry run gate v1 vs v2, chain-aware prompt", conditions=CONDITIONS,
                              reference="gated_v1_chain_prompt", model="gpt-4o", seeds=range(1, 2))
