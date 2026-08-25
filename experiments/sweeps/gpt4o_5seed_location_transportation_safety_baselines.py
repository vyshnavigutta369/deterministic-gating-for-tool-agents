"""
gpt-4o, location+transportation, seeds 1-5 (matching the original small-scale
scope, not the n=20 scale-up) -- compares the real safety-gated batching
condition against two baselines that skip validate_batch as an execution
gate, mimicking prior work's methodology:

- "blind": mimics W&D's method -- identical free-choice batching prompt to
  "gated", but whatever the model proposes executes unchecked (safety_mode=
  "blind" in run_episode/apply_batch_blind).
- "self_judged": mimics MCP-Bench's approach -- the model is prompted to
  self-assess batch safety itself (build_prompt's self_judged=True), still
  with no external validation gate.

Both baselines still run validate_batch every round as a pure DIAGNOSTIC
(never gating execution) to measure how often they actually executed
something the real safety gate would have rejected -- this is the key
number this script exists to produce: the real would-have-been-rejected
rate for each baseline, not just calls/cost/goal-completion.

Uses run_llm_sweep_conditions (experiments/llm_sweep.py), the generalized
N-condition version of run_llm_sweep, with "gated" as the reference
condition for paired significance.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.llm_sweep import run_llm_sweep_conditions

CONDITIONS = {
    "gated": dict(allow_batch=True, safety_mode="gated"),
    "blind": dict(allow_batch=True, safety_mode="blind"),
    "self_judged": dict(allow_batch=True, safety_mode="self_judged"),
}

if __name__ == "__main__":
    run_llm_sweep_conditions(task="location", categories=["location", "transportation"],
                              label="location+transportation (safety baselines)",
                              conditions=CONDITIONS, reference="gated",
                              model="gpt-4o", seeds=range(1, 6))
