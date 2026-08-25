"""
Diagnostic: is call_llm_stub actually deterministic at temperature=0 across two
back-to-back runs of the SAME episode (same seed, same allow_batch=False), or
does the tool-cost variance between sweeps come from the model itself?
The tool registry was already confirmed deterministic separately -- this
isolates the other half of the hypothesis.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from run_brts_minimal import run_episode

SEEDS_TO_CHECK = [4, 5, 7, 10]

for seed in SEEDS_TO_CHECK:
    print(f"=== seed={seed}: run A ===")
    result_a = run_episode(task="location", categories=["location", "transportation"],
                            seed=seed, allow_batch=False)
    print(f"=== seed={seed}: run B ===")
    result_b = run_episode(task="location", categories=["location", "transportation"],
                            seed=seed, allow_batch=False)

    seq_a = [r["valid"] for r in result_a["round_log"]]
    seq_b = [r["valid"] for r in result_b["round_log"]]
    proposed_a = [r["proposed"] for r in result_a["round_log"]]
    proposed_b = [r["proposed"] for r in result_b["round_log"]]

    identical_valid = seq_a == seq_b
    identical_proposed = proposed_a == proposed_b
    print(f"\nseed={seed}: cost_A={result_a['total_tool_cost']:.2f} cost_B={result_b['total_tool_cost']:.2f} "
          f"identical_proposed_sequence={identical_proposed} identical_valid_sequence={identical_valid}")
    if not identical_proposed:
        print(f"  proposed_A: {proposed_a}")
        print(f"  proposed_B: {proposed_b}")
    print()
