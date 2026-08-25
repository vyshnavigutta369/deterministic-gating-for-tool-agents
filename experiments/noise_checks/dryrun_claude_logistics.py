"""Minimal live smoke test for the Claude-ported PDDL Logistics harness:
task_independent only, batched, max_rounds=8 (well under the real 15/30
budgets), to confirm call_llm_stub_claude + the empty-response feedback
(alongside the existing cycle-detection feedback) work end-to-end before
spending real money on the full task_independent / task_independent_mixed
sweep."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from logistics_llm_harness import run_logistics_episode

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logistics")
    r = run_logistics_episode(
        domain_file=os.path.join(base, "logistics_domain.pddl"),
        problem_file=os.path.join(base, "task_independent.pddl"),
        allow_batch=True, model="claude-haiku-4-5-20251001", max_rounds=8,
    )
    print(f"\nresult: llm_calls={r['llm_calls']} goal_reached={r['goal_reached']}")
