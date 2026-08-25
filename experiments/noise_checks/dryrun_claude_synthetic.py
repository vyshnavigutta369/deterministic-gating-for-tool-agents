"""Minimal live smoke test for the Claude-ported synthetic harness: one
seed, batched only, max_rounds=5, to confirm call_llm_stub_claude + the
empty-response feedback path work end-to-end before spending real money
on the full claude_haiku_synthetic_5seed.py sweep."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.synthetic_episode import run_synthetic_episode

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)
    r = run_synthetic_episode(["attraction", "dining", "shopping"], seed=1, allow_batch=True,
                               max_rounds=5, model="claude-haiku-4-5-20251001")
    print(f"\nresult: rounds={r['rounds']} llm_calls={r['llm_calls']} goal_reached={r['goal_reached']}")
