"""Real replay of the exact known-affected case from the log analysis:
gpt-4o-mini, location+transportation, seed=1, safety_mode="gated" -- this
seed reliably hit the Location_Finish_from_Step3/Transportation_Finish_from_Step3
hallucination and ran to max_rounds=30 without recovering, across at least
5 separate real prior runs. Checks whether the new full_rejection_feedback
branch (on by default) lets it recover instead."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from run_brts_minimal import run_episode

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)
    r = run_episode(task="location", categories=["location", "transportation"], seed=1,
                     allow_batch=True, model="gpt-4o-mini", safety_mode="gated")
    print(f"\nresult: rounds={r['rounds']} llm_calls={r['llm_calls']} goal_reached={r['goal_reached']}")
