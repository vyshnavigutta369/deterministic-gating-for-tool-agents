"""Real replay of the exact known-affected case under gate v3: gpt-4o-mini,
location+transportation, seed=1, safety_mode="gated", gate_version="v3".
Same case last turn's full_rejection_feedback fix (v1) was verified against
-- checks whether v3 also recovers, and whether it does so via the earlier
partial-rejection feedback (round 3) rather than waiting for the full
rejection (round 4) the way the v1 fix had to."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from run_brts_minimal import run_episode

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)
    r = run_episode(task="location", categories=["location", "transportation"], seed=1,
                     allow_batch=True, model="gpt-4o-mini", safety_mode="gated", gate_version="v3")
    print(f"\nresult: rounds={r['rounds']} llm_calls={r['llm_calls']} goal_reached={r['goal_reached']}")
