"""Minimal live smoke test for the Claude-ported BFCL multi_turn harness:
just the first qualifying task, batched only, to confirm call_llm_stub_claude
+ validate_batch_bfcl + multi_turn_checker work end-to-end before spending
real money on the full 5-task sweep."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from bfcl.classify_tasks import find_independent_multi_call_turns
from experiments.bfcl_episode import run_bfcl_episode

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)
    q = find_independent_multi_call_turns(max_tasks=1, verbose=False)[0]
    task, ground_truth = q["task"], q["ground_truth"]
    r = run_bfcl_episode(task, ground_truth, allow_batch=True, model="claude-haiku-4-5-20251001",
                          condition_label="dryrun_batched")
    print(f"\nresult: task_id={task['id']} llm_calls={r['llm_calls']} goal_reached={r['goal_reached']}")
