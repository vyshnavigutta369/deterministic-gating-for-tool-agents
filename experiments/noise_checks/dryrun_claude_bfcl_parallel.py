"""Minimal live smoke test for the Claude-ported BFCL parallel harness: just
task parallel_0, batched only, to confirm call_llm_stub_claude + ast_checker
scoring work end-to-end before spending real money on the full 20-task sweep."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.bfcl_parallel_episode import run_parallel_episode
from experiments.sweeps.claude_haiku_bfcl_parallel_20tasks import load_parallel_tasks

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)
    tasks, ground_truths = load_parallel_tasks(1)
    task = tasks[0]
    gt = ground_truths[task["id"]]
    r = run_parallel_episode(task, gt, allow_batch=True, model="claude-haiku-4-5-20251001")
    print(f"\nresult: task_id={task['id']} llm_calls={r['llm_calls']} goal_reached={r['goal_reached']}")
    print(f"accumulated_calls={r['accumulated_calls']}")
