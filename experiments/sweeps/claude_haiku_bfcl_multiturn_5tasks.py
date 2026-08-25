"""
claude-haiku-4-5, BFCL multi_turn category, 5 real tasks drawn from
bfcl.classify_tasks.find_independent_multi_call_turns -- same selection
method as gpt4o_bfcl_multiturn_5tasks.py (deterministic given the same BFCL
dataset, so this should select the identical 5 tasks). run_bfcl_episode is
now Claude-aware (temperature=0 + robust JSON extraction via
call_llm_stub_claude); no empty-response feedback change, matching
bfcl_parallel's treatment (empty = intentional turn-completion here, not a
stall). No cost tracking here, matching the original GPT-4o script's format
exactly (it doesn't track token usage for this domain either).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from bfcl.classify_tasks import find_independent_multi_call_turns
from experiments.bfcl_episode import run_bfcl_episode

MODEL = "claude-haiku-4-5-20251001"

if __name__ == "__main__":
    print(f"call_llm_stub now makes REAL Anthropic API calls (model={MODEL}) and will incur cost.")
    print("Requires ANTHROPIC_API_KEY set in your environment.\n")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    print("Selecting 5 real BFCL multi_turn_base tasks with a genuinely-independent multi-call turn...")
    qualifying = find_independent_multi_call_turns(max_tasks=5, verbose=False)
    print(f"Selected: {[q['task']['id'] for q in qualifying]}\n")

    results = []
    for q in qualifying:
        task, ground_truth = q["task"], q["ground_truth"]
        print(f"=== {task['id']} : BATCHED ===")
        batched = run_bfcl_episode(task, ground_truth, allow_batch=True, model=MODEL,
                                    condition_label=f"{task['id']}_batched")
        print(f"  -> llm_calls={batched['llm_calls']} goal_reached={batched['goal_reached']}\n")

        print(f"=== {task['id']} : SEQUENTIAL ===")
        sequential = run_bfcl_episode(task, ground_truth, allow_batch=False, model=MODEL,
                                       condition_label=f"{task['id']}_sequential")
        print(f"  -> llm_calls={sequential['llm_calls']} goal_reached={sequential['goal_reached']}\n")

        results.append({
            "task_id": task["id"],
            "batched_calls": batched["llm_calls"], "batched_ok": batched["goal_reached"],
            "sequential_calls": sequential["llm_calls"], "sequential_ok": sequential["goal_reached"],
        })

    print(f"=== SUMMARY (5 BFCL multi_turn tasks, model={MODEL}): LLM calls / goal-completion ===")
    print(f"{'task_id':>20} {'batch_calls':>11} {'batch_ok':>9} {'seq_calls':>9} {'seq_ok':>7}")
    for r in results:
        print(f"{r['task_id']:>20} {r['batched_calls']:>11} {str(r['batched_ok']):>9} "
              f"{r['sequential_calls']:>9} {str(r['sequential_ok']):>7}")

    n = len(results)
    avg_batch_calls = sum(r["batched_calls"] for r in results) / n
    avg_seq_calls = sum(r["sequential_calls"] for r in results) / n
    batch_completion_rate = sum(r["batched_ok"] for r in results) / n
    seq_completion_rate = sum(r["sequential_ok"] for r in results) / n
    print(f"{'avg/rate':>20} {avg_batch_calls:>11.2f} {batch_completion_rate:>9.0%} "
          f"{avg_seq_calls:>9.2f} {seq_completion_rate:>7.0%}")
