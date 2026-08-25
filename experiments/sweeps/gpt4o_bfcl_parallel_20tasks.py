"""
gpt-4o, BFCL parallel category, 20 real tasks (parallel_0 .. parallel_19 --
no special selection needed, since every task in this category already
requires multiple independent function calls by construction). Runs
free-choice batching vs sequential (one-call-per-round), with real token
usage tracked for real dollar cost (same pattern as the CostBench n=20
sweeps), and reports LLM calls + correctness rate (via BFCL's real
ast_checker) for both conditions.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.bfcl_parallel_episode import run_parallel_episode

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bfcl", "vendor", "bfcl_eval", "data")
NUM_TASKS = 20

INPUT_PRICE_PER_TOKEN = 2.50 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 10.00 / 1_000_000


def api_dollar_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * INPUT_PRICE_PER_TOKEN + completion_tokens * OUTPUT_PRICE_PER_TOKEN


def _stats(values):
    return {
        "mean": statistics.mean(values), "min": min(values), "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def load_parallel_tasks(n: int):
    with open(os.path.join(DATA_DIR, "BFCL_v4_parallel.json")) as f:
        tasks = [json.loads(line) for line in f if line.strip()][:n]
    with open(os.path.join(DATA_DIR, "possible_answer", "BFCL_v4_parallel.json")) as f:
        ground_truths = {json.loads(line)["id"]: json.loads(line)["ground_truth"]
                          for line in f if line.strip()}
    return tasks, ground_truths


if __name__ == "__main__":
    print("call_llm_stub now makes REAL OpenAI API calls and will incur cost.")
    print("Requires OPENAI_API_KEY set in your environment (your own key -- see run_brts_minimal._get_client).")
    print(f"This run makes TWO full episodes (batched + sequential) PER TASK, {NUM_TASKS} real BFCL parallel tasks.\n")
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    tasks, ground_truths = load_parallel_tasks(NUM_TASKS)
    results = []

    for task in tasks:
        gt = ground_truths[task["id"]]
        print(f"=== {task['id']} : BATCHED ===")
        batched = run_parallel_episode(task, gt, allow_batch=True)
        batched_api_cost = api_dollar_cost(batched["total_prompt_tokens"], batched["total_completion_tokens"])
        print(f"  -> llm_calls={batched['llm_calls']} goal_reached={batched['goal_reached']} "
              f"api_cost=${batched_api_cost:.4f}\n")

        print(f"=== {task['id']} : SEQUENTIAL ===")
        sequential = run_parallel_episode(task, gt, allow_batch=False)
        sequential_api_cost = api_dollar_cost(sequential["total_prompt_tokens"], sequential["total_completion_tokens"])
        print(f"  -> llm_calls={sequential['llm_calls']} goal_reached={sequential['goal_reached']} "
              f"api_cost=${sequential_api_cost:.4f}\n")

        results.append({
            "task_id": task["id"],
            "batched_calls": batched["llm_calls"], "batched_ok": batched["goal_reached"],
            "batched_api_cost": batched_api_cost,
            "sequential_calls": sequential["llm_calls"], "sequential_ok": sequential["goal_reached"],
            "sequential_api_cost": sequential_api_cost,
        })

    print(f"=== SUMMARY ({NUM_TASKS} BFCL parallel tasks): LLM calls / correctness ===")
    print(f"{'task_id':>12} {'batch_calls':>11} {'batch_ok':>9} {'seq_calls':>9} {'seq_ok':>7}")
    for r in results:
        print(f"{r['task_id']:>12} {r['batched_calls']:>11} {str(r['batched_ok']):>9} "
              f"{r['sequential_calls']:>9} {str(r['sequential_ok']):>7}")

    total_api_cost = sum(r["batched_api_cost"] + r["sequential_api_cost"] for r in results)
    total_episodes = len(results) * 2
    total_llm_calls = sum(r["batched_calls"] + r["sequential_calls"] for r in results)

    print(f"\n=== REAL API DOLLAR COST (GPT-4o: $2.50/1M input, $10.00/1M output) ===")
    print(f"Total real API dollar cost across all {total_episodes} episodes: ${total_api_cost:.4f}")
    print(f"Cost per episode: ${total_api_cost / total_episodes:.4f}")
    print(f"Cost per LLM call: ${total_api_cost / total_llm_calls:.6f}")

    print("\n=== STATS (mean/min/max/std across 20 tasks) + correctness rate ===")
    for label, calls_key, ok_key in [("batched", "batched_calls", "batched_ok"),
                                      ("sequential", "sequential_calls", "sequential_ok")]:
        calls_stats = _stats([r[calls_key] for r in results])
        correctness_rate = sum(r[ok_key] for r in results) / len(results)
        print(f"{label}: calls mean={calls_stats['mean']:.2f} min={calls_stats['min']} "
              f"max={calls_stats['max']} std={calls_stats['std']:.2f} | "
              f"correctness_rate={correctness_rate:.0%} ({sum(r[ok_key] for r in results)}/{len(results)})")
