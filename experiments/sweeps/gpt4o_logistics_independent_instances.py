"""
gpt-4o, PDDL logistics domain, the two verified-independent instances
(logistics/task_independent.pddl -- 3 fully parallel truck-only deliveries;
logistics/task_independent_mixed.pddl -- 2 truck-only + 1 truck-then-airplane
delivery). Runs free-choice batching vs sequential (one-action-per-round)
via run_logistics_episode from logistics_llm_harness.py, with real token
usage tracked for real dollar cost (same GPT-4o pricing convention as the
other sweeps), and reports LLM calls + goal_reached for both conditions on
both instances.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from logistics_llm_harness import run_logistics_episode

LOGISTICS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logistics")
DOMAIN_FILE = os.path.join(LOGISTICS_DIR, "logistics_domain.pddl")
INSTANCES = {
    "task_independent": os.path.join(LOGISTICS_DIR, "task_independent.pddl"),
    "task_independent_mixed": os.path.join(LOGISTICS_DIR, "task_independent_mixed.pddl"),
}

INPUT_PRICE_PER_TOKEN = 2.50 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 10.00 / 1_000_000


def api_dollar_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * INPUT_PRICE_PER_TOKEN + completion_tokens * OUTPUT_PRICE_PER_TOKEN


if __name__ == "__main__":
    print("call_llm_stub now makes REAL OpenAI API calls and will incur cost.")
    print("Requires OPENAI_API_KEY set in your environment (your own key -- see run_brts_minimal._get_client).\n")
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    results = []
    for instance_name, problem_file in INSTANCES.items():
        for allow_batch in (True, False):
            label = "BATCHED" if allow_batch else "SEQUENTIAL"
            print(f"=== {instance_name} : {label} ===")
            r = run_logistics_episode(DOMAIN_FILE, problem_file, allow_batch=allow_batch)
            cost = api_dollar_cost(r["total_prompt_tokens"], r["total_completion_tokens"])
            print(f"  -> llm_calls={r['llm_calls']} goal_reached={r['goal_reached']} api_cost=${cost:.4f}\n")
            results.append({
                "instance": instance_name, "condition": label,
                "llm_calls": r["llm_calls"], "goal_reached": r["goal_reached"], "api_cost": cost,
            })

    print("=== SUMMARY: LLM calls / goal_reached per instance x condition ===")
    print(f"{'instance':>24} {'condition':>10} {'llm_calls':>9} {'goal_reached':>12} {'api_cost':>9}")
    for r in results:
        print(f"{r['instance']:>24} {r['condition']:>10} {r['llm_calls']:>9} "
              f"{str(r['goal_reached']):>12} {r['api_cost']:>9.4f}")

    total_cost = sum(r["api_cost"] for r in results)
    print(f"\nTotal real API dollar cost across {len(results)} episodes: ${total_cost:.4f}")
