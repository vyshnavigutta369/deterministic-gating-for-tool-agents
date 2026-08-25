"""
gpt-4o, PDDL logistics task_independent_mixed ONLY, with max_rounds raised
from 15 to 30 (matching the CostBench max_rounds convention used elsewhere
in this project). The first run at max_rounds=15 failed to complete in both
conditions despite the cycle-detection fix working correctly (it fired
multiple times, changed behavior each time) -- this checks whether that was
a round-budget problem rather than a capability ceiling: a perfect batched
plan for this instance needs a theoretical minimum of ~6 rounds (load all
3 -> drive all 3 -> unload all 3 -> load airplane -> fly -> unload airplane),
so 15 was tight but 30 gives real room for self-correction detours.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from logistics_llm_harness import run_logistics_episode

LOGISTICS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logistics")
DOMAIN_FILE = os.path.join(LOGISTICS_DIR, "logistics_domain.pddl")
PROBLEM_FILE = os.path.join(LOGISTICS_DIR, "task_independent_mixed.pddl")
MAX_ROUNDS = 30

INPUT_PRICE_PER_TOKEN = 2.50 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 10.00 / 1_000_000


def api_dollar_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * INPUT_PRICE_PER_TOKEN + completion_tokens * OUTPUT_PRICE_PER_TOKEN


if __name__ == "__main__":
    print("call_llm_stub now makes REAL OpenAI API calls and will incur cost.")
    print("Requires OPENAI_API_KEY set in your environment (your own key -- see run_brts_minimal._get_client).")
    print(f"task_independent_mixed only, max_rounds={MAX_ROUNDS} (was 15).\n")
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    results = []
    for allow_batch in (True, False):
        label = "BATCHED" if allow_batch else "SEQUENTIAL"
        print(f"=== task_independent_mixed : {label} (max_rounds={MAX_ROUNDS}) ===")
        r = run_logistics_episode(DOMAIN_FILE, PROBLEM_FILE, allow_batch=allow_batch, max_rounds=MAX_ROUNDS)
        cost = api_dollar_cost(r["total_prompt_tokens"], r["total_completion_tokens"])
        print(f"  -> llm_calls={r['llm_calls']} goal_reached={r['goal_reached']} api_cost=${cost:.4f}\n")
        results.append({"condition": label, "llm_calls": r["llm_calls"],
                         "goal_reached": r["goal_reached"], "api_cost": cost})

    print("=== SUMMARY: task_independent_mixed, max_rounds=30 ===")
    print(f"{'condition':>10} {'llm_calls':>9} {'goal_reached':>12} {'api_cost':>9}")
    for r in results:
        print(f"{r['condition']:>10} {r['llm_calls']:>9} {str(r['goal_reached']):>12} {r['api_cost']:>9.4f}")
    print(f"\nTotal real API dollar cost: ${sum(r['api_cost'] for r in results):.4f}")
