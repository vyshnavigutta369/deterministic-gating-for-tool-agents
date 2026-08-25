"""
claude-haiku-4-5, PDDL logistics domain, the two verified-independent
instances (task_independent, task_independent_mixed). Mirrors
gpt4o_logistics_independent_instances.py exactly (same instances, same
max_rounds=15 default, same batched/sequential structure), swapping model +
pricing. run_logistics_episode is now Claude-aware: temperature=0 + robust
JSON extraction (via call_llm_stub_claude), and empty-response corrective
feedback now applies alongside the pre-existing cycle-detection feedback.
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
MODEL = "claude-haiku-4-5-20251001"

INPUT_PRICE_PER_TOKEN = 1.00 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 5.00 / 1_000_000


def api_dollar_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * INPUT_PRICE_PER_TOKEN + completion_tokens * OUTPUT_PRICE_PER_TOKEN


if __name__ == "__main__":
    print(f"call_llm_stub now makes REAL Anthropic API calls (model={MODEL}) and will incur cost.")
    print("Requires ANTHROPIC_API_KEY set in your environment.\n")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    results = []
    for instance_name, problem_file in INSTANCES.items():
        for allow_batch in (True, False):
            label = "BATCHED" if allow_batch else "SEQUENTIAL"
            print(f"=== {instance_name} : {label} ===")
            r = run_logistics_episode(DOMAIN_FILE, problem_file, allow_batch=allow_batch, model=MODEL)
            cost = api_dollar_cost(r["total_prompt_tokens"], r["total_completion_tokens"])
            print(f"  -> llm_calls={r['llm_calls']} goal_reached={r['goal_reached']} api_cost=${cost:.4f}\n")
            results.append({
                "instance": instance_name, "condition": label,
                "llm_calls": r["llm_calls"], "goal_reached": r["goal_reached"], "api_cost": cost,
            })

    print(f"=== SUMMARY ({MODEL}): LLM calls / goal_reached per instance x condition ===")
    print(f"{'instance':>24} {'condition':>10} {'llm_calls':>9} {'goal_reached':>12} {'api_cost':>9}")
    for r in results:
        print(f"{r['instance']:>24} {r['condition']:>10} {r['llm_calls']:>9} "
              f"{str(r['goal_reached']):>12} {r['api_cost']:>9.4f}")

    total_cost = sum(r["api_cost"] for r in results)
    print(f"\nTotal real API dollar cost across {len(results)} episodes: ${total_cost:.4f}")
