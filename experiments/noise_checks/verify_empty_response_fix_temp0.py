"""
Targeted live verification (NOT a full rerun) of the empty-response
corrective-feedback fix in run_brts_minimal.py: reruns exactly the 4
episodes that got permanently stuck at temperature=0 (location+transportation
seed=1, location+transportation+dining seeds 1/3/4, all BATCHED) to confirm
the fix breaks the deterministic deadlock. Scoped to just these 4 -- not
the other 26 episodes that already succeeded -- to keep cost minimal before
deciding whether a full 3-pairing/5-seed rerun is warranted.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from run_brts_minimal import run_episode

MODEL = "claude-haiku-4-5-20251001"
INPUT_PRICE_PER_TOKEN = 1.00 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 5.00 / 1_000_000

CASES = [
    ("location", ["location", "transportation"], 1),
    ("travel", ["location", "transportation", "dining"], 1),
    ("travel", ["location", "transportation", "dining"], 3),
    ("travel", ["location", "transportation", "dining"], 4),
]


def api_dollar_cost(prompt_tokens, completion_tokens):
    return prompt_tokens * INPUT_PRICE_PER_TOKEN + completion_tokens * OUTPUT_PRICE_PER_TOKEN


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    results = []
    for task, categories, seed in CASES:
        label = f"{'+'.join(categories)} seed={seed}"
        print(f"=== BATCHED, {label} (previously stuck all 30 rounds at temperature=0) ===")
        r = run_episode(task=task, categories=categories, seed=seed, allow_batch=True, model=MODEL)
        cost = api_dollar_cost(r["total_prompt_tokens"], r["total_completion_tokens"])
        print(f"  -> rounds={r['rounds']} llm_calls={r['llm_calls']} goal_reached={r['goal_reached']} "
              f"api_cost=${cost:.4f}\n")
        results.append({"label": label, "rounds": r["rounds"], "llm_calls": r["llm_calls"],
                         "goal_reached": r["goal_reached"], "api_cost": cost})

    print("=== SUMMARY: empty-response fix verification (previously-stuck episodes only) ===")
    for r in results:
        print(f"{r['label']:>45}: rounds={r['rounds']:>3} llm_calls={r['llm_calls']:>3} "
              f"goal_reached={str(r['goal_reached']):>5} api_cost=${r['api_cost']:.4f}")
    fixed = sum(1 for r in results if r["goal_reached"])
    print(f"\n{fixed}/{len(results)} previously-stuck episodes now reach the goal.")
    print(f"Total real API dollar cost: ${sum(r['api_cost'] for r in results):.4f}")
