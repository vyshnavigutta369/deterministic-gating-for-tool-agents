"""
claude-haiku-4-5, synthetic no-bundle-tools domain (attraction+dining+shopping),
seeds 1-5. Mirrors gpt4o_synthetic_5seed.py exactly (same categories, same
seed range, same TWO-episode-per-seed structure), swapping model + pricing.
Uses run_synthetic_episode (experiments/synthetic_episode.py), now
Claude-aware: temperature=0 + robust JSON extraction (via call_llm_stub_claude)
and empty-response corrective feedback both apply here unmodified, since this
domain shares CostBench's "empty proposal = stall" semantics.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from experiments.synthetic_episode import run_synthetic_episode

MODEL = "claude-haiku-4-5-20251001"
INPUT_PRICE_PER_TOKEN = 1.00 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 5.00 / 1_000_000
CATEGORIES = ["attraction", "dining", "shopping"]  # same categories/scale as the synthetic no-LLM test


def api_dollar_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * INPUT_PRICE_PER_TOKEN + completion_tokens * OUTPUT_PRICE_PER_TOKEN


if __name__ == "__main__":
    print(f"call_llm_stub now makes REAL Anthropic API calls (model={MODEL}) and will incur cost.")
    print("Requires ANTHROPIC_API_KEY set in your environment.")
    print(f"This run makes TWO full episodes (batched + sequential) PER SEED, seeds 1-5, "
          f"SYNTHETIC domain (no bundle tools), categories={CATEGORIES}.\n")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    seed_results = []
    for seed in range(1, 6):
        print(f"=== BATCHED (allow_batch=True), seed={seed} ===")
        batched = run_synthetic_episode(CATEGORIES, seed=seed, allow_batch=True, model=MODEL)
        batched_api_cost = api_dollar_cost(batched["total_prompt_tokens"], batched["total_completion_tokens"])
        print(f"Batched summary: rounds={batched['rounds']}, llm_calls={batched['llm_calls']}, "
              f"total_tool_cost={batched['total_tool_cost']:.2f}, real_api_dollar_cost=${batched_api_cost:.4f}, "
              f"goal_reached={batched['goal_reached']}\n")

        print(f"=== SEQUENTIAL (allow_batch=False), same seed={seed} ===")
        sequential = run_synthetic_episode(CATEGORIES, seed=seed, allow_batch=False, model=MODEL)
        sequential_api_cost = api_dollar_cost(sequential["total_prompt_tokens"], sequential["total_completion_tokens"])
        print(f"Sequential summary: rounds={sequential['rounds']}, llm_calls={sequential['llm_calls']}, "
              f"total_tool_cost={sequential['total_tool_cost']:.2f}, real_api_dollar_cost=${sequential_api_cost:.4f}, "
              f"goal_reached={sequential['goal_reached']}\n")

        seed_results.append({
            "seed": seed,
            "batched_calls": batched["llm_calls"], "batched_cost": batched["total_tool_cost"],
            "batched_api_cost": batched_api_cost, "batched_ok": batched["goal_reached"],
            "sequential_calls": sequential["llm_calls"], "sequential_cost": sequential["total_tool_cost"],
            "sequential_api_cost": sequential_api_cost, "sequential_ok": sequential["goal_reached"],
        })

    print(f"=== SUMMARY (seeds 1-5, synthetic domain, categories={CATEGORIES}, model={MODEL}) ===")
    print(f"{'seed':>4} {'b_calls':>7} {'b_cost':>9} {'b_apicost':>10} {'b_ok':>5} | "
          f"{'s_calls':>7} {'s_cost':>9} {'s_apicost':>10} {'s_ok':>5}")
    for r in seed_results:
        print(f"{r['seed']:>4} {r['batched_calls']:>7} {r['batched_cost']:>9.2f} "
              f"{r['batched_api_cost']:>10.4f} {str(r['batched_ok']):>5} | "
              f"{r['sequential_calls']:>7} {r['sequential_cost']:>9.2f} "
              f"{r['sequential_api_cost']:>10.4f} {str(r['sequential_ok']):>5}")

    failures = [f"seed={r['seed']} {'BATCHED' if not r['batched_ok'] else 'SEQUENTIAL'} FAILED"
                for r in seed_results if not r["batched_ok"] or not r["sequential_ok"]]
    print(f"\nFailures: {failures if failures else 'None -- all episodes reached the goal.'}")

    avg_batch_calls = sum(r["batched_calls"] for r in seed_results) / len(seed_results)
    avg_batch_cost = sum(r["batched_cost"] for r in seed_results) / len(seed_results)
    avg_seq_calls = sum(r["sequential_calls"] for r in seed_results) / len(seed_results)
    avg_seq_cost = sum(r["sequential_cost"] for r in seed_results) / len(seed_results)
    total_api_cost = sum(r["batched_api_cost"] + r["sequential_api_cost"] for r in seed_results)
    print(f"\navg batched:    {avg_batch_calls:.2f} calls / {avg_batch_cost:.2f} tool cost")
    print(f"avg sequential: {avg_seq_calls:.2f} calls / {avg_seq_cost:.2f} tool cost")
    print(f"total real API dollar cost across {len(seed_results) * 2} episodes: ${total_api_cost:.4f}")
