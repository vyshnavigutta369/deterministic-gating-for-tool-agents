"""
Estimate run-to-run LLM noise: repeat each of 3 conditions (sequential, batched,
batched cost-aware) 3 times each, for 3 seeds only (1, 5, 7 -- a "small effect"
and two "large effect" seeds from the earlier cost-aware comparison). 27
episodes total. Reports mean/min/max/stdev of llm_calls and total_tool_cost
per seed x condition, to see how much of the earlier cost-aware deltas were
real signal vs. temperature=0 sampling noise (confirmed non-zero separately).
"""
import os
import sys
import statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from run_brts_minimal import run_episode

SEEDS = [1, 5, 7]
CONDITIONS = [
    ("sequential", dict(allow_batch=False, allow_batch_cost_aware=False)),
    ("batched", dict(allow_batch=True, allow_batch_cost_aware=False)),
    ("batched_cost_aware", dict(allow_batch=True, allow_batch_cost_aware=True)),
]
REPEATS = 3

results = {}  # (seed, label) -> list of (llm_calls, total_tool_cost, goal_reached)

for seed in SEEDS:
    for label, kwargs in CONDITIONS:
        runs = []
        for rep in range(REPEATS):
            print(f"=== seed={seed} condition={label} rep={rep + 1}/{REPEATS} ===")
            r = run_episode(task="location", categories=["location", "transportation"],
                             seed=seed, **kwargs)
            print(f"  -> llm_calls={r['llm_calls']} total_tool_cost={r['total_tool_cost']:.2f} "
                  f"goal_reached={r['goal_reached']}\n")
            runs.append((r["llm_calls"], r["total_tool_cost"], r["goal_reached"]))
        results[(seed, label)] = runs

print("=== NOISE ESTIMATE: mean / min / max / stdev per seed x condition (n=3 each) ===")
header = (f"{'seed':>4} {'condition':>19} "
          f"{'calls_mean':>10} {'calls_min':>9} {'calls_max':>9} {'calls_std':>9} "
          f"{'cost_mean':>10} {'cost_min':>9} {'cost_max':>9} {'cost_std':>9} {'all_goal_ok':>11}")
print(header)
for seed in SEEDS:
    for label, _ in CONDITIONS:
        runs = results[(seed, label)]
        calls = [r[0] for r in runs]
        costs = [r[1] for r in runs]
        oks = [r[2] for r in runs]
        calls_std = statistics.stdev(calls) if len(calls) > 1 else 0.0
        cost_std = statistics.stdev(costs) if len(costs) > 1 else 0.0
        print(f"{seed:>4} {label:>19} "
              f"{statistics.mean(calls):>10.2f} {min(calls):>9} {max(calls):>9} {calls_std:>9.2f} "
              f"{statistics.mean(costs):>10.2f} {min(costs):>9.2f} {max(costs):>9.2f} {cost_std:>9.2f} "
              f"{str(all(oks)):>11}")
