"""
Noise check for seed=3, categories=[location, transportation, dining], which
showed batching cheaper AND fewer-call in all 5 seeds of that sweep (the most
consistent result of any pairing tested so far). Repeats both conditions 3x
each (6 episodes total) for seed 3 specifically to see if that consistency is
real or partly a single-run artifact, same method as the earlier noise checks.
Reuses run_episode from run_brts_minimal.py unchanged.
"""
import os
import sys
import statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from run_brts_minimal import run_episode

SEED = 3
CATEGORIES = ["location", "transportation", "dining"]
CONDITIONS = [
    ("sequential", dict(allow_batch=False)),
    ("batched", dict(allow_batch=True)),
]
REPEATS = 3

ORIGINAL = {
    "batched": {"calls": 4, "cost": 522.02},
    "sequential": {"calls": 19, "cost": 765.34},
}

results = {}
for label, kwargs in CONDITIONS:
    runs = []
    for rep in range(REPEATS):
        print(f"=== seed={SEED} condition={label} rep={rep + 1}/{REPEATS} ===")
        r = run_episode(task="travel", categories=CATEGORIES, seed=SEED, **kwargs)
        print(f"  -> llm_calls={r['llm_calls']} total_tool_cost={r['total_tool_cost']:.2f} "
              f"goal_reached={r['goal_reached']}\n")
        runs.append((r["llm_calls"], r["total_tool_cost"], r["goal_reached"]))
    results[label] = runs

print("=== NOISE CHECK: seed=3, location+transportation+dining (n=3 each) ===")
header = (f"{'condition':>11} {'calls_mean':>10} {'calls_min':>9} {'calls_max':>9} {'calls_std':>9} "
          f"{'cost_mean':>10} {'cost_min':>9} {'cost_max':>9} {'cost_std':>9} {'orig_calls':>10} {'orig_cost':>9} {'all_ok':>7}")
print(header)
for label, _ in CONDITIONS:
    runs = results[label]
    calls = [r[0] for r in runs]
    costs = [r[1] for r in runs]
    oks = [r[2] for r in runs]
    calls_std = statistics.stdev(calls) if len(calls) > 1 else 0.0
    cost_std = statistics.stdev(costs) if len(costs) > 1 else 0.0
    print(f"{label:>11} "
          f"{statistics.mean(calls):>10.2f} {min(calls):>9} {max(calls):>9} {calls_std:>9.2f} "
          f"{statistics.mean(costs):>10.2f} {min(costs):>9.2f} {max(costs):>9.2f} {cost_std:>9.2f} "
          f"{ORIGINAL[label]['calls']:>10} {ORIGINAL[label]['cost']:>9.2f} {str(all(oks)):>7}")

orig_diff = ORIGINAL["batched"]["cost"] - ORIGINAL["sequential"]["cost"]
new_batch_mean = statistics.mean([r[1] for r in results["batched"]])
new_seq_mean = statistics.mean([r[1] for r in results["sequential"]])
new_diff_mean = new_batch_mean - new_seq_mean
print(f"\nOriginal single-run cost diff (batched - sequential): {orig_diff:.2f}")
print(f"New 3-rep mean cost diff (batched - sequential): {new_diff_mean:.2f}")
