"""
Zero-cost real-data check, run BEFORE any real n=5 API test: does the real
CostBench tool registry actually contain same-output-type/different-cost
tool pairs at all? If not, there's no real opportunity for
apply_cost_substitution to ever do anything, and no real experiment is
worth running -- this is exactly the kind of cheap check to do first.

No API calls, no LLM involved -- just loads the real registry (same
tools_ready_in_memory + GroundtruthSolver machinery run_episode itself
uses) and groups real Tool objects by output_type.
"""
import sys
sys.path.insert(0, '.')
from collections import defaultdict
from env.domains.travel.tool_registry import tools_ready_in_memory
from env.utils.solver import GroundtruthSolver

SEEDS_TO_CHECK = range(1, 11)

for seed in SEEDS_TO_CHECK:
    data = tools_ready_in_memory(refinement_level=5, min_atomic_cost=19, max_atomic_cost=21,
                                  noise_std=0.1, random_seed=seed)
    solver = GroundtruthSolver(data['tools'])
    tools = solver.tools

    by_output = defaultdict(list)
    for name, tool in tools.items():
        by_output[tool.output_type].append((name, tool.cost))

    real_opportunities = {ot: sorted(pairs, key=lambda p: p[1])
                           for ot, pairs in by_output.items()
                           if len(pairs) > 1 and len(set(c for _, c in pairs)) > 1}

    print(f"seed={seed}: {len(tools)} total tools, {len(by_output)} distinct output_types, "
          f"{len(real_opportunities)} output_types with >=2 tools at DIFFERENT costs")
    if seed == 1:
        print("  sample (seed=1), first 8 output_types with real cost spread:")
        for ot, pairs in list(real_opportunities.items())[:8]:
            cheapest, priciest = pairs[0], pairs[-1]
            print(f"    {ot}: {len(pairs)} tools, cost range [{cheapest[1]}, {priciest[1]}] "
                  f"-- cheapest={cheapest[0]}, priciest={priciest[0]}")
