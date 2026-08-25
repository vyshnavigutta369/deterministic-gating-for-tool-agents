"""
Zero-cost, real-data check standing in for a literal offline log replay
(no persisted raw round-log from the earlier Phase 1 gpt-4o-mini
location+transportation sweep survives on disk -- task-output files are
ephemeral and were cleared; fabricating a transcript is not an option).

What this DOES check, honestly, with zero API calls: does apply_batch's
own real solver path (GroundtruthSolver.solve, Dijkstra-based, real cost
data) ever leave a cost-substitution opportunity on the table? Walks the
REAL optimal path round-by-round the same way run_episode would (using
get_currently_executable as `visible` each round, exactly like the real
harness), and asks whether apply_cost_substitution would ever swap
anything in.

This tells us whether the solver's own optimal choices already exhaust
same-output-type cost savings (expected: yes, since Dijkstra by
construction picks the globally cheapest path) -- so any REAL exploitable
premium must come from LLM sub-optimal tool choice, not solver structure.
"""
import sys
sys.path.insert(0, '.')
from env.domains.travel.tool_registry import tools_ready_in_memory
from env.utils.solver import GroundtruthSolver
from brts_core import apply_cost_substitution, apply_batch, get_currently_executable

for seed in range(1, 6):
    data = tools_ready_in_memory(refinement_level=5, min_atomic_cost=19, max_atomic_cost=21,
                                  noise_std=0.1, random_seed=seed)
    solver = GroundtruthSolver(data['tools'])
    tools = solver.tools
    initial_state = solver._infer_initial_state('location')

    total_subs = 0
    total_saved = 0.0
    for task in ("location", "transportation"):
        result = solver.solve(task=task)
        gt_path = result.tools
        state = initial_state
        for name in gt_path:
            visible = get_currently_executable(state, tools)
            substituted, subs = apply_cost_substitution([name], state, tools, visible)
            total_subs += len(subs)
            total_saved += sum(s["cost_saved"] for s in subs)
            state = apply_batch(state, tools, [name])

    print(f"seed={seed}: GT-optimal path substitution opportunities found = {total_subs}, "
          f"cost that would be saved = {total_saved:.2f}")
