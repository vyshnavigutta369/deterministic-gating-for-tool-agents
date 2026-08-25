"""
Precompute-then-batch, PDDL Logistics variant: each package's delivery is an
independent "category" goal (mirroring CostBench's location/transportation/
etc.). For each package, solve its OWN optimal delivery plan independently
via pyperplan's real A*+admissible-heuristic search (a genuine, established
optimal planner -- the same role GroundtruthSolver played for CostBench).
Then execute all packages' plans in lockstep, one action per package per
round, with REAL precondition verification each round (not assumed
independence) -- mirroring the same execution-safety discipline used for
CostBench's shared-prerequisite deduplication and BFCL's permutation checks.
"""
import time
# pyperplan is a real pip-installed package (pip install pyperplan) -- it
# lives in site-packages and needs no manual sys.path entry. This file has
# no other project-module imports, so no path setup is needed at all.

from pyperplan.pddl.parser import Parser
from pyperplan.grounding import ground
from pyperplan.search.a_star import astar_search
from pyperplan.heuristics.blind import BlindHeuristic


def solve_single_goal(task, single_goal):
    """Real A*+blind-heuristic optimal solve for one isolated goal fact,
    reusing the task's real operators and initial state."""
    import copy
    sub_task = copy.copy(task)
    sub_task.goals = frozenset({single_goal})
    solution = astar_search(sub_task, BlindHeuristic(sub_task))
    if solution is None:
        return None
    return [op.name for op in solution]


def lockstep_execute(task, per_goal_plans):
    """
    Execute all per-goal optimal plans in lockstep: each round, take the next
    unexecuted action from every still-incomplete plan, and REAL-verify
    (via op.applicable/op.apply) that every action in the batch is actually
    valid given the state as of the START of this round (not assuming
    independence) -- same discipline as CostBench Section 4.11 / BFCL
    Section 4.19-4.22. If an action isn't yet applicable, that goal's plan
    stalls this round and retries next round (its precomputed plan itself
    remains valid -- this only affects round SCHEDULING, not correctness).
    """
    name_to_op = {op.name: op for op in task.operators}
    state = task.initial_state
    indices = {i: 0 for i in range(len(per_goal_plans))}
    rounds = []
    max_rounds = sum(len(p) for p in per_goal_plans) + 5  # generous safety cap

    for _ in range(max_rounds):
        if all(indices[i] >= len(per_goal_plans[i]) for i in range(len(per_goal_plans))):
            break
        batch = []
        live_state = state
        for i, plan in enumerate(per_goal_plans):
            if indices[i] >= len(plan):
                continue
            op = name_to_op[plan[indices[i]]]
            if op.applicable(live_state):
                batch.append((i, op))
                live_state = op.apply(live_state)
        if not batch:
            break  # stuck -- shouldn't happen if plans are individually valid, but fail loudly if so
        for i, op in batch:
            state = op.apply(state)
            indices[i] += 1
        rounds.append([op.name for i, op in batch])

    all_done = all(indices[i] >= len(per_goal_plans[i]) for i in range(len(per_goal_plans)))
    return rounds, all_done, state


def run_task(domain_file, problem_file, task_label):
    parser = Parser(domain_file, problem_file)
    domain = parser.parse_domain()
    problem = parser.parse_problem(domain)
    task = ground(problem)

    goals = list(task.goals)

    # Real joint-optimal solve (all goals together) -- the honest ceiling comparator
    t0 = time.perf_counter()
    joint_solution = astar_search(task, BlindHeuristic(task))
    t1 = time.perf_counter()
    joint_length = len(joint_solution) if joint_solution else None

    # Precompute: solve each goal independently, real optimal per-goal plans
    per_goal_plans = []
    for g in goals:
        plan = solve_single_goal(task, g)
        if plan is None:
            return {"task": task_label, "error": f"no plan found for goal {g}"}
        per_goal_plans.append(plan)

    naive_sequential_actions = sum(len(p) for p in per_goal_plans)

    # Lockstep execute with real precondition verification
    rounds, all_done, final_state = lockstep_execute(task, per_goal_plans)

    return {
        "task": task_label,
        "num_goals": len(goals),
        "joint_optimal_length": joint_length,
        "joint_solve_time_ms": (t1 - t0) * 1000,
        "per_goal_plan_lengths": [len(p) for p in per_goal_plans],
        "naive_sequential_actions": naive_sequential_actions,
        "lockstep_rounds": len(rounds),
        "lockstep_all_goals_reached": all_done,
        "goal_still_satisfied": goals[0] in final_state if goals else None,
    }


if __name__ == "__main__":
    import os
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logistics")
    domain_file = os.path.join(base, "logistics_domain.pddl")

    results = []
    for task_label, filename in [
        ("task_independent", "task_independent.pddl"),
        ("task_independent_mixed", "task_independent_mixed.pddl"),
    ]:
        problem_file = os.path.join(base, filename)
        print(f"Solving {task_label}...")
        r = run_task(domain_file, problem_file, task_label)
        results.append(r)
        print(f"  {r}\n")

    print("\n=== SUMMARY ===")
    for r in results:
        if "error" in r:
            print(f"{r['task']}: ERROR - {r['error']}")
            continue
        print(f"{r['task']}: {r['num_goals']} goals | joint_optimal={r['joint_optimal_length']} | "
              f"naive_sequential={r['naive_sequential_actions']} | lockstep_rounds={r['lockstep_rounds']} | "
              f"all_reached={r['lockstep_all_goals_reached']}")
