"""
Precompute-length-optimal selection: identical structure to
brts.precomputed's precompute_then_batch_selection, except each category's
precomputed sequence minimizes STEP COUNT first (fewest tool invocations to
reach the category's goal type), breaking ties by lowest total cost among
equally-short paths -- rather than GroundtruthSolver.solve()'s cost-optimal
Dijkstra (which minimizes cost first).

GroundtruthSolver has no length-optimal search built in (its solve() always
optimizes cost), so this implements the search loop directly -- but reuses
GroundtruthSolver's real successor logic (_get_successors) rather than
reimplementing "which tools are applicable from this state". This is exactly
Dijkstra with the priority tuple's sort keys swapped: (path_length, cost)
instead of GroundtruthSolver's own (cost, path_length) -- BFS-with-a-cost-
tiebreaker is precisely lexicographic Dijkstra ordered length-first.

Execution reuses execute_precomputed_sequences from brts.precomputed
unchanged -- the lockstep round-robin + shared-prerequisite deduplication
logic is identical regardless of how each category's sequence was computed.
"""
import heapq
import os
import sys
from typing import Dict, FrozenSet, List, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, regardless of cwd
from env.core.data_types import get_final_type
from env.utils.solver import GroundtruthSolver, NoPathFoundError, PathResult
from .precomputed import execute_precomputed_sequences


def _length_optimal_solve(solver: GroundtruthSolver, current_state: FrozenSet[str], goal_types: Set[str]) -> PathResult:
    """Same interface as GroundtruthSolver.solve(current_state=..., goal_types=...),
    but minimizes path_length first and total cost second, using the solver's
    real _get_successors for successor generation (not reimplemented)."""
    pq = [(0, 0.0, current_state, [])]  # (path_length, total_cost, state, path)
    visited = set()

    while pq:
        path_length, total_cost, state, path = heapq.heappop(pq)

        if state in visited:
            continue
        visited.add(state)

        if goal_types.issubset(state):
            return PathResult(tools=path, total_cost=total_cost, path_length=path_length, state_sequence=[])

        for next_state, tool_name, tool_cost in solver._get_successors(state):
            if next_state not in visited:
                heapq.heappush(pq, (path_length + 1, total_cost + tool_cost, next_state, path + [tool_name]))

    raise NoPathFoundError(f"No path found from {current_state} to target containing {goal_types}")


def precompute_length_optimal_selection(initial_state: FrozenSet[str], tools: Dict, categories: List[str],
                                         max_rounds: int = 100) -> Dict:
    """
    Precompute: for each category still needing progress, find its
    length-optimal (fewest steps, cost-tiebroken) tool sequence via
    _length_optimal_solve, ONCE.

    Execute: reuses execute_precomputed_sequences (lockstep + shared-
    prerequisite deduplication) from brts.precomputed, unchanged.

    Returns a dict with: rounds, total_cost, goal_reached, sequences,
    round_log (see execute_precomputed_sequences), plus
    per_category_reported_cost (the length-optimal solve's own total_cost per
    category, computed in isolation).
    """
    solver = GroundtruthSolver(tools)

    sequences = {}
    per_category_reported_cost = {}
    for category in categories:
        goal_type = get_final_type(category)
        if goal_type in initial_state:
            sequences[category] = []
            per_category_reported_cost[category] = 0.0
            continue
        result = _length_optimal_solve(solver, initial_state, {goal_type})
        sequences[category] = result.tools
        per_category_reported_cost[category] = result.total_cost

    result = execute_precomputed_sequences(initial_state, tools, categories, sequences, max_rounds=max_rounds)
    result["per_category_reported_cost"] = per_category_reported_cost
    return result
