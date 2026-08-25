"""
RIBQ-faithful scored batch selection: replaces the LLM's free-choice batching
with a deterministic, principled selection -- cluster candidates by category,
score each candidate within its cluster, pick exactly one per cluster, batch
the selections. No LLM call is needed to decide the batch, since the
selection is fully deterministic given the current state and tool registry.

Reuses real, already-verified machinery rather than reimplementing it:
- get_currently_executable / _category_of from brts_core.py, for finding and
  grouping candidate tools.
- GroundtruthSolver's real Dijkstra (_dijkstra_shortest_path via solve()) for
  D(state, goal_type) -- the actual shortest remaining cost to a goal type,
  not an approximation.

KNOWN LIMITATION (discovered by running this against real data, not
theoretical): faithful, per-category-independent scoring can select a
category-locally-optimal route that bypasses a type another still-needed
category structurally depends on. Concretely: some tools reach TravelLocation
via composite "Full_Planning"/"Refine" pipelines that never produce the
explicit LocationPreference type, because it isn't needed on LOCATION's own
shortest path -- but every entry point into the transportation (and other
non-location) branches requires LocationPreference as an input. A purely
per-cluster-optimal choice for location can therefore finish location while
permanently starving every other category, since nothing ever produces the
type they all depend on. This has no analogue in RIBQ's original single-task
clustering setting, where clusters were independent by construction and never
shared a hidden cross-cluster dependency like this. The shared-prerequisite
safeguard below is a deliberate EXTENSION beyond faithful RIBQ, added
specifically to close this gap -- not part of the original algorithm.
"""
import os
import sys
from typing import Dict, FrozenSet, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, regardless of cwd
from env.core.data_types import get_final_type
from env.utils.solver import GroundtruthSolver, NoPathFoundError
from brts_core import get_currently_executable, _category_of


def _shortest_remaining_cost(solver: GroundtruthSolver, state: FrozenSet[str], goal_type: str) -> float:
    """D(state, goal_type): real shortest remaining cost to reach goal_type from
    state, computed via GroundtruthSolver's actual Dijkstra machinery (not
    reimplemented). Unreachable is treated as infinite distance."""
    try:
        return solver.solve(current_state=state, goal_types={goal_type}).total_cost
    except NoPathFoundError:
        return float("inf")


def _shared_prerequisite_types(state: FrozenSet[str], tools: Dict, needed_categories: List[str]) -> set:
    """Extension beyond faithful RIBQ (see module docstring): types that some
    still-needed category structurally depends on (appears as an input on at
    least one of that category's own tools) but that aren't in state yet.
    Used to keep a producing category from permanently starving another
    category by choosing a locally-optimal route that happens to skip a type
    the other category needs."""
    required = set()
    for category in needed_categories:
        goal_type = get_final_type(category)
        for tool in tools.values():
            if _category_of(tool.output_type) == category.capitalize() or tool.output_type == goal_type:
                for input_type in tool.input_types:
                    if input_type not in state:
                        required.add(input_type)
    return required


def scored_batch_selection(state: FrozenSet[str], tools: Dict, categories: List[str],
                            round_tax: float = 0.0) -> List[str]:
    """
    For each category still needing progress (its final goal type not yet in
    state):
      1. Cluster: gather currently-executable tools (brts_core.get_currently_executable)
         whose output falls in this category (brts_core._category_of prefix match,
         broadened to also include the exact final type -- see KNOWN LIMITATION
         note above on why _category_of alone excludes finishing tools).
      2. Score: for each candidate tool, progress = D(state, goal) - D(state | {tool.output_type}, goal),
         score = progress / (tool.cost + round_tax). D is the real Dijkstra
         shortest-remaining-cost, via a GroundtruthSolver instantiated on the
         current tool registry. round_tax (default 0, preserving prior
         behavior) is an artificial per-round cost added only to the
         denominator of the scoring ratio -- it does NOT change the real cost
         actually charged when the tool is applied. Its purpose is to let the
         score account for rounds as a resource, not just dollars: with
         round_tax=0 the formula is pure $-efficiency and, as observed, tends
         to prefer many small cheap steps over fewer bundled ones, taking
         more rounds than free-choice LLM batching even though each step is
         individually a good deal. A positive round_tax penalizes cheap
         small-progress steps relative to expensive-but-more-progress bundle
         steps, trading dollar-efficiency for fewer rounds.
      3. Select: the highest-scoring tool in that category's cluster -- EXCEPT
         if any candidate in this cluster produces a type another still-needed
         category structurally depends on and doesn't have yet, that candidate
         is preferred even if it isn't locally cost-optimal for this category
         alone (the shared-prerequisite safeguard; a deliberate extension
         beyond faithful RIBQ, not part of the original algorithm).

    Returns the batch of selected tool names (at most one per category) --
    this is the whole round's batch, chosen without any LLM call, since
    selection is fully deterministic given state and the tool registry.
    """
    solver = GroundtruthSolver(tools)
    needed_categories = [c for c in categories if get_final_type(c) not in state]
    required_types = _shared_prerequisite_types(state, tools, needed_categories)

    selections = []
    for category in needed_categories:
        goal_type = get_final_type(category)

        cluster = [
            name for name in get_currently_executable(state, tools)
            if _category_of(tools[name].output_type) == category.capitalize()
            or tools[name].output_type == goal_type
        ]
        if not cluster:
            continue  # blocked this round -- no candidate tool executable yet

        dist_before = _shortest_remaining_cost(solver, state, goal_type)

        scored = []
        for name in cluster:
            tool = tools[name]
            state_after = frozenset(state | {tool.output_type})
            dist_after = _shortest_remaining_cost(solver, state_after, goal_type)
            progress = dist_before - dist_after
            score = progress / (tool.cost + round_tax)
            scored.append((name, score, tool.output_type in required_types))

        # Shared-prerequisite safeguard: prefer a required-type producer over
        # this category's own locally-optimal candidate, if one exists here.
        required_candidates = [s for s in scored if s[2]]
        pool = required_candidates if required_candidates else scored
        best_name, _, _ = max(pool, key=lambda s: s[1])
        selections.append(best_name)

    return selections
