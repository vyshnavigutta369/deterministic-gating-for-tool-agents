"""
Precompute-then-batch selection: for each category, call GroundtruthSolver's
real Dijkstra machinery ONCE upfront to get its full, cost-optimal tool
sequence, then round-robin exactly one step per still-active category per
round until every sequence is exhausted. Fully deterministic -- zero LLM
calls, and no per-round scoring at all (unlike brts.scored's greedy
step-by-step selection, which recomputes distances every round).

Because each category's sequence comes from a real, complete Dijkstra path to
its OWN goal from the shared initial state, it already accounts for any
genuine cross-category prerequisite (e.g. a transportation path that must
first pass through Decide_Location_Preference, since LocationPreference is a
real input dependency of every transportation entry point) -- there is no
analogue here of brts.scored's shared-prerequisite blocking bug, because
Dijkstra finds the true necessary path for each category on its own, rather
than a per-round greedy choice that can permanently skip a shared dependency.

DEDUPLICATION AGAINST A REAL SHARED STATE: because each category's sequence
is computed in ISOLATION (as if it alone had to pay to reach its own goal),
two or more categories can independently include the very same prerequisite
tool in their own plans (e.g. both transportation's and dining's sequences
each including Decide_Location_Preference, since both need LocationPreference
and neither knows the other also needs it). Executing every category's steps
blindly would charge that tool's cost once per category that happens to list
it -- a real, quantified inefficiency (confirmed: total cost exactly equaled
the naive sum of each category's independently-reported cost, which double-
or triple-counts any shared prerequisite). The execution loop below instead
tracks ONE real cumulative state across all categories and, before executing
any category's next step each round, skips that step for free (no cost, no
round consumed) if its output is already present -- whether that's because a
PRIOR round already produced it, or because another category is producing
the exact same output THIS SAME round (checked via a live in-round state so
same-round duplicates are also caught, not just cross-round ones).
"""
import os
import sys
from typing import Dict, FrozenSet, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, regardless of cwd
from env.core.data_types import get_final_type
from env.utils.solver import GroundtruthSolver
from brts_core import apply_batch, is_goal_reached


def execute_precomputed_sequences(initial_state: FrozenSet[str], tools: Dict, categories: List[str],
                                   sequences: Dict[str, List[str]], max_rounds: int = 100) -> Dict:
    """
    Shared lockstep execution + shared-prerequisite deduplication, factored
    out so any precompute-a-sequence-per-category variant (cost-optimal here,
    length-optimal in brts.length_optimal) can reuse the same, already
    verified execution logic without reimplementing it.

    Each round, and for each category still holding unexecuted steps, skip
    ahead (free -- no cost, no round consumed) past any of that category's
    next steps whose output is already present in the one real cumulative
    shared state (checked live, so a duplicate produced by another category
    earlier in this SAME round is caught too, not just prior rounds'
    outputs). The first genuinely new-output step found becomes that
    category's contribution to this round's batch. Continue until every
    sequence is exhausted (skipped-to-the-end counts as exhausted). No LLM
    call and no per-round scoring is involved -- the whole plan was fixed
    upfront; only the redundancy-skipping is decided per round.

    Returns a dict with: rounds, total_cost (real, accumulated only for
    steps actually executed -- deduplicated against the shared state),
    goal_reached, sequences (as given), and round_log (the actual executed
    batch per round).
    """
    indices = {category: 0 for category in categories}
    state = initial_state
    target_types = {get_final_type(c) for c in categories}

    rounds = 0
    total_cost = 0.0
    round_log = []
    while any(indices[c] < len(sequences[c]) for c in categories) and rounds < max_rounds:
        batch = []
        live_state = state  # provisional: includes this round's picks so far, for same-round dedup
        for category in categories:
            i = indices[category]
            seq = sequences[category]
            while i < len(seq) and tools[seq[i]].output_type in live_state:
                i += 1  # free skip -- output already available, no cost, no round consumed
            if i < len(seq):
                tool_name = seq[i]
                batch.append(tool_name)
                live_state = live_state | {tools[tool_name].output_type}
                i += 1
            indices[category] = i
        if not batch:
            break
        total_cost += sum(tools[name].cost for name in batch)
        state = apply_batch(state, tools, batch)
        round_log.append(list(batch))
        rounds += 1

    return {
        "rounds": rounds,
        "total_cost": total_cost,
        "goal_reached": is_goal_reached(state, target_types),
        "sequences": sequences,
        "round_log": round_log,
    }


def precompute_then_batch_selection(initial_state: FrozenSet[str], tools: Dict, categories: List[str],
                                     max_rounds: int = 100) -> Dict:
    """
    Precompute: for each category still needing progress, call
    GroundtruthSolver.solve(current_state=initial_state, goal_types={goal_type})
    ONCE -- the real Dijkstra shortest path -- giving that category's full,
    cost-optimal tool sequence (PathResult.tools) and its reported total_cost.

    Execute: reuses execute_precomputed_sequences (lockstep + shared-
    prerequisite deduplication).

    Returns a dict with: rounds, total_cost, goal_reached, sequences,
    round_log (see execute_precomputed_sequences), plus
    per_category_reported_cost (solver.solve()'s own total_cost per category,
    computed in isolation -- comparing this against total_cost shows exactly
    how much the deduplication saved).
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
        result = solver.solve(current_state=initial_state, goal_types={goal_type})
        sequences[category] = result.tools
        per_category_reported_cost[category] = result.total_cost

    result = execute_precomputed_sequences(initial_state, tools, categories, sequences, max_rounds=max_rounds)
    result["per_category_reported_cost"] = per_category_reported_cost
    return result
