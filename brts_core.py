"""
BRTS core: independence-validation logic for batching tool calls, implemented
and tested against REAL CostBench tool registries and REAL cost data.

HONEST SCOPE NOTE: no LLM API key is available in this environment, so this
does NOT run a real LLM end-to-end. What IS real: the tool data, the cost
model, the typed independence-checking logic, and the round-count comparison.
The "batched" policy below is a MOCK standing in for an LLM that proposes the
best possible batch every round -- i.e. this measures the MECHANICAL UPPER
BOUND of what batching could achieve, not what a real LLM will actually
achieve (real LLMs will likely do worse than this upper bound, per MCP-Bench's
real finding that models under-batch even when explicitly instructed to).
This is stated here, in the code, and must be stated identically in any
write-up -- this is an upper-bound simulation, not an LLM evaluation.
"""
import sys
import copy
from typing import Dict, List, Set, FrozenSet

sys.path.insert(0, '.')
from env.domains.travel.tool_registry import tools_ready_in_memory
from env.utils.solver import GroundtruthSolver


def get_currently_executable(state: FrozenSet[str], tools: Dict) -> List[str]:
    """All tools whose inputs are satisfied by the CURRENT state alone --
    this is the real independence check: if a tool's inputs are already
    satisfied without needing any other proposed tool's output this round,
    it's safe to batch. Mirrors CostBench's own GroundtruthSolver._get_successors
    logic (verified real, Appendix F), reimplemented directly here since we
    are building our own harness layer, not modifying their solver in place.
    """
    executable = []
    for name, tool in tools.items():
        if set(tool.input_types).issubset(state):
            executable.append(name)
    return executable


def validate_batch(proposed_names: List[str], state: FrozenSet[str], tools: Dict):
    """Given a batch an agent PROPOSES, split into what's genuinely safe to
    execute this round (inputs satisfied by current state alone) vs what must
    be rejected (depends on another call's output, or invalid). This is the
    real correctness gate BRTS needs: even if the LLM proposes something
    unsafe, we never silently execute a call whose dependency isn't met."""
    valid, rejected = [], []
    for name in proposed_names:
        tool = tools.get(name)
        if tool is not None and set(tool.input_types).issubset(state):
            valid.append(name)
        else:
            rejected.append(name)
    return valid, rejected


def validate_batch_v2(proposed_names: List[str], state: FrozenSet[str], tools: Dict):
    """
    Gate v2: chain-aware batch resolution. Does NOT modify or replace
    validate_batch (v1) -- an additional, separate gate function, run
    side-by-side via run_episode's gate_version flag.

    validate_batch (v1) only accepts a proposed tool if ALL its inputs are
    satisfied by the PRE-ROUND state alone -- a tool that genuinely depends
    on another proposed tool's output within the SAME round is always
    rejected by v1 and pushed to a later round, even when the dependency
    is real and fully resolvable within the batch itself. v2 additionally
    allows a tool's inputs to be satisfied by another ACCEPTED tool earlier
    in a resolved order within the same round, using each tool's
    already-declared input_types/output_type (the same real Tool
    attributes validate_batch/apply_batch already rely on -- output_type
    is a required, validated field on every real Tool object, confirmed in
    env/utils/solver.py, not invented for this).

    Builds a dependency graph over the proposed batch and resolves it via
    repeated relaxation (a topological sort, equivalent in effect to
    Kahn's algorithm): on each pass, accept any not-yet-accepted tool
    whose still-unmet inputs are now covered by the pre-round state plus
    the output types of tools already accepted in an earlier pass. Repeat
    until a pass makes no further progress. Anything left over is
    rejected -- either a genuinely missing dependency (not provided by the
    state or by anything else in the batch) or part of a real cycle (A
    needs B's output and B needs A's output, directly or transitively).

    Returns (valid_ordered, rejected): valid_ordered is the accepted tools
    in a REAL, dependency-respecting execution order (unlike v1's valid
    list, whose order never matters since every entry already
    independently satisfies the pre-round state alone). Every tool in
    valid_ordered is guaranteed to have its real inputs satisfied at the
    exact point it appears in that order -- v2 never accepts a tool whose
    inputs are not actually available when it would run, it just allows
    "available" to include earlier same-round outputs, not only the
    original pre-round state.
    """
    unresolved = {}
    for name in proposed_names:
        tool = tools.get(name)
        if tool is None:
            continue  # hallucinated/nonexistent name -- can never resolve, falls into rejected below
        unresolved[name] = set(tool.input_types) - state

    resolved_order = []
    available_types = set(state)
    progress = True
    while unresolved and progress:
        progress = False
        for name in list(unresolved):
            if unresolved[name].issubset(available_types):
                resolved_order.append(name)
                available_types.add(tools[name].output_type)
                del unresolved[name]
                progress = True

    rejected = [n for n in proposed_names if n not in resolved_order]
    return resolved_order, rejected


def validate_batch_v3(proposed_names: List[str], state: FrozenSet[str], tools: Dict):
    """
    Gate v3: a unified gate, not a second patch stacked on v2. Combines two
    capabilities that were previously separate mechanisms into one:

    1. CHAIN-AWARE RESOLUTION -- the same topological-sort algorithm as
       validate_batch_v2 (unchanged, does not modify or replace v1 or v2 --
       both remain available side-by-side via run_episode's gate_version
       flag). A tool's inputs may be satisfied by the pre-round state OR by
       another accepted tool earlier in a resolved same-round order.

    2. UNIVERSAL REJECTION REASONING -- for EVERY tool the gate rejects,
       not only in a full-rejection round but in a partial one too,
       structured reasoning is a core part of what the gate returns, not a
       separate feedback-builder called after the fact. v1 and v2 both
       return `rejected` as a flat list of bare names, discarding WHY each
       one failed; the caller has to reconstruct that separately (which is
       exactly what build_full_rejection_feedback had to do as a bolted-on
       patch). v3 returns that reasoning directly, because "was this
       genuinely resolvable" and "why wasn't it" are one question asked
       against the same dependency graph, not two.

    Returns (valid_ordered, rejected): valid_ordered is identical in
    meaning to v2's (a real, dependency-respecting execution order --
    every tool in it has its actual inputs satisfied at the exact point it
    appears). `rejected` is a list of dicts, one per rejected name, each
    with:
      - "name": the proposed name
      - "reason": "nonexistent_tool" (no tool with that exact name exists
        at all -- e.g. the real, repeatedly-observed
        Location_Finish_from_Step3 hallucination) or "missing_precondition"
        (a real tool, but at least one required input was never satisfied
        by the pre-round state or by anything else in this batch, even
        after considering every possible same-round chain)
      - "missing_inputs": for "missing_precondition", the specific input
        type(s) still unmet (computed against the FINAL settled available-
        types set, after chain resolution -- so a tool that needed two
        inputs and got one of them from another accepted tool in this same
        batch is reported as missing only the one that's genuinely still
        absent, not the original full requirement); None for
        "nonexistent_tool", since there is no precondition to describe for
        a tool that was never real.
    """
    unresolved = {}
    nonexistent = set()
    for name in proposed_names:
        tool = tools.get(name)
        if tool is None:
            nonexistent.add(name)
            continue
        unresolved[name] = set(tool.input_types) - state

    resolved_order = []
    available_types = set(state)
    progress = True
    while unresolved and progress:
        progress = False
        for name in list(unresolved):
            if unresolved[name].issubset(available_types):
                resolved_order.append(name)
                available_types.add(tools[name].output_type)
                del unresolved[name]
                progress = True

    rejected = []
    for name in proposed_names:
        if name in resolved_order:
            continue
        if name in nonexistent:
            rejected.append({"name": name, "reason": "nonexistent_tool", "missing_inputs": None})
        else:
            still_missing = sorted(unresolved[name] - available_types)
            rejected.append({"name": name, "reason": "missing_precondition", "missing_inputs": still_missing})

    return resolved_order, rejected


def apply_redundancy_elimination(valid_ordered: List[str], tools: Dict):
    """
    Redundancy-elimination pass: an ADDITIONAL capability, run strictly
    AFTER a gate (v1/v2/v3, unmodified) has already decided what's safe to
    execute this round. Not a gate -- never rejects a tool for being
    unsafe, only drops a tool that is genuinely REDUNDANT within THIS
    batch: when 2+ accepted slots target the identical output_type, only
    the cheapest survives; the rest are dropped before execution.

    Found live (see Stage 4's seed=2/round=7 log): a real gate-accepted
    batch containing 4 different tools that all produce TravelTransportation
    (Transportation_Finish_from_Step{1..4}_*Steps, costs 123.31/103.56/
    82.74/62.26) -- all 4 executed, when only the cheapest was needed for
    the state to gain TravelTransportation at all. Unlike cost-substitution
    (which swaps one slot for an equal-output cheaper alternative, batch
    size unchanged), this REMOVES excess slots outright, since they'd
    produce output the batch already has from elsewhere in the SAME round.

    Safety property, the reason this can never change downstream
    correctness: apply_batch computes new_types as a SET of output_type
    over the batch -- {tools[n].output_type for n in batch}. Dropping every
    same-output duplicate except the cheapest never removes an output_type
    from that set (one survivor per type is always kept), so the resulting
    state is IDENTICAL whether duplicates are dropped or not. Only total
    cost can go down (or stay the same when there's nothing to drop).

    Returns (deduped_ordered, dropped): deduped_ordered is valid_ordered
    with 0+ entries removed (relative order of survivors preserved, not
    reordered); dropped is a list of {"name", "output_type", "kept",
    "cost_saved"} dicts, one per removed slot (empty if nothing was
    redundant -- a true no-op in that case).
    """
    by_output = {}
    for idx, name in enumerate(valid_ordered):
        by_output.setdefault(tools[name].output_type, []).append(idx)

    keep_indices = set()
    dropped = []
    for output_type, idxs in by_output.items():
        if len(idxs) == 1:
            keep_indices.add(idxs[0])
            continue
        cheapest_idx = min(idxs, key=lambda i: tools[valid_ordered[i]].cost)
        keep_indices.add(cheapest_idx)
        for i in idxs:
            if i != cheapest_idx:
                dropped.append({"name": valid_ordered[i], "output_type": output_type,
                                 "kept": valid_ordered[cheapest_idx],
                                 "cost_saved": tools[valid_ordered[i]].cost})

    deduped_ordered = [valid_ordered[i] for i in range(len(valid_ordered)) if i in keep_indices]
    return deduped_ordered, dropped


def apply_cost_substitution(valid_ordered: List[str], state: FrozenSet[str], tools: Dict,
                             visible_tools: List[str]):
    """
    Cost-substitution pass: an ADDITIONAL capability, run strictly AFTER a
    gate (v1/v2/v3, unmodified) has already decided which proposed tools are
    safe to execute this round and in what order. Not itself a gate --
    named apply_ rather than validate_ since it never rejects anything, only
    swaps which concrete tool fills an already-accepted slot.

    For each accepted slot, in order: look at every OTHER tool in
    visible_tools (the same "currently executable, not yet in state"
    universe already shown to the model this round -- see
    get_currently_executable in run_brts_minimal.py) that produces the
    IDENTICAL output_type and whose own inputs are satisfiable at this exact
    point in the batch (pre-round state plus any earlier accepted slot's
    output, mirroring v2/v3's own available_types bookkeeping). Among the
    original tool and all such candidates, keep the cheapest. If it's not
    the original, substitute it in -- same position, same execution order.

    Because a substitute is only ever chosen from tools sharing the exact
    output_type of the slot it replaces, apply_batch's new_types computation
    is IDENTICAL whether or not a substitution happened -- this guarantees,
    by construction, that this pass can never change len(valid_ordered),
    execution order, or the resulting state/round count. It can only ever
    lower (or leave unchanged) total cost.

    Returns (substituted_ordered, substitutions): substituted_ordered is
    valid_ordered with 0+ names replaced (same length/order); substitutions
    is a list of {"slot", "replaced_with", "output_type", "cost_saved"}
    dicts, one per real substitution made (empty if none were possible --
    a true no-op in that case, both in output and in state).
    """
    available_types = set(state)
    substituted = []
    substitutions = []
    for name in valid_ordered:
        original_tool = tools[name]
        output_type = original_tool.output_type
        best_name = name
        best_cost = original_tool.cost
        for other_name in visible_tools:
            if other_name == name:
                continue
            other_tool = tools.get(other_name)
            if other_tool is None or other_tool.output_type != output_type:
                continue
            if not set(other_tool.input_types).issubset(available_types):
                continue
            if other_tool.cost < best_cost:
                best_name, best_cost = other_name, other_tool.cost
        if best_name != name:
            substitutions.append({"slot": name, "replaced_with": best_name,
                                   "output_type": output_type,
                                   "cost_saved": original_tool.cost - best_cost})
        substituted.append(best_name)
        available_types.add(output_type)
    return substituted, substitutions


def apply_batch(state: FrozenSet[str], tools: Dict, batch: List[str]) -> FrozenSet[str]:
    new_types = {tools[name].output_type for name in batch}
    return frozenset(state | new_types)


def apply_batch_blind(state: FrozenSet[str], tools: Dict, proposed_names: List[str]):
    """Execute every proposed name that corresponds to a REAL tool, with NO
    check on whether its input preconditions are actually satisfied by the
    current state -- this is what skipping validate_batch as a safety GATE
    means (used by the "blind batching" and "self-judged batching" baseline
    conditions, which mimic prior work -- W&D and MCP-Bench respectively --
    that executes a proposed batch without an external independence check).
    A nonexistent/hallucinated name still can't be executed at all (that's
    basic name resolution, not a safety check being skipped), so it's
    dropped from what actually runs; any REAL tool name executes and
    accrues cost regardless of whether its dependencies were genuinely
    ready yet, which is exactly the risk this baseline exists to expose.

    Returns (new_state, executed_names, cost) -- executed_names is the
    subset of proposed_names that were real tools and therefore ran."""
    executed = [name for name in proposed_names if name in tools]
    new_state = apply_batch(state, tools, executed)
    cost = sum(tools[n].cost for n in executed)
    return new_state, executed, cost


def is_goal_reached(state: FrozenSet[str], target_types: Set[str]) -> bool:
    return target_types.issubset(state)


def run_sequential_baseline(tools: Dict, initial_state: FrozenSet[str], target_types: Set[str],
                             max_rounds: int = 50):
    """Real baseline matching CostBench's ACTUAL, verified rule: exactly one
    tool call per round (env/run.py line ~413, `tool_calls_list[0]`)."""
    state = initial_state
    rounds = 0
    total_cost = 0.0
    while not is_goal_reached(state, target_types) and rounds < max_rounds:
        executable = get_currently_executable(state, tools)
        if not executable:
            break
        # Pick the cheapest currently-executable tool -- a reasonable,
        # simple stand-in policy for "which one call would a sensible agent
        # make," since we're isolating the ROUND-COUNT effect of batching,
        # not modeling full LLM tool-selection quality here.
        choice = min(executable, key=lambda n: tools[n].cost)
        state = apply_batch(state, tools, [choice])
        total_cost += tools[choice].cost
        rounds += 1
    return rounds, total_cost, is_goal_reached(state, target_types)


def _category_of(output_type: str) -> str:
    """Real categories aren't an explicit field on Tool objects (verified --
    only name/input_types/output_type/cost/description exist), but they're
    embedded in the output_type string prefix (e.g., 'LocationPreference',
    'LocationCandidate_L1_Available' both start with 'Location'). Extracting
    the leading capitalized word as a category label -- an honest heuristic,
    not a guaranteed-correct parser, but checkable against the visible type
    names we've verified."""
    import re
    match = re.match(r'^([A-Z][a-z]+)', output_type)
    return match.group(1) if match else output_type


def run_batched_upper_bound(tools: Dict, initial_state: FrozenSet[str], target_types: Set[str],
                             max_rounds: int = 50):
    """MOCK upper-bound policy, CORRECTED: batches at most ONE tool per
    category per round (not literally everything executable). The first
    version of this function batched every currently-executable tool,
    including mutually-redundant alternatives for the SAME sub-goal (e.g.
    several different tools that each individually satisfy the same step) --
    that produced a nonsensical, inflated result (found by actually running
    it and noticing the round count was implausibly low; see the design doc's
    record of this). Real, legitimate batching is across INDEPENDENT
    categories (Location progressing alongside Transportation), not across
    redundant alternatives within the same category."""
    state = initial_state
    rounds = 0
    total_cost = 0.0
    while not is_goal_reached(state, target_types) and rounds < max_rounds:
        executable = get_currently_executable(state, tools)
        if not executable:
            break
        # Group by category, pick the cheapest option within each category
        # (a reasonable stand-in decision rule), batch one per category.
        by_category: Dict[str, List[str]] = {}
        for name in executable:
            cat = _category_of(tools[name].output_type)
            by_category.setdefault(cat, []).append(name)
        batch = [min(names, key=lambda n: tools[n].cost) for names in by_category.values()]

        valid, rejected = validate_batch(batch, state, tools)
        assert not rejected, "Should never reject here -- each was independently state-satisfied"
        state = apply_batch(state, tools, valid)
        total_cost += sum(tools[n].cost for n in valid)
        rounds += 1
    return rounds, total_cost, is_goal_reached(state, target_types)


def run_comparison(seed: int):
    data = tools_ready_in_memory(refinement_level=5, min_atomic_cost=19, max_atomic_cost=21,
                                  noise_std=0.1, random_seed=seed)
    raw_tools = data['tools']
    solver = GroundtruthSolver(raw_tools)  # converts raw dicts into real Tool objects with attributes
    tools = solver.tools  # use the converted objects everywhere, matching what worked in lpastar_planner.py
    initial_state = solver._infer_initial_state('location')
    target_types = solver._infer_target_types('location')

    seq_rounds, seq_cost, seq_ok = run_sequential_baseline(tools, initial_state, target_types)
    batch_rounds, batch_cost, batch_ok = run_batched_upper_bound(tools, initial_state, target_types)

    return {
        "seed": seed,
        "sequential_rounds": seq_rounds, "sequential_cost": seq_cost, "sequential_reached_goal": seq_ok,
        "batched_rounds": batch_rounds, "batched_cost": batch_cost, "batched_reached_goal": batch_ok,
        "round_reduction_pct": 100.0 * (1 - batch_rounds / seq_rounds) if seq_rounds else 0.0,
    }


def compress_real_gt_path_into_rounds(gt_path: List[str], initial_state: FrozenSet[str], tools: Dict):
    """
    Uses the REAL, already-verified-correct GT path (from GroundtruthSolver.solve,
    not a made-up policy) and asks an honest question: using our REAL,
    working validate_batch/get_currently_executable logic, how many rounds
    would this exact, correct sequence actually require if we greedily batch
    together any consecutive steps that don't truly depend on each other?

    This replaces the earlier broken mock policies (pick-cheapest / batch-
    everything), which had no relationship to a real correct solution and
    produced nonsensical results. This version can never invent progress the
    real solver didn't already validate -- it only asks whether the REAL
    path's steps could have been grouped into fewer rounds.
    """
    state = initial_state
    remaining = list(gt_path)
    rounds = []
    while remaining:
        # Take the longest prefix of the remaining REAL path whose inputs are
        # ALL already satisfied by the state as of the START of this round
        # (not by each other) -- this is exactly what real batching requires.
        batch = []
        i = 0
        while i < len(remaining):
            name = remaining[i]
            if set(tools[name].input_types).issubset(state):
                batch.append(name)
                i += 1
            else:
                break  # this step needs something not yet available; stop growing this round
        if not batch:
            # Shouldn't happen for a genuinely valid GT path, but fail loudly
            # rather than silently if it does -- another honest correctness check.
            raise RuntimeError(f"GT path step '{remaining[0]}' has unsatisfied inputs even alone -- "
                               f"the path itself may be invalid, or our state tracking has a bug.")
        state = apply_batch(state, tools, batch)
        rounds.append(batch)
        remaining = remaining[len(batch):]
    return rounds


def run_real_gt_comparison(seed: int):
    """Honest comparison: real GT path length (= CostBench's actual real
    sequential-round count under the one-tool-per-step rule) vs. the number
    of rounds our real validate_batch logic can compress that SAME real path
    into. No mock agent policy involved on either side."""
    data = tools_ready_in_memory(refinement_level=5, min_atomic_cost=19, max_atomic_cost=21,
                                  noise_std=0.1, random_seed=seed)
    solver = GroundtruthSolver(data['tools'])
    tools = solver.tools
    initial_state = solver._infer_initial_state('location')

    gt_result = solver.solve(task='location')
    gt_path = gt_result.tools  # the REAL, verified-correct sequence

    rounds = compress_real_gt_path_into_rounds(gt_path, initial_state, tools)

    return {
        "seed": seed,
        "gt_path": gt_path,
        "gt_path_length": len(gt_path),          # = real sequential round count (one-tool-per-step)
        "compressed_round_count": len(rounds),    # = rounds needed if genuinely-independent steps are batched
        "rounds": rounds,
        "reduction_pct": 100.0 * (1 - len(rounds) / len(gt_path)) if gt_path else 0.0,
    }


def compress_tool_set_into_rounds(remaining_tools: List[str], initial_state: FrozenSet[str], tools: Dict):
    """
    Order-independent version of compress_real_gt_path_into_rounds: instead
    of only batching a PREFIX of a fixed concatenation order (which
    understates real achievable compression -- an artifact of merge order,
    not a real limit), this looks across ALL remaining known-necessary tools
    each round and takes everything currently satisfiable. Restricted to a
    known, real, finite set of tools (the ones actually used in the real
    per-category GT paths) rather than the full registry, so it can't
    reproduce the earlier redundant-alternative-explosion bug -- these are
    already-known-necessary tools, not arbitrary alternatives.
    """
    state = initial_state
    remaining = set(remaining_tools)
    rounds = []
    while remaining:
        batch = [name for name in remaining if set(tools[name].input_types).issubset(state)]
        if not batch:
            raise RuntimeError(f"Stuck: no remaining tool is currently executable. Remaining: {remaining}")
        state = apply_batch(state, tools, batch)
        rounds.append(sorted(batch))
        remaining -= set(batch)
    return rounds


def run_multi_category_comparison(seed: int, categories=('location', 'transportation')):
    """
    The REAL test of BRTS's actual value: does batching help when a task
    genuinely spans multiple categories (which real CostBench queries do,
    per the QUERY_INSTRUCTION prompt's "{tool_num_str} parts" phrasing),
    rather than the single-category case already shown to have 0% benefit.

    Construction, stated honestly: solver._infer_initial_state's per-task
    convenience method assumes LocationPreference already exists for non-
    location tasks (verified in source) -- that's an artifact of solving
    ONE category at a time, not a genuine multi-goal query where the user
    specifies both categories' preferences upfront. To test genuine
    simultaneous independence, we override the initial state to be the
    union of TimeInfo + each category's own raw enum requirements (what a
    user would actually provide upfront for a combined query), then solve
    each category's real GT path against that shared combined state.
    """
    data = tools_ready_in_memory(refinement_level=5, min_atomic_cost=19, max_atomic_cost=21,
                                  noise_std=0.1, random_seed=seed)
    solver = GroundtruthSolver(data['tools'])
    tools = solver.tools

    from env.domains.travel.enums import ENUM_MAPPINGS
    combined_initial = {"TimeInfo"}
    for cat in categories:
        for enum_class in ENUM_MAPPINGS[cat]["search"]:
            combined_initial.add(enum_class.__name__)
    combined_initial_state = frozenset(combined_initial)

    per_category_paths = {}
    target_types = set()
    for cat in categories:
        # Solve each category's real optimal path, seeded from the SHARED
        # combined initial state (a superset of what solve() would normally
        # assume) -- Dijkstra correctness is unaffected by extra available
        # types, so this still finds each category's real optimal chain.
        solver._infer_initial_state = lambda task: combined_initial_state
        result = solver.solve(task=cat)
        per_category_paths[cat] = result.tools
        target_types.add(get_final_type_local(cat))

    merged_path = []
    for cat in categories:
        merged_path.extend(per_category_paths[cat])

    naive_sequential_rounds = len(merged_path)  # what CostBench's real one-tool-per-step rule requires

    rounds_naive_order = compress_real_gt_path_into_rounds(merged_path, combined_initial_state, tools)
    rounds_best_order = compress_tool_set_into_rounds(merged_path, combined_initial_state, tools)

    return {
        "seed": seed,
        "categories": categories,
        "per_category_path_lengths": {c: len(p) for c, p in per_category_paths.items()},
        "naive_sequential_rounds": naive_sequential_rounds,
        "compressed_rounds_fixed_order": len(rounds_naive_order),
        "compressed_rounds_best_order": len(rounds_best_order),
        "reduction_pct_fixed_order": 100.0 * (1 - len(rounds_naive_order) / naive_sequential_rounds) if naive_sequential_rounds else 0.0,
        "reduction_pct_best_order": 100.0 * (1 - len(rounds_best_order) / naive_sequential_rounds) if naive_sequential_rounds else 0.0,
        "rounds_best_order": rounds_best_order,
    }


def get_final_type_local(subtask: str) -> str:
    final_types = {
        "location": "TravelLocation", "transportation": "TravelTransportation",
        "accommodation": "TravelAccommodation", "attraction": "TravelAttraction",
        "dining": "TravelDining", "shopping": "TravelShopping",
    }
    return final_types.get(subtask.lower(), "")


if __name__ == "__main__":
    print("Single-category result (already run, kept for reference): 0.0% reduction across")
    print("10 seeds -- expected, since a refinement chain has no independent structure.\n")
    print("Now the real test: multi-category (location + transportation), where genuine")
    print("cross-category independence should exist. Reporting BOTH a naive fixed-merge-order")
    print("compression AND an order-independent best-case compression, since the fixed-order")
    print("version could understate real achievable savings as a merge-order artifact.\n")
    print(f"{'seed':>5} {'naive':>7} {'fixed_ord':>10} {'best_ord':>9} {'fixed%':>8} {'best%':>7}")
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        r = run_multi_category_comparison(seed)
        print(f"{r['seed']:>5} {r['naive_sequential_rounds']:>7} {r['compressed_rounds_fixed_order']:>10} "
              f"{r['compressed_rounds_best_order']:>9} {r['reduction_pct_fixed_order']:>7.1f}% "
              f"{r['reduction_pct_best_order']:>6.1f}%   ({r['per_category_path_lengths']})")
