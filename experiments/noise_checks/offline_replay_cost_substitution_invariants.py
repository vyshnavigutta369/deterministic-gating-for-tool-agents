"""
Confound-free, zero-additional-API-cost re-verification of the
cost-substitution invariants, using the REAL proposal sequences already
recorded by Phase 1's v3_no_cost_sub condition (parsed straight from the
raw sweep output already on disk -- genuine gpt-4o-mini output across all
60 episodes, 3 pairings x 20 seeds, not synthetic/hand-crafted data).

WHY this is a different (and structurally decisive) test from the live
n=20 comparison: in the live sweep, v3_no_cost_sub and v3_cost_aware are
TWO SEPARATE real API episodes -- two independent series of LLM calls.
Even at temperature=0, OpenAI's API is not perfectly deterministic across
separate calls (well-documented: floating-point non-associativity in
batched/MoE inference), so the two live episodes can genuinely propose
different things starting from an identical prompt, for reasons that have
nothing to do with cost_substitution. That confound makes the live
round-count/cost invariant checks noisy evidence about the FEATURE itself.

This script removes that confound entirely: it takes ONE fixed, already-
observed real proposal sequence per episode and replays it through BOTH
cost_substitution=False and cost_substitution=True locally (no LLM calls),
using the exact same validate_batch_v3 / apply_cost_substitution /
apply_batch functions run_episode itself uses. Since the input sequence is
identical in both branches by construction, this isolates the real
question: does turning on cost_substitution, applied to a FIXED real
trajectory, ever change the resulting state or increase cost at any step?
"""
import re
import sys
sys.path.insert(0, '.')
from env.domains.travel.tool_registry import tools_ready_in_memory
from env.domains.travel.enums import ENUM_MAPPINGS
from env.core.data_types import get_final_type
from env.utils.solver import GroundtruthSolver
from brts_core import validate_batch_v3, apply_cost_substitution, apply_batch, get_currently_executable

# Development note: original sweep-run stdout capture, ephemeral session
# storage since rotated out. Point at your own saved sweep-run log (same
# format) to rerun the replay.
LOG_PATH = "<path-to-saved-sweep-log>/claude_haiku_run.output"

PAIRINGS = [
    ("location", ["location", "transportation"], "location+transportation"),
    ("location", ["location", "accommodation"], "location+accommodation"),
    ("travel", ["location", "transportation", "dining"], "location+transportation+dining"),
]

pairing_header_re = re.compile(r"^##########\s+(.+?)\s+##########$")
seed_header_re = re.compile(r"^=== V3_NO_COST_SUB, seed=(\d+) ===$")
round_re = re.compile(r"^\s*round \d+: proposed=(\[.*?\]) valid=")


def parse_episodes(path):
    """Returns {pairing_label: {seed: [proposed_list, proposed_list, ...]}}"""
    episodes = {}
    current_pairing = None
    current_seed = None
    with open(path) as f:
        for line in f:
            m = pairing_header_re.match(line.strip())
            if m:
                current_pairing = m.group(1)
                episodes.setdefault(current_pairing, {})
                current_seed = None
                continue
            m = seed_header_re.match(line.strip())
            if m:
                current_seed = int(m.group(1))
                episodes[current_pairing][current_seed] = []
                continue
            m = round_re.match(line)
            if m and current_pairing is not None and current_seed is not None:
                proposed = eval(m.group(1))  # trusted, locally-generated log file -- literal Python list syntax
                episodes[current_pairing][current_seed].append(proposed)
    return episodes


def replay(task, categories, seed, proposed_sequence):
    data = tools_ready_in_memory(refinement_level=5, min_atomic_cost=19, max_atomic_cost=21,
                                  noise_std=0.1, random_seed=seed)
    solver = GroundtruthSolver(data['tools'])
    tools = solver.tools
    initial = {"TimeInfo"}
    for cat in categories:
        for enum_class in ENUM_MAPPINGS[cat]["search"]:
            initial.add(enum_class.__name__)
    state_no_sub = frozenset(initial)
    state_sub = frozenset(initial)
    cost_no_sub = 0.0
    cost_sub = 0.0
    total_subs = 0

    for proposed in proposed_sequence:
        # Branch A: no substitution
        valid_a, _ = validate_batch_v3(proposed, state_no_sub, tools)
        state_no_sub = apply_batch(state_no_sub, tools, valid_a)
        cost_no_sub += sum(tools[n].cost for n in valid_a)

        # Branch B: cost substitution on, against its OWN independently-tracked state
        # (must be its own state, not state_no_sub, since a real live run would
        # feed cost-substitution's own evolving state back in -- but since both
        # states are proven output-type-identical every step, this is equivalent)
        valid_b_gate, _ = validate_batch_v3(proposed, state_sub, tools)
        visible_b = [n for n in get_currently_executable(state_sub, tools) if tools[n].output_type not in state_sub]
        valid_b, subs = apply_cost_substitution(valid_b_gate, state_sub, tools, visible_b)
        total_subs += len(subs)
        state_sub = apply_batch(state_sub, tools, valid_b)
        cost_sub += sum(tools[n].cost for n in valid_b)

        if state_no_sub != state_sub:
            return None, f"STATE DIVERGED mid-replay: no_sub={sorted(state_no_sub)} vs sub={sorted(state_sub)}"

    return {
        "rounds": len(proposed_sequence), "cost_no_sub": cost_no_sub, "cost_sub": cost_sub,
        "total_subs": total_subs, "state_match": state_no_sub == state_sub,
    }, None


if __name__ == "__main__":
    episodes = parse_episodes(LOG_PATH)
    print(f"Parsed {sum(len(v) for v in episodes.values())} real recorded episodes "
          f"across {len(episodes)} pairings from {LOG_PATH}\n")

    total_checked = 0
    violations = []
    for task, categories, label in PAIRINGS:
        if label not in episodes:
            print(f"WARNING: no parsed episodes for {label} -- skipping")
            continue
        pairing_cost_no_sub = 0.0
        pairing_cost_sub = 0.0
        pairing_subs = 0
        for seed, proposed_sequence in sorted(episodes[label].items()):
            total_checked += 1
            result, error = replay(task, categories, seed, proposed_sequence)
            if error:
                violations.append(f"{label} seed={seed}: {error}")
                continue
            if result["cost_sub"] > result["cost_no_sub"] + 1e-9:
                violations.append(f"{label} seed={seed}: cost_sub ({result['cost_sub']:.2f}) > "
                                   f"cost_no_sub ({result['cost_no_sub']:.2f})")
            pairing_cost_no_sub += result["cost_no_sub"]
            pairing_cost_sub += result["cost_sub"]
            pairing_subs += result["total_subs"]
        print(f"{label}: {len(episodes[label])} real episodes replayed offline -- "
              f"total cost_no_sub={pairing_cost_no_sub:.2f}, total cost_sub={pairing_cost_sub:.2f} "
              f"({100*(1 - pairing_cost_sub/pairing_cost_no_sub):.1f}% reduction), "
              f"{pairing_subs} real substitutions found")

    print(f"\n=== OFFLINE REPLAY RESULT: {total_checked} real recorded episodes checked ===")
    if violations:
        print(f"{len(violations)} VIOLATION(S) FOUND:")
        for v in violations:
            print(f"  - {v}")
    else:
        print("ZERO violations -- state matched at every single round, and cost_sub never exceeded "
              "cost_no_sub, on every one of the real recorded trajectories. Confirms apply_cost_substitution "
              f"itself is safe against real proposal data from {LOG_PATH}, isolated from cross-call LLM stochasticity.")
