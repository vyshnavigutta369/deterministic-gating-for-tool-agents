"""
Confound-free, zero-additional-API-cost re-verification of the
redundancy-elimination invariants, same discipline established for
cost-substitution: the live n=5 comparison showed "violations" (round
count/cost differing between conditions on 3/5 and 2/5 seeds), but
v3_no_redundancy_elim and v3_redundancy_elim are TWO SEPARATE live API
episodes -- cross-call temp=0 stochasticity alone can (and, per the
cost-substitution investigation, reliably does) cause two "identical"
episodes to diverge for reasons having nothing to do with the mechanism
under test. This replays the REAL recorded proposal sequences from the
v3_no_redundancy_elim condition (ground truth, already collected, zero
new API calls) through both redundancy_elimination=False/True locally,
isolating the real question: applied to a FIXED real trajectory, does
redundancy elimination ever change the resulting state, or increase cost?
"""
import re
import sys
sys.path.insert(0, '.')
from env.domains.travel.tool_registry import tools_ready_in_memory
from env.domains.travel.enums import ENUM_MAPPINGS
from env.utils.solver import GroundtruthSolver
from brts_core import validate_batch_v3, apply_redundancy_elimination, apply_batch

# Development note: original Stage 3 stdout capture, saved to a scratch
# file during investigation and not included in this release. Point at
# your own saved run output (same format) to rerun the replay.
LOG_PATH = "<path-to-saved-run-log>/redundancy_elim_stage3_log.txt"

TASK, CATEGORIES = "location", ["location", "transportation"]

any_header_re = re.compile(r"^=== .+ ===$")
round_re = re.compile(r"^\s*round \d+: proposed=(\[.*?\]) valid=")


def parse_episodes(path, condition_label):
    """Only captures rounds while inside a section matching condition_label
    -- must reset current_seed on ANY other header (e.g. the interleaved
    other condition for the same seed number), not just on the next
    matching header, or rounds from the wrong condition bleed in."""
    seed_header_re = re.compile(rf"^=== {condition_label}, seed=(\d+) ===$")
    episodes = {}
    current_seed = None
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            m = seed_header_re.match(stripped)
            if m:
                current_seed = int(m.group(1))
                episodes[current_seed] = []
                continue
            if any_header_re.match(stripped):
                current_seed = None  # left the matching-condition section
                continue
            m = round_re.match(line)
            if m and current_seed is not None:
                proposed = eval(m.group(1))  # trusted, locally-generated log file
                episodes[current_seed].append(proposed)
    return episodes


def replay(seed, proposed_sequence):
    data = tools_ready_in_memory(refinement_level=5, min_atomic_cost=19, max_atomic_cost=21,
                                  noise_std=0.1, random_seed=seed)
    solver = GroundtruthSolver(data['tools'])
    tools = solver.tools
    initial = {"TimeInfo"}
    for cat in CATEGORIES:
        for enum_class in ENUM_MAPPINGS[cat]["search"]:
            initial.add(enum_class.__name__)
    state_off = frozenset(initial)
    state_on = frozenset(initial)
    cost_off = 0.0
    cost_on = 0.0
    total_dropped = 0

    for proposed in proposed_sequence:
        valid_off, _ = validate_batch_v3(proposed, state_off, tools)
        state_off = apply_batch(state_off, tools, valid_off)
        cost_off += sum(tools[n].cost for n in valid_off)

        valid_on_gate, _ = validate_batch_v3(proposed, state_on, tools)
        valid_on, dropped = apply_redundancy_elimination(valid_on_gate, tools)
        total_dropped += len(dropped)
        state_on = apply_batch(state_on, tools, valid_on)
        cost_on += sum(tools[n].cost for n in valid_on)

        if state_off != state_on:
            return None, f"STATE DIVERGED mid-replay: off={sorted(state_off)} vs on={sorted(state_on)}"

    return {"rounds": len(proposed_sequence), "cost_off": cost_off, "cost_on": cost_on,
            "total_dropped": total_dropped}, None


def run_for_condition(condition_label):
    episodes = parse_episodes(LOG_PATH, condition_label)
    print(f"=== Ground truth: {condition_label} ({len(episodes)} real recorded episodes) ===")

    violations = []
    total_cost_off = 0.0
    total_cost_on = 0.0
    total_dropped = 0
    for seed, proposed_sequence in sorted(episodes.items()):
        result, error = replay(seed, proposed_sequence)
        if error:
            violations.append(f"seed={seed}: {error}")
            continue
        if result["cost_on"] > result["cost_off"] + 1e-9:
            violations.append(f"seed={seed}: cost_on ({result['cost_on']:.2f}) > cost_off ({result['cost_off']:.2f})")
        print(f"seed={seed}: rounds={result['rounds']} (identical both branches by construction), "
              f"cost_off={result['cost_off']:.2f}, cost_on={result['cost_on']:.2f}, "
              f"dropped={result['total_dropped']}")
        total_cost_off += result["cost_off"]
        total_cost_on += result["cost_on"]
        total_dropped += result["total_dropped"]

    print(f"\n--- RESULT ({condition_label}): {len(episodes)} real recorded episodes checked ---")
    if violations:
        print(f"{len(violations)} VIOLATION(S):")
        for v in violations:
            print(f"  - {v}")
    else:
        pct = 100 * (1 - total_cost_on / total_cost_off) if total_cost_off else 0
        print(f"ZERO violations -- state matched at every single round, cost_on never exceeded cost_off, "
              f"on every one of the real recorded trajectories. Total: cost_off={total_cost_off:.2f}, "
              f"cost_on={total_cost_on:.2f} ({pct:.1f}% reduction), {total_dropped} real redundant calls dropped.")
    print()
    return violations


if __name__ == "__main__":
    v1 = run_for_condition("V3_NO_REDUNDANCY_ELIM")
    v2 = run_for_condition("V3_REDUNDANCY_ELIM")
    total_violations = len(v1) + len(v2)
    print(f"=== COMBINED: {total_violations} total violations across both ground-truth batches (10 real episodes) ===")
