"""
Confound-free, zero-additional-API-cost re-verification of Stage 4's
composition claim (redundancy_elimination + cost_substitution, dedup
first then substitute). Same discipline as both individual mechanisms:
the live 3-condition comparison showed the usual cross-call-stochasticity
noise between separate live episodes, so this replays each condition's
OWN real recorded proposal sequence locally through
(a) that condition's own real settings and (b) full composed settings,
confirming state is identical and cost never increases -- isolating the
mechanism's real safety claim from the live-comparison confound.
"""
import re
import sys
sys.path.insert(0, '.')
from env.domains.travel.tool_registry import tools_ready_in_memory
from env.domains.travel.enums import ENUM_MAPPINGS
from env.utils.solver import GroundtruthSolver
from brts_core import validate_batch_v3, apply_redundancy_elimination, apply_cost_substitution, apply_batch, get_currently_executable

# Development note: original Stage 4 stdout capture, saved to a scratch
# file during investigation and not included in this release. Point at
# your own saved run output (same format) to rerun the replay.
LOG_PATH = "<path-to-saved-run-log>/redundancy_plus_costsub_stage4_log.txt"
TASK, CATEGORIES = "location", ["location", "transportation"]

any_header_re = re.compile(r"^=== .+ ===$")
round_re = re.compile(r"^\s*round \d+: proposed=(\[.*?\]) valid=")


def parse_episodes(path, condition_label):
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
                current_seed = None
                continue
            m = round_re.match(line)
            if m and current_seed is not None:
                episodes[current_seed].append(eval(m.group(1)))  # trusted, locally-generated log file
    return episodes


def run_branch(proposed_sequence, seed, use_redundancy, use_cost_sub):
    data = tools_ready_in_memory(refinement_level=5, min_atomic_cost=19, max_atomic_cost=21,
                                  noise_std=0.1, random_seed=seed)
    solver = GroundtruthSolver(data['tools'])
    tools = solver.tools
    initial = {"TimeInfo"}
    for cat in CATEGORIES:
        for enum_class in ENUM_MAPPINGS[cat]["search"]:
            initial.add(enum_class.__name__)
    state = frozenset(initial)
    cost = 0.0
    dropped_total = 0
    subs_total = 0

    for proposed in proposed_sequence:
        visible = [n for n in get_currently_executable(state, tools) if tools[n].output_type not in state]
        valid, _ = validate_batch_v3(proposed, state, tools)
        if use_redundancy:
            valid, dropped = apply_redundancy_elimination(valid, tools)
            dropped_total += len(dropped)
        if use_cost_sub:
            valid, subs = apply_cost_substitution(valid, state, tools, visible)
            subs_total += len(subs)
        state = apply_batch(state, tools, valid)
        cost += sum(tools[n].cost for n in valid)

    return state, cost, dropped_total, subs_total


def check(ground_truth_label, use_redundancy_baseline, use_cost_sub_baseline):
    episodes = parse_episodes(LOG_PATH, ground_truth_label)
    print(f"=== Ground truth: {ground_truth_label} ({len(episodes)} real episodes) ===")
    violations = []
    total_cost_baseline = 0.0
    total_cost_composed = 0.0
    for seed, proposed_sequence in sorted(episodes.items()):
        state_baseline, cost_baseline, _, _ = run_branch(
            proposed_sequence, seed, use_redundancy_baseline, use_cost_sub_baseline)
        state_composed, cost_composed, dropped, subs = run_branch(
            proposed_sequence, seed, True, True)  # full composition

        if state_baseline != state_composed:
            violations.append(f"seed={seed}: STATE DIVERGED -- baseline={sorted(state_baseline)} "
                               f"vs composed={sorted(state_composed)}")
        if cost_composed > cost_baseline + 1e-9:
            violations.append(f"seed={seed}: composed cost ({cost_composed:.2f}) > "
                               f"baseline cost ({cost_baseline:.2f})")
        print(f"seed={seed}: baseline_cost={cost_baseline:.2f}, composed_cost={cost_composed:.2f}, "
              f"redundancy_dropped={dropped}, cost_substitutions={subs}")
        total_cost_baseline += cost_baseline
        total_cost_composed += cost_composed

    print(f"\n--- RESULT: {len(episodes)} real episodes checked ---")
    if violations:
        print(f"{len(violations)} VIOLATION(S):")
        for v in violations:
            print(f"  - {v}")
    else:
        pct = 100 * (1 - total_cost_composed / total_cost_baseline) if total_cost_baseline else 0
        print(f"ZERO violations. Total: baseline={total_cost_baseline:.2f}, composed={total_cost_composed:.2f} "
              f"({pct:.1f}% reduction vs this ground truth's own settings).")
    print()
    return violations


if __name__ == "__main__":
    v1 = check("V3_BASELINE", use_redundancy_baseline=False, use_cost_sub_baseline=False)
    v2 = check("V3_COST_SUB_ONLY", use_redundancy_baseline=False, use_cost_sub_baseline=True)
    total = len(v1) + len(v2)
    print(f"=== COMBINED: {total} total violations across both ground-truth batches (10 real episodes) ===")
