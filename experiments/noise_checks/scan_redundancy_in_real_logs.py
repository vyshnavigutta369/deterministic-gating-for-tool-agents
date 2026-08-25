"""
Stage 2 of the redundancy-elimination verification: zero-cost real-log
scan, reusing the same 4 surviving sweep logs already parsed for the
state-caching investigation (2x gpt-4o-mini, GPT-4o, Claude Haiku; all
v3-gated; all 3 pairings; n=20 each; both cost_substitution conditions).

Scans the REAL `valid=[...]` (executed) batches, not `proposed` -- this is
what actually incurred cost live -- for any round where 2+ executed tools
share the same real output_type (per the real per-seed registry, since
tool identity/cost is seed-randomized but structure is consistent).
Confirms the known seed=2/round=7 case is caught, then reports true
recurrence and real dollar waste across every real episode on disk.
"""
import re
import sys
sys.path.insert(0, '.')
from collections import defaultdict
from env.domains.travel.tool_registry import tools_ready_in_memory
from env.utils.solver import GroundtruthSolver
from brts_core import apply_redundancy_elimination

# Development note: these point at the original sweep-run stdout captures
# used during investigation. That ephemeral session storage has since
# rotated out, so this script is not directly runnable as shipped -- kept
# for methodology transparency (Section 5.2 / Table 2); point LOG_FILES
# at your own saved sweep-run logs (same format) to rerun the scan.
LOG_FILES = [
    ("<path-to-saved-sweep-log>/gpt4o_mini_run1.output", "gpt-4o-mini"),
    ("<path-to-saved-sweep-log>/gpt4o_mini_run2.output", "gpt-4o-mini"),
    ("<path-to-saved-sweep-log>/gpt4o_run.output", "gpt-4o"),
    ("<path-to-saved-sweep-log>/claude_haiku_run.output", "claude-haiku"),
]

PAIRINGS = {
    "location+transportation": ("location", ["location", "transportation"]),
    "location+accommodation": ("location", ["location", "accommodation"]),
    "location+transportation+dining": ("travel", ["location", "transportation", "dining"]),
}

pairing_header_re = re.compile(r"^##########\s+(.+?)\s+##########$")
condition_header_re = re.compile(r"^=== (V3_NO_COST_SUB|V3_COST_AWARE|V3_NO_CACHE|V3_WITH_CACHE), seed=(\d+) ===$")
round_re = re.compile(r"^\s*round (\d+): proposed=(\[.*?\]) valid=(\[.*?\]) rejected=")


def parse_log(path):
    """Returns {pairing_label: {(condition, seed): [(round_idx, valid_list), ...]}}"""
    episodes = defaultdict(dict)
    current_pairing = None
    current_key = None
    with open(path) as f:
        for line in f:
            m = pairing_header_re.match(line.strip())
            if m:
                current_pairing = m.group(1)
                current_key = None
                continue
            m = condition_header_re.match(line.strip())
            if m and current_pairing is not None:
                current_key = (m.group(1).lower(), int(m.group(2)))
                episodes[current_pairing][current_key] = []
                continue
            m = round_re.match(line)
            if m and current_pairing is not None and current_key is not None:
                round_idx = int(m.group(1))
                valid = eval(m.group(3))  # trusted, locally-generated log file
                episodes[current_pairing][current_key].append((round_idx, valid))
    return episodes


def build_tools_for_seed(seed):
    data = tools_ready_in_memory(refinement_level=5, min_atomic_cost=19, max_atomic_cost=21,
                                  noise_std=0.1, random_seed=seed)
    solver = GroundtruthSolver(data['tools'])
    return solver.tools


if __name__ == "__main__":
    # --- Known-positive sanity check first ---
    tools_seed2 = build_tools_for_seed(2)
    known_batch = ['Transportation_Finish_from_Step1_6Steps', 'Transportation_Finish_from_Step2_5Steps',
                   'Transportation_Finish_from_Step3_4Steps', 'Transportation_Finish_from_Step4_3Steps']
    deduped, dropped = apply_redundancy_elimination(known_batch, tools_seed2)
    known_waste = sum(d["cost_saved"] for d in dropped)
    print(f"Known-positive sanity check (seed=2, round=7): detected {len(dropped)} redundant call(s), "
          f"${known_waste:.2f} waste -- {'PASS' if len(dropped) == 3 else 'FAIL'}\n")

    total_rounds_scanned = 0
    total_redundant_rounds = 0
    total_redundant_calls_dropped = 0
    total_dollar_waste = 0.0
    by_model = defaultdict(lambda: {"rounds": 0, "redundant_rounds": 0, "calls_dropped": 0, "waste": 0.0})
    by_pairing = defaultdict(lambda: {"rounds": 0, "redundant_rounds": 0, "calls_dropped": 0, "waste": 0.0})
    examples = []

    tools_cache = {}  # seed -> tools dict, built once per seed regardless of file/condition/pairing

    for path, model in LOG_FILES:
        episodes = parse_log(path)
        for pairing_label, by_key in episodes.items():
            if pairing_label not in PAIRINGS:
                continue
            for (condition, seed), rounds in by_key.items():
                if seed not in tools_cache:
                    tools_cache[seed] = build_tools_for_seed(seed)
                tools = tools_cache[seed]
                for round_idx, valid in rounds:
                    total_rounds_scanned += 1
                    by_model[model]["rounds"] += 1
                    by_pairing[pairing_label]["rounds"] += 1
                    if len(valid) < 2:
                        continue
                    deduped, dropped = apply_redundancy_elimination(valid, tools)
                    if dropped:
                        total_redundant_rounds += 1
                        total_redundant_calls_dropped += len(dropped)
                        waste = sum(d["cost_saved"] for d in dropped)
                        total_dollar_waste += waste
                        by_model[model]["redundant_rounds"] += 1
                        by_model[model]["calls_dropped"] += len(dropped)
                        by_model[model]["waste"] += waste
                        by_pairing[pairing_label]["redundant_rounds"] += 1
                        by_pairing[pairing_label]["calls_dropped"] += len(dropped)
                        by_pairing[pairing_label]["waste"] += waste
                        if len(examples) < 5:
                            examples.append((path.split("/")[-1], pairing_label, model, condition, seed,
                                              round_idx, valid, dropped))

    print(f"=== OVERALL: {total_rounds_scanned} real rounds scanned across all 4 logs ===")
    print(f"Rounds with real redundancy (executed batch had 2+ tools sharing an output_type): "
          f"{total_redundant_rounds} ({100*total_redundant_rounds/total_rounds_scanned:.2f}%)")
    print(f"Total redundant calls that would have been dropped: {total_redundant_calls_dropped}")
    print(f"Total real dollar-equivalent tool-cost waste: ${total_dollar_waste:.2f}\n")

    print("=== BY MODEL ===")
    for model, stats in by_model.items():
        pct = 100 * stats["redundant_rounds"] / stats["rounds"] if stats["rounds"] else 0
        print(f"{model}: {stats['redundant_rounds']}/{stats['rounds']} rounds ({pct:.2f}%), "
              f"{stats['calls_dropped']} calls dropped, ${stats['waste']:.2f} waste")

    print("\n=== BY PAIRING ===")
    for pairing, stats in by_pairing.items():
        pct = 100 * stats["redundant_rounds"] / stats["rounds"] if stats["rounds"] else 0
        print(f"{pairing}: {stats['redundant_rounds']}/{stats['rounds']} rounds ({pct:.2f}%), "
              f"{stats['calls_dropped']} calls dropped, ${stats['waste']:.2f} waste")

    print("\n=== SAMPLE REAL REDUNDANT ROUNDS ===")
    for path_name, pairing_label, model, condition, seed, round_idx, valid, dropped in examples:
        print(f"  [{path_name}] {pairing_label}/{model}/{condition}/seed={seed}/round={round_idx}: "
              f"valid={valid}")
        for d in dropped:
            print(f"    dropped {d['name']!r} (output={d['output_type']}, cost_saved=${d['cost_saved']:.2f}), "
                  f"kept {d['kept']!r}")
