"""
Scans BFCL's real BFCL_v4_multi_turn_base tasks (200 human-curated
trajectories) for ones containing at least one turn whose multiple ground-
truth function calls are genuinely order-independent -- i.e.
validate_batch_bfcl says every permutation of that turn's calls executes
without error. Reuses validate_batch_bfcl unchanged; this file only decides
WHICH turns to test and in what state (replaying prior turns' real ground
truth first, so later-turn tests start from a state consistent with the
task actually having been carried out correctly so far).
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, regardless of cwd
from bfcl.bfcl_validate_batch import validate_batch_bfcl

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "bfcl_eval", "data")


def load_tasks():
    with open(os.path.join(DATA_DIR, "BFCL_v4_multi_turn_base.json")) as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(DATA_DIR, "possible_answer", "BFCL_v4_multi_turn_base.json")) as f:
        ground_truths = {json.loads(line)["id"]: json.loads(line)["ground_truth"] for line in f if line.strip()}
    return tasks, ground_truths


def find_independent_multi_call_turns(max_tasks: int = 5, min_calls_in_turn: int = 2, verbose: bool = True):
    """
    Returns a list of up to `max_tasks` dicts: {"task": task_dict,
    "ground_truth": [[...], ...], "turn_index": int, "turn_calls": [...]} --
    one qualifying turn per task (the first one found), stopping as soon as
    `max_tasks` tasks have been found (not necessarily scanning all 200).
    """
    tasks, ground_truths = load_tasks()
    qualifying = []

    for task in tasks:
        gt = ground_truths[task["id"]]
        prior_calls = []
        found_in_this_task = None
        for turn_index, turn_calls in enumerate(gt):
            if len(turn_calls) >= min_calls_in_turn:
                result = validate_batch_bfcl(turn_calls, task, prior_calls,
                                              trial_model_name=f"classify_{task['id']}")
                if result["valid"]:
                    found_in_this_task = {
                        "task": task, "ground_truth": gt,
                        "turn_index": turn_index, "turn_calls": turn_calls,
                    }
                    if verbose:
                        print(f"  {task['id']}: turn {turn_index} qualifies -- {turn_calls}")
                    break
            prior_calls = prior_calls + turn_calls
        if found_in_this_task:
            qualifying.append(found_in_this_task)
            if len(qualifying) >= max_tasks:
                break

    return qualifying


if __name__ == "__main__":
    print("Scanning BFCL_v4_multi_turn_base (200 real tasks) for genuinely-independent multi-call turns...\n")
    results = find_independent_multi_call_turns(max_tasks=5)
    print(f"\nFound {len(results)} qualifying tasks (stopped early once 5 were found, did not necessarily scan all 200):")
    for r in results:
        print(f"  {r['task']['id']}: turn {r['turn_index']}, calls={r['turn_calls']}")
