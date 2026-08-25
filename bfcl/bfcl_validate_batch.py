"""
validate_batch_bfcl: BFCL's analogue of brts_core.py's validate_batch.

CostBench's validate_batch could check independence statically (declared
input/output types on each Tool). BFCL function calls have no such static
contract -- whether two calls are safe to batch (truly order-independent)
can only be determined by actually trying every possible execution order
against the real simulated environment and checking none of them error.
So this reuses bfcl_eval's real execute_multi_turn_func_call (never
reimplemented) to try every permutation of a proposed batch, starting each
trial from a fresh instance replaying the same prior_calls, and only
approves the batch if EVERY ordering executes without error.
"""
import itertools
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, regardless of cwd
import bfcl  # noqa: F401 -- sets up vendor/ on sys.path as a side effect
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

# Exhaustively checking every ordering is O(n!) -- cap how large a proposed
# batch can be before we give up trying to prove independence and just
# reject it as unsafe to batch (5! = 120 trials is already generous for how
# many calls a model realistically proposes in one round).
MAX_BATCH_SIZE_FOR_PERMUTATION_CHECK = 5


def _is_error(result) -> bool:
    """A call "fails" in BFCL's real simulators in TWO distinct ways, both of
    which must be caught (discovered by actually probing real functions, not
    assumed): (1) a genuine Python exception, wrapped by
    execute_multi_turn_func_call as the string "Error during execution: ...";
    (2) a normal, non-exception return value that is itself a semantic error
    dict, e.g. mv() returning {"error": "mv: cannot move ...: No such file or
    directory"} -- these never raise, they're serialized to a JSON string by
    execute_multi_turn_func_call's own post-processing, so they'd otherwise
    look like a perfectly successful call."""
    if not isinstance(result, str):
        return False
    if result.startswith("Error during execution"):
        return True
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(parsed, dict) and "error" in parsed


def validate_batch_bfcl(proposed_calls, task: dict, prior_calls, trial_model_name: str = "validate") -> dict:
    """
    proposed_calls: function-call strings proposed for the CURRENT round.
    task: full BFCL task dict (needs 'id', 'initial_config', 'involved_classes').
    prior_calls: function-call strings already executed earlier in this same
                 episode -- replayed fresh for every permutation trial so
                 each trial starts from the identical real state.

    Returns {"valid": [...], "rejected": [...], "canonical_results": [...],
    "reason": str}:
    - valid == proposed_calls unchanged if every ordering executes without
      error; otherwise valid is empty and everything is rejected (never
      partially approved, since a partial approval could still depend on
      order).
    - canonical_results: the REAL per-call results of executing
      proposed_calls in the exact order given (always computed, even when
      rejected) -- this is what lets a caller show the model concrete
      feedback about its own proposal instead of a bare "rejected", which
      is what let the model repeat an identical failing guess for an
      entire round budget in the first BFCL sweep run.
    - reason: one of "accepted", "too_large_to_verify",
      "canonical_order_failed" (the proposal itself doesn't even work, in
      the order given), "order_dependent" (the proposal works as given but
      some OTHER ordering fails -- so it's not safe to batch, even though
      nothing is wrong with the calls themselves).
    """
    if not proposed_calls:
        return {"valid": [], "rejected": [], "canonical_results": [], "reason": "empty"}

    if len(proposed_calls) > MAX_BATCH_SIZE_FOR_PERMUTATION_CHECK:
        return {"valid": [], "rejected": list(proposed_calls), "canonical_results": [],
                "reason": "too_large_to_verify"}

    # Always compute the canonical (as-proposed) order's real results first --
    # both for feedback, and because if the proposed order itself already
    # fails, there is no need to spend n! trials checking other orderings.
    canonical_trial_id = f"{task['id']}_{trial_model_name}_canonical"
    canonical_all, _ = execute_multi_turn_func_call(
        list(prior_calls) + list(proposed_calls), task["initial_config"], task["involved_classes"],
        model_name=trial_model_name, test_entry_id=canonical_trial_id, is_evaL_run=True,
    )
    canonical_results = canonical_all[len(prior_calls):]
    if any(_is_error(r) for r in canonical_results):
        return {"valid": [], "rejected": list(proposed_calls), "canonical_results": canonical_results,
                "reason": "canonical_order_failed"}

    if len(proposed_calls) > 1:
        canonical_tuple = tuple(proposed_calls)
        for i, perm in enumerate(itertools.permutations(proposed_calls)):
            if perm == canonical_tuple:
                continue  # already checked above
            trial_id = f"{task['id']}_{trial_model_name}_perm{i}"
            results, _ = execute_multi_turn_func_call(
                list(prior_calls) + list(perm), task["initial_config"], task["involved_classes"],
                model_name=trial_model_name, test_entry_id=trial_id, is_evaL_run=True,
            )
            this_perm_results = results[len(prior_calls):]
            if any(_is_error(r) for r in this_perm_results):
                return {"valid": [], "rejected": list(proposed_calls), "canonical_results": canonical_results,
                        "reason": "order_dependent"}

    return {"valid": list(proposed_calls), "rejected": [], "canonical_results": canonical_results,
            "reason": "accepted"}
