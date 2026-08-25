"""
Minimal LLM-facing harness for BFCL's multi_turn category.

Reuses, unchanged:
- validate_batch_bfcl from bfcl/bfcl_validate_batch.py as the safety gate --
  a proposed batch only executes for real if every possible ordering of it
  succeeds against BFCL's real simulators (bfcl_eval's own
  execute_multi_turn_func_call, never reimplemented).
- multi_turn_checker from BFCL's own vendored eval_checker -- the real,
  state-based correctness check (does the model's executed state match the
  ground truth's executed state), for goal-completion, rather than
  inventing a pass/fail criterion.
- call_llm_stub from run_brts_minimal.py, completely unchanged -- it's a
  pure OpenAI wrapper that parses a "tool_calls" JSON key regardless of
  domain, so the BFCL prompt below deliberately asks for the same key
  rather than forking call_llm_stub for a different key name.

build_prompt is adapted (not reused unchanged) because CostBench's version
describes tools via static input/output types with a known state set;
BFCL functions have no such static contract -- all functions for the task's
involved_classes are shown every round, and there is no "state" to print,
just the running list of already-executed calls for context.

Turn-completion rule: the model itself signals it's done with the current
turn by proposing zero function calls (empty tool_calls) -- the standard
function-calling-agent convention, mirroring how a real multi-turn agent
loop works (not a made-up stopping rule).
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, regardless of cwd
import bfcl  # noqa: F401 -- sets up vendor/ on sys.path as a side effect
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl.bfcl_validate_batch import validate_batch_bfcl
from run_brts_minimal import call_llm_stub, call_llm_stub_claude

FUNC_DOC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bfcl", "vendor",
                            "bfcl_eval", "data", "multi_turn_func_doc")

# Maps involved_classes -> function-doc file, duplicated here (not imported)
# because the real mapping (MULTI_TURN_FUNC_DOC_FILE_MAPPING in
# executable_backend_config.py) sits alongside CLASS_FILE_PATH_MAPPING,
# which we DO reuse directly -- this one's just a filename lookup.
from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING

_CLASS_TO_DOC_FILE = {
    "GorillaFileSystem": "gorilla_file_system.json", "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json", "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json", "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json", "VehicleControlAPI": "vehicle_control.json",
    "WebSearchAPI": "web_search.json", "MemoryAPI_kv": "memory_kv.json",
    "MemoryAPI_vector": "memory_vector.json", "MemoryAPI_rec_sum": "memory_rec_sum.json",
}


def _load_function_docs(involved_classes, excluded_function=()):
    docs = []
    for class_name in involved_classes:
        doc_file = _CLASS_TO_DOC_FILE.get(class_name)
        if not doc_file:
            continue
        path = os.path.join(FUNC_DOC_DIR, doc_file)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    if doc["name"] not in excluded_function:
                        docs.append(doc)
    return docs


def _format_feedback(proposed_calls: list, validation: dict) -> str:
    """Turns validate_batch_bfcl's real per-call results into concrete
    feedback for the NEXT round's prompt. The first BFCL sweep run showed
    the model repeat an identical failing proposal for an entire round
    budget when given nothing but silence after a rejection -- this exists
    specifically to close that gap."""
    reason = validation["reason"]
    if reason == "too_large_to_verify":
        return (f"Your previous proposal of {len(proposed_calls)} calls was REJECTED: too many calls to "
                f"verify safe batching in one round. Propose fewer calls.")
    lines = "\n".join(f"  {call} -> {result}" for call, result in
                       zip(proposed_calls, validation["canonical_results"]))
    if reason == "canonical_order_failed":
        return (f"Your previous proposal was REJECTED because at least one call failed when executed:\n{lines}\n"
                f"Do not repeat the same failing call verbatim -- fix the argument(s) or try a different "
                f"approach based on the actual error above.")
    if reason == "order_dependent":
        return (f"Your previous proposal succeeded when executed in the order you gave, but was REJECTED for "
                f"batching because a DIFFERENT order of the same calls would fail (so they are not safe to "
                f"batch together):\n{lines}\n"
                f"Propose these one at a time instead, or batch only calls that are genuinely order-independent.")
    return ""


def build_bfcl_prompt(task: dict, user_message: str, executed_calls: list, allow_batch: bool,
                       last_round_feedback: str = None) -> str:
    # excluded_function exists specifically to block an easier shortcut for
    # this task variant (e.g. disallowing 'cp' so the task must be solved
    # via mkdir+mv instead) -- showing it would let the model bypass that.
    docs = _load_function_docs(task["involved_classes"], task.get("excluded_function", ()))
    func_block = "\n".join(
        f"- {d['name']}({', '.join(d['parameters'].get('properties', {}).keys())}): {d['description']}"
        for d in docs
    )
    executed_block = "\n".join(executed_calls) if executed_calls else "(none yet)"
    batch_rule = (
        "You MAY propose multiple function calls this round, but ONLY if they are genuinely "
        "independent of each other and of order -- never a call that depends on another "
        "proposed call's result."
        if allow_batch else
        "You may propose exactly ONE function call this round."
    )
    user_block = f"New user message this turn: {user_message}" if user_message is not None else (
        "No new user message -- continue working on the current turn, or respond with an "
        "empty tool_calls list if the current turn's request is now fully satisfied."
    )
    feedback_block = f"\n{last_round_feedback}\n" if last_round_feedback else ""
    return f"""You are a function-calling agent completing a multi-turn task.
{user_block}

Function calls already executed so far in this conversation:
{executed_block}
{feedback_block}
Currently available functions:
{func_block}

{batch_rule}
If the current turn's request is fully satisfied, respond with an empty list.
Respond ONLY with JSON: {{"tool_calls": ["function_name(arg1=value1, ...)", ...]}}
"""


def run_bfcl_episode(task: dict, ground_truth: list, allow_batch: bool, model: str = "gpt-4o",
                      condition_label: str = "batched", max_rounds_per_turn: int = 6):
    executed_calls = []
    model_result_list = []  # [turn][round][call_str, ...] -- shape multi_turn_checker expects
    total_llm_calls = 0
    stub = call_llm_stub_claude if model.startswith("claude") else call_llm_stub

    for turn_index, turn_messages in enumerate(task["question"]):
        user_message = "\n".join(m["content"] for m in turn_messages)
        this_turn_rounds = []
        last_round_feedback = None
        for round_in_turn in range(max_rounds_per_turn):
            prompt = build_bfcl_prompt(task, user_message if round_in_turn == 0 else None,
                                        executed_calls, allow_batch, last_round_feedback)
            proposed, usage = stub(prompt, model=model)
            total_llm_calls += 1

            # NOTE: deliberately no empty-response corrective feedback here,
            # same as bfcl_parallel_episode.py -- an empty proposal is this
            # domain's intentional "this turn is complete" signal, not a
            # stall (see module docstring).
            if not proposed:
                break  # model signals this turn is complete

            if not allow_batch and len(proposed) > 1:
                proposed = proposed[:1]  # enforce the same single-call sequential constraint used elsewhere

            validation = validate_batch_bfcl(proposed, task, executed_calls, trial_model_name=condition_label)
            if validation["valid"]:
                execute_multi_turn_func_call(
                    proposed, task["initial_config"], task["involved_classes"],
                    model_name=condition_label, test_entry_id=task["id"], is_evaL_run=True,
                )
                executed_calls.extend(proposed)
                this_turn_rounds.append(list(proposed))
                last_round_feedback = None  # success -- don't carry a stale failure note forward
                print(f"  turn={turn_index} round={round_in_turn}: proposed={proposed} EXECUTED")
            else:
                this_turn_rounds.append([])
                last_round_feedback = _format_feedback(proposed, validation)
                print(f"  turn={turn_index} round={round_in_turn}: proposed={proposed} REJECTED "
                      f"(reason={validation['reason']})")
        model_result_list.append(this_turn_rounds)

    check = multi_turn_checker(model_result_list, ground_truth, task,
                                test_category="multi_turn_base", model_name=f"{condition_label}_check")

    return {
        "llm_calls": total_llm_calls,
        "goal_reached": check["valid"],
        "check_detail": check,
        "executed_calls": executed_calls,
        "model_result_list": model_result_list,
    }
