"""
Minimal LLM-facing harness for BFCL's parallel category: single-turn,
stateless tasks where the user's request genuinely requires multiple
independent function calls (e.g. "play X and Y on Spotify" needing two
separate spotify.play calls).

Reuses, unchanged:
- ast_checker (bfcl_eval.eval_checker.ast_eval, vendored real code) for
  correctness -- the model's accumulated proposed calls are compared against
  ground truth via BFCL's own parallel_function_checker_no_order logic, the
  same check the official leaderboard uses for this category, not an
  invented comparison.
- call_llm_stub from run_brts_minimal.py, completely unchanged.

Different from the multi_turn harness in one structural way: parallel tasks
have NO backend state at all -- nothing is executed, so there is no
validate_batch_bfcl-style permutation check (no real order-dependence risk
exists here, unlike multi_turn's stateful filesystem/Twitter/etc). The only
per-round gate is a lightweight well-formedness check (is each proposed call
a single-key dict naming an available function?), with the same corrective-
feedback pattern proven useful for multi_turn -- malformed proposals are
dropped and reported back rather than silently discarded.

Turn-completion rule: same convention as multi_turn -- the model signals
it's done by proposing an empty list.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, regardless of cwd
import bfcl  # noqa: F401 -- sets up vendor/ on sys.path as a side effect
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.constants.enums import Language
from run_brts_minimal import call_llm_stub, call_llm_stub_claude


def build_parallel_prompt(task: dict, accumulated_calls: list, allow_batch: bool,
                           last_round_feedback: str = None) -> str:
    func_block = "\n".join(
        f"- {f['name']}({', '.join(f['parameters'].get('properties', {}).keys())}): {f['description']}"
        for f in task["function"]
    )
    accumulated_block = json.dumps(accumulated_calls) if accumulated_calls else "(none yet)"
    batch_rule = (
        "You MAY propose multiple function calls this round if the user's request genuinely needs "
        "more than one independent call -- propose all of them together when you can."
        if allow_batch else
        "You may propose exactly ONE function call this round."
    )
    feedback_block = f"\n{last_round_feedback}\n" if last_round_feedback else ""
    user_message = task["question"][0][0]["content"]
    return f"""You are a function-calling agent. Respond to the user's request using the available functions.
User request: {user_message}

Function calls already proposed so far: {accumulated_block}
{feedback_block}
Available functions:
{func_block}

{batch_rule}
If you believe all necessary calls have already been proposed above, respond with an empty list.
Each function call must be a single-key JSON object where the key is the EXACT function name from
the list above (e.g. "spotify.play") and the value is a dict of that function's arguments -- for
example {{"spotify.play": {{"artist": "Drake", "duration": 10}}}}. Do NOT wrap a call in a generic
envelope like {{"function_name": ..., "params": ...}}.
Respond ONLY with JSON: {{"tool_calls": [{{"<exact_function_name>": {{"<param_name>": value, ...}}}}, ...]}}
"""


def _normalize_call_envelope(call):
    """Seen for real with Claude Haiku: instead of the canonical single-key
    {<function_name>: {...args...}} shape the prompt asks for, it wraps
    calls in a {"function_name": <name>, "params": {...}} envelope -- reject
    that outright (as the original code did) and every round gets dropped
    with no valid calls ever recorded, since the model reliably repeats the
    same envelope shape. The call's intent is unambiguous, so normalize it
    to the canonical shape rather than treating it as malformed."""
    if (isinstance(call, dict) and set(call.keys()) == {"function_name", "params"}
            and isinstance(call.get("params"), dict)):
        return {call["function_name"]: call["params"]}
    return call


def _validate_wellformed(proposed: list, valid_function_names: set):
    valid, rejected = [], []
    for call in proposed:
        normalized = _normalize_call_envelope(call)
        if isinstance(normalized, dict) and len(normalized) == 1 and next(iter(normalized)) in valid_function_names:
            valid.append(normalized)
        else:
            rejected.append(call)
    return valid, rejected


def run_parallel_episode(task: dict, ground_truth: list, allow_batch: bool, model: str = "gpt-4o",
                          max_rounds: int = 6):
    valid_function_names = {f["name"] for f in task["function"]}
    accumulated_calls = []
    total_llm_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    last_round_feedback = None

    stub = call_llm_stub_claude if model.startswith("claude") else call_llm_stub
    for round_idx in range(max_rounds):
        prompt = build_parallel_prompt(task, accumulated_calls, allow_batch, last_round_feedback)
        proposed, usage = stub(prompt, model=model)
        total_llm_calls += 1
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]

        # NOTE: deliberately no empty-response corrective feedback here, unlike
        # CostBench/synthetic -- an empty proposal is this domain's intentional
        # "I'm done" signal (see module docstring), not a stall. Applying the
        # CostBench-style warning would misfire on every normal completion.
        if not proposed:
            print(f"  round={round_idx}: proposed=[] (model signals done)")
            break

        if not allow_batch and len(proposed) > 1:
            proposed = proposed[:1]  # enforce the same single-call sequential constraint used elsewhere

        valid, rejected = _validate_wellformed(proposed, valid_function_names)
        accumulated_calls.extend(valid)
        if rejected:
            last_round_feedback = (
                f"These proposals were malformed or used an unavailable function and were DROPPED: "
                f"{rejected}. Each call must be a single-key dict "
                f'{{"function_name": {{...args...}}}} using one of the available functions listed below.'
            )
        else:
            last_round_feedback = None
        print(f"  round={round_idx}: proposed={proposed} valid={valid} rejected={rejected}")

    check = ast_checker(task["function"], accumulated_calls, ground_truth, Language.PYTHON,
                         "parallel", model)

    return {
        "llm_calls": total_llm_calls,
        "goal_reached": check["valid"],
        "check_detail": check,
        "accumulated_calls": accumulated_calls,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
    }
