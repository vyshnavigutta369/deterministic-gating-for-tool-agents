"""
First LLM-driven BRTS test on PDDL Logistics -- previously only the
deterministic/theorem side was tested here (logistics_precompute.py). This
tests whether a real LLM discovers and correctly batches independent
package-delivery actions, mirroring the CostBench/BFCL harness pattern:
real execution-based validation (op.applicable/op.apply), not just prompted
trust.
"""
import os
import sys
# pyperplan is now a real pip-installed package (pip install pyperplan) --
# it lives in site-packages and needs no manual sys.path entry at all. Only
# this project's own repo root needs adding, for `from run_brts_minimal
# import call_llm_stub` to resolve regardless of invocation directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyperplan.pddl.parser import Parser
from pyperplan.grounding import ground
from run_brts_minimal import call_llm_stub, call_llm_stub_claude  # reuse the real, already-verified wrappers


def get_applicable_actions(task, state):
    return [op for op in task.operators if op.applicable(state)]


def format_state(state):
    return ", ".join(sorted(state))


def build_logistics_prompt(state, applicable_ops, goals, allow_batch, last_round_feedback=None):
    action_block = "\n".join(f"- {op.name}" for op in applicable_ops)
    goal_block = ", ".join(sorted(goals))
    batch_rule = (
        "You MAY propose MULTIPLE actions this round, but ONLY actions that are genuinely "
        "independent (neither depends on the other's effect). If unsure, propose one at a time."
        if allow_batch else
        "Propose exactly ONE action this round."
    )
    feedback_block = f"\n{last_round_feedback}\n" if last_round_feedback else ""
    return f"""You are solving a logistics planning problem: deliver packages to their destinations
using trucks (city-local) and airplanes (between cities).

Current world state (true facts): {format_state(state)}

Goal (facts that must become true): {goal_block}
{feedback_block}
Currently applicable actions (preconditions satisfied):
{action_block}

{batch_rule}
Respond ONLY with JSON: {{"tool_calls": ["exact-action-name-from-list-above", ...]}}
If you believe the goal is already satisfied, respond with an empty list.
"""


def build_empty_response_feedback_logistics(applicable_ops):
    """Same failure family as run_brts_minimal.build_empty_response_feedback
    (CostBench) -- an empty proposal here is always a stall, never an
    intentional completion signal (unlike BFCL), so it deserves the same
    corrective treatment. A logistics-flavored ("action", not "tool")
    version so the wording matches this domain's own prompt framing."""
    return (
        "WARNING: last round you proposed zero actions (an empty "
        "\"tool_calls\": [] list), but the goal has not yet been reached and "
        "no progress was made. At least one action IS genuinely applicable "
        f"right now: {[op.name for op in applicable_ops]}. Do not repeat an "
        "empty response -- reconsider and propose at least one of the "
        "actions listed above as 'Currently applicable actions'."
    )


def _normalize_action_name(name: str) -> str:
    """Match key robust to parenthesization/whitespace: op.name is literally
    "(load-truck obj1 tru1 pos1)" (pyperplan includes the parens), but a
    real run showed the model consistently echoing action names WITHOUT the
    surrounding parens -- a pure formatting mismatch that caused a strict
    equality check to reject every single proposal, every round, across all
    4 episodes (100% rejection, zero actions ever executed), with nothing to
    do with planning correctness. Stripping parens/whitespace on both sides
    before comparing fixes the actual bug instead of just adding feedback
    for the model to work around a harness brittleness."""
    return " ".join(name.strip().strip("()").split())


def validate_batch_logistics(proposed_names, task, state):
    """Real execution-based validation: a batch is safe only if every action
    is genuinely applicable given the state as of the START of the round
    (not depending on another proposed action's effect) -- same discipline
    as validate_batch (CostBench) and validate_batch_bfcl. Matches proposed
    names against the real operators via _normalize_action_name, returning
    each match's own canonical op.name (not the raw proposed string) so
    downstream code keeps using task.operators' real names unchanged."""
    normalized_to_op = {_normalize_action_name(op.name): op for op in task.operators}
    valid, rejected = [], []
    for name in proposed_names:
        op = normalized_to_op.get(_normalize_action_name(name))
        if op is not None and op.applicable(state):
            valid.append(op.name)  # canonical name, not the raw proposed string
        else:
            rejected.append(name)
    return valid, rejected


def run_logistics_episode(domain_file, problem_file, allow_batch, model="gpt-4o", max_rounds=15):
    parser = Parser(domain_file, problem_file)
    domain = parser.parse_domain()
    problem = parser.parse_problem(domain)
    task = ground(problem)

    name_to_op = {op.name: op for op in task.operators}
    state = task.initial_state
    visited_states = {state}  # cycle detection: states already seen this episode
    llm_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    round_log = []
    last_round_feedback = None
    stub = call_llm_stub_claude if model.startswith("claude") else call_llm_stub

    for round_num in range(max_rounds):
        if task.goal_reached(state):
            break
        applicable = get_applicable_actions(task, state)
        if not applicable:
            break
        prompt = build_logistics_prompt(state, applicable, task.goals, allow_batch, last_round_feedback)
        proposed, usage = stub(prompt, model=model)
        llm_calls += 1
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]

        valid, rejected = validate_batch_logistics(proposed, task, state)
        if not allow_batch and len(valid) > 1:
            valid = valid[:1]
        new_state = state
        for name in valid:
            new_state = name_to_op[name].apply(new_state)

        # Cycle detection: the action(s) were individually "valid" (legal
        # given the round's starting state), but if the RESULTING state was
        # already visited earlier in this episode, no real progress was
        # made (e.g. loading then immediately unloading undoes itself). A
        # real run got stuck oscillating exactly like this, forever, with
        # no signal telling it so -- flag it explicitly for the next round.
        if not proposed:
            # Disjoint from the cycle-detection branch below: valid is
            # necessarily [] whenever proposed is [], so this never
            # shadows a genuine cycle warning.
            last_round_feedback = build_empty_response_feedback_logistics(applicable)
        elif valid and new_state in visited_states:
            last_round_feedback = (
                f"WARNING: the action(s) you just proposed ({valid}) returned to a state you have "
                f"ALREADY visited earlier in this episode -- no real progress was made (e.g. loading "
                f"then immediately unloading a package undoes itself). Do NOT repeat this cycle -- "
                f"choose a genuinely different action this round (e.g. drive the truck to its "
                f"destination before unloading)."
            )
        else:
            last_round_feedback = None
        state = new_state
        visited_states.add(state)

        round_log.append({"round": round_num, "proposed": proposed, "valid": valid, "rejected": rejected})
        if not proposed:
            tag = " [EMPTY RESPONSE]"
        elif last_round_feedback:
            tag = " [CYCLE DETECTED]"
        else:
            tag = ""
        print(f"  round={round_num}: proposed={proposed} valid={valid} rejected={rejected}{tag}")

    goal_reached = task.goal_reached(state)
    return {
        "llm_calls": llm_calls, "goal_reached": goal_reached, "round_log": round_log,
        "total_prompt_tokens": total_prompt_tokens, "total_completion_tokens": total_completion_tokens,
    }


if __name__ == "__main__":
    print("Harness built. Dry-checking structure with no LLM calls first.")
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logistics")
    parser = Parser(f"{base}/logistics_domain.pddl", f"{base}/task_independent.pddl")
    domain = parser.parse_domain()
    problem = parser.parse_problem(domain)
    task = ground(problem)
    print("goal_reached method exists:", hasattr(task, "goal_reached"))
    applicable = get_applicable_actions(task, task.initial_state)
    print(f"Applicable actions from initial state: {len(applicable)}")
    prompt = build_logistics_prompt(task.initial_state, applicable, task.goals, allow_batch=True)
    print("\n--- Sample prompt ---")
    print(prompt)
