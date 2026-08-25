"""
Minimal LLM-facing episode runner for the synthetic domain (brts.synthetic):
generic categories, strictly linear atomic chains, NO bundle/shortcut tools.

Reuses, unchanged:
- validate_batch / apply_batch / get_currently_executable / is_goal_reached
  from brts_core.py -- already domain-agnostic, operate purely on
  FrozenSet[str] state and a name->Tool dict, no CostBench-specific logic.
- build_prompt / build_tool_description_block / call_llm_stub from
  run_brts_minimal.py -- build_prompt's tool description block only reads
  tool.cost/.input_types/.output_type (already generic), and call_llm_stub is
  a pure OpenAI wrapper with no CostBench-specific logic either. Neither
  needed any adaptation.

Only run_synthetic_episode itself is new: the original run_episode builds
its initial state from CostBench's real tools_ready_in_memory + ENUM_MAPPINGS;
this builds it from brts.synthetic.generate_synthetic_domain instead. Kept
separate from experiments/llm_sweep.py's run_llm_sweep because the episode
function itself differs (different domain, different task label), not just
the parameters.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, regardless of cwd
from env.core.data_types import get_final_type
from brts_core import validate_batch, apply_batch, get_currently_executable, is_goal_reached
from brts.synthetic import generate_synthetic_domain
from run_brts_minimal import build_prompt, call_llm_stub, call_llm_stub_claude, build_empty_response_feedback


def run_synthetic_episode(categories, seed: int, allow_batch: bool, max_rounds: int = 30, model: str = "gpt-4o"):
    initial_state, tools = generate_synthetic_domain(categories, seed=seed)
    target_types = {get_final_type(c) for c in categories}
    state = initial_state

    rounds = 0
    total_llm_calls = 0
    total_tool_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    round_log = []
    last_round_feedback = None
    while not is_goal_reached(state, target_types) and rounds < max_rounds:
        visible = [n for n in get_currently_executable(state, tools) if tools[n].output_type not in state]
        if not visible:
            break
        prompt = build_prompt(state, visible, tools, task="synthetic multi-category planning",
                               allow_batch=allow_batch, last_round_feedback=last_round_feedback)
        stub = call_llm_stub_claude if model.startswith("claude") else call_llm_stub
        proposed, usage = stub(prompt, model=model)
        total_llm_calls += 1
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]

        valid, rejected = validate_batch(proposed, state, tools)
        if not allow_batch and len(valid) > 1:
            valid = valid[:1]  # enforce the same single-call sequential constraint used elsewhere
        state = apply_batch(state, tools, valid)
        total_tool_cost += sum(tools[n].cost for n in valid)
        round_log.append({"round": rounds, "proposed": proposed, "valid": valid, "rejected": rejected})
        print(f"  round {rounds}: proposed={proposed} valid={valid} rejected={rejected}")
        # An empty proposal here is always a stall (same semantics as CostBench's
        # run_episode) -- unlike BFCL, there's no "empty = intentionally done"
        # convention in this domain, so the CostBench fix applies unmodified.
        last_round_feedback = build_empty_response_feedback(visible) if not proposed else None
        rounds += 1

    return {
        "rounds": rounds, "llm_calls": total_llm_calls, "total_tool_cost": total_tool_cost,
        "total_prompt_tokens": total_prompt_tokens, "total_completion_tokens": total_completion_tokens,
        "goal_reached": is_goal_reached(state, target_types),
        "round_log": round_log,
    }
