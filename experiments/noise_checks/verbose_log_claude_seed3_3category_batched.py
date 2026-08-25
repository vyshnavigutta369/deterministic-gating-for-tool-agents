"""
Verbose-logging rerun of claude-haiku-4-5, seed=3, location+transportation+dining,
BATCHED only -- captures the full prompt text and raw (pre-JSON-parse) response
text for every round. Exists because run_episode's round_log (run_brts_minimal.py)
only stores parsed proposed/valid/rejected lists and token counts, not the raw
prompt/response strings -- so the original sweep's 10-consecutive-empty-round
stretch for this exact seed/condition can't be inspected after the fact without
this rerun.

Caveat: call_llm_stub_claude never sets temperature (unlike the OpenAI path,
which pins temperature=0), so this uses Anthropic's API default and is NOT
guaranteed to reproduce the exact same empty-round stretch -- only the same
environment (same seed -> same tool costs/graph).
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, regardless of cwd
from run_brts_minimal import (
    build_prompt, tools_ready_in_memory, GroundtruthSolver,
    validate_batch, apply_batch, get_currently_executable, is_goal_reached,
    _get_anthropic_client, _extract_json_object,
)
from env.domains.travel.enums import ENUM_MAPPINGS
from env.core.data_types import get_final_type

SEED = 3
CATEGORIES = ["location", "transportation", "dining"]
TASK = "travel"
MODEL = "claude-haiku-4-5-20251001"
MAX_ROUNDS = 30
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verbose_log_claude_seed3_3category_batched.json")

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- aborting before making any real calls.")
        sys.exit(1)

    data = tools_ready_in_memory(refinement_level=5, min_atomic_cost=19, max_atomic_cost=21,
                                  noise_std=0.1, random_seed=SEED)
    solver = GroundtruthSolver(data["tools"])
    tools = solver.tools

    initial = {"TimeInfo"}
    target_types = set()
    for cat in CATEGORIES:
        for enum_class in ENUM_MAPPINGS[cat]["search"]:
            initial.add(enum_class.__name__)
        target_types.add(get_final_type(cat))
    state = frozenset(initial)

    client = _get_anthropic_client()
    round_log = []
    rounds = 0
    while not is_goal_reached(state, target_types) and rounds < MAX_ROUNDS:
        visible = [n for n in get_currently_executable(state, tools) if tools[n].output_type not in state]
        if not visible:
            break
        prompt = build_prompt(state, visible, tools, TASK, allow_batch=True)

        response = client.messages.create(
            model=MODEL, max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_content = response.content[0].text
        try:
            parsed = _extract_json_object(raw_content)
            proposed = parsed.get("tool_calls", [])
        except (json.JSONDecodeError, AttributeError):
            proposed = []

        valid, rejected = validate_batch(proposed, state, tools)
        state = apply_batch(state, tools, valid)

        round_log.append({
            "round": rounds,
            "prompt": prompt,
            "raw_response": raw_content,
            "proposed": proposed, "valid": valid, "rejected": rejected,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        })
        print(f"  round {rounds}: proposed={proposed} valid={valid} rejected={rejected}")
        rounds += 1

    result = {
        "seed": SEED, "categories": CATEGORIES, "model": MODEL,
        "rounds": rounds, "goal_reached": is_goal_reached(state, target_types),
        "round_log": round_log,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved full round-by-round log ({len(round_log)} rounds) to {OUT_PATH}")
    print(f"goal_reached={result['goal_reached']}")
