"""
Synthetic domain generator (FakeTool-style, recreated -- no scale_sweep_experiment.py
or FakeTool code was found anywhere in this project, so this rebuilds the
same idea from scratch): each category gets its own strictly linear chain of
atomic tools, with NO bundle/shortcut tools at all -- every category must be
completed via its FULL atomic chain, no exceptions. This is deliberately the
opposite of CostBench's real registry, which offers composite "shortcut"
tools (Full_Planning_to_StepN, Complete_8Steps_Pipeline, etc.) that let a
path skip steps. The point: with no shortcut to choose between, there is
only ONE possible path per category, so precompute_then_batch_selection
(cost-optimal) and precompute_length_optimal_selection (length-optimal)
should collapse to the identical sequence and identical rounds/cost -- this
isolates whether the two precompute variants only differ because CostBench's
real registry happens to offer shortcuts, not because of any difference in
the algorithms themselves.

Reuses the real Tool class (env.core.base_types.create_tool_from_dict) so the
synthetic registry is a drop-in for GroundtruthSolver / execute_precomputed_sequences,
exactly like the real CostBench registry -- and reuses real category names
(e.g. "attraction" -> get_final_type("attraction") == "TravelAttraction") so
precompute_then_batch_selection / precompute_length_optimal_selection work
completely unmodified against this synthetic domain.
"""
import random
import os
import sys
from typing import Dict, FrozenSet, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, regardless of cwd
from env.core.base_types import create_tool_from_dict
from env.core.data_types import get_final_type


def generate_synthetic_domain(categories: List[str], seed: int, min_chain_length: int = 5,
                               max_chain_length: int = 8, min_cost: float = 15.0,
                               max_cost: float = 25.0) -> Tuple[FrozenSet[str], Dict]:
    """
    Build a synthetic domain: each category gets its own strictly linear
    chain of atomic tools, seeded by `seed` for reproducible randomized chain
    lengths (uniform integer in [min_chain_length, max_chain_length]) and
    costs (uniform in [min_cost, max_cost], CostBench-scale). Categories are
    fully independent -- no shared types, no cross-category prerequisites,
    and no bundle/shortcut tools of any kind. Each chain's final tool
    produces exactly get_final_type(category), so the category names must be
    real CostBench subtask names (location/transportation/accommodation/
    attraction/dining/shopping) for compatibility with the existing
    precompute functions, even though the chains themselves are entirely
    synthetic and unrelated to the real registry.

    Returns (initial_state, tools).
    """
    rng = random.Random(seed)
    initial_types = {"TimeInfo"}
    tools = {}

    for category in categories:
        chain_length = rng.randint(min_chain_length, max_chain_length)
        seed_type = f"{category.capitalize()}SyntheticSeed"
        initial_types.add(seed_type)

        prev_type = seed_type
        for step in range(1, chain_length + 1):
            is_last = (step == chain_length)
            output_type = get_final_type(category) if is_last else f"{category.capitalize()}_Synth_S{step}"
            tool_name = f"{category.capitalize()}_Synth_Step{step}"
            cost = round(rng.uniform(min_cost, max_cost), 2)
            tools[tool_name] = create_tool_from_dict({
                "type": "atomic",
                "name": tool_name,
                "input_types": [prev_type],
                "output_type": output_type,
                "cost": cost,
                "description": f"Synthetic atomic step {step}/{chain_length} for category {category}",
            })
            prev_type = output_type

    return frozenset(initial_types), tools
