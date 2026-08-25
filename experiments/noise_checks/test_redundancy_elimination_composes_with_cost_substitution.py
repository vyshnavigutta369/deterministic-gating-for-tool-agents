"""
Stage 4 of the redundancy-elimination verification: zero-cost synthetic
tests first, confirming the two independently-safe mechanisms (already
each verified alone) genuinely compose without interference in the
"dedup first, then substitute" order run_episode already wires
(redundancy_elimination applied to `valid`, then cost_substitution applied
to the result -- see run_brts_minimal.py).
"""
import sys
sys.path.insert(0, '.')
from collections import namedtuple
from brts_core import apply_redundancy_elimination, apply_cost_substitution, apply_batch

Tool = namedtuple("Tool", ["input_types", "output_type", "cost"])


def test_dedup_then_substitute_beats_either_alone():
    """A batch with BOTH a redundant duplicate (dedup's job) AND a
    substitutable non-duplicate slot (cost-substitution's job) -- applying
    both, in order, must fire both mechanisms and reach a cost at or below
    what either alone would achieve."""
    tools = {
        "ExpensiveFinish": Tool(input_types=[], output_type="TravelTransportation", cost=123.31),
        "MidFinish": Tool(input_types=[], output_type="TravelTransportation", cost=103.56),
        "CheapFinish": Tool(input_types=[], output_type="TravelTransportation", cost=62.26),
        "ExpensiveLocationFinder": Tool(input_types=[], output_type="LocationX", cost=50),
        "CheapLocationFinder": Tool(input_types=[], output_type="LocationX", cost=10),
    }
    state = frozenset()
    visible = list(tools.keys())
    valid = ["ExpensiveFinish", "MidFinish", "ExpensiveLocationFinder"]  # 2 redundant + 1 substitutable
    original_cost = sum(tools[n].cost for n in valid)

    deduped, dropped = apply_redundancy_elimination(valid, tools)
    assert deduped == ["MidFinish", "ExpensiveLocationFinder"], deduped  # ExpensiveFinish is pricier, dropped
    assert len(dropped) == 1 and dropped[0]["name"] == "ExpensiveFinish"

    substituted, subs = apply_cost_substitution(deduped, state, tools, visible)
    # MidFinish (the dedup survivor) is itself NOT the cheapest TravelTransportation
    # tool overall -- CheapFinish is visible and cheaper, so cost-substitution swaps
    # it in, on top of dedup already having removed the redundant duplicate:
    assert substituted == ["CheapFinish", "CheapLocationFinder"], substituted
    final_cost = sum(tools[n].cost for n in substituted)

    assert final_cost < original_cost
    dedup_only_cost = sum(tools[n].cost for n in deduped)
    assert final_cost <= dedup_only_cost, "composing must never be worse than dedup alone"
    print(f"PASS: test_dedup_then_substitute_beats_either_alone "
          f"(original=${original_cost}, dedup_only=${dedup_only_cost}, composed=${final_cost})")


def test_dedup_removes_slot_substitution_would_have_also_fixed():
    """When ALL duplicates share output_type with each other (the pure
    redundancy case), dedup alone already reaches the cheapest single
    survivor -- cost-substitution then has nothing further to improve on
    that slot (no cheaper VISIBLE alternative beyond what's already
    kept), confirming no double-counting or conflict between the two
    passes when their job overlaps on the exact same output_type."""
    tools = {
        "Step1": Tool(input_types=[], output_type="TravelTransportation", cost=123.31),
        "Step2": Tool(input_types=[], output_type="TravelTransportation", cost=103.56),
        "Step3": Tool(input_types=[], output_type="TravelTransportation", cost=82.74),
        "Step4": Tool(input_types=[], output_type="TravelTransportation", cost=62.26),  # cheapest
    }
    state = frozenset()
    visible = list(tools.keys())
    valid = ["Step1", "Step2", "Step3", "Step4"]

    deduped, dropped = apply_redundancy_elimination(valid, tools)
    assert deduped == ["Step4"]
    assert len(dropped) == 3

    substituted, subs = apply_cost_substitution(deduped, state, tools, visible)
    assert substituted == ["Step4"], substituted  # already cheapest -- no further substitution possible
    assert subs == []
    print("PASS: test_dedup_removes_slot_substitution_would_have_also_fixed")


def test_composed_state_still_identical_to_neither_applied():
    """The combined safety property: apply_batch's resulting state must be
    IDENTICAL whether neither, either, or both mechanisms ran -- both
    individually preserve the output_type SET, so composing them must too."""
    tools = {
        "ExpensiveFinish": Tool(input_types=[], output_type="TravelTransportation", cost=123.31),
        "MidFinish": Tool(input_types=[], output_type="TravelTransportation", cost=103.56),
        "ExpensiveLocationFinder": Tool(input_types=[], output_type="LocationX", cost=50),
        "CheapLocationFinder": Tool(input_types=[], output_type="LocationX", cost=10),
    }
    state = frozenset({"Start"})
    visible = list(tools.keys())
    valid = ["ExpensiveFinish", "MidFinish", "ExpensiveLocationFinder"]

    state_neither = apply_batch(state, tools, valid)

    deduped, _ = apply_redundancy_elimination(valid, tools)
    state_dedup_only = apply_batch(state, tools, deduped)

    substituted, _ = apply_cost_substitution(valid, state, tools, visible)
    state_sub_only = apply_batch(state, tools, substituted)

    deduped2, _ = apply_redundancy_elimination(valid, tools)
    composed, _ = apply_cost_substitution(deduped2, state, tools, visible)
    state_composed = apply_batch(state, tools, composed)

    assert state_neither == state_dedup_only == state_sub_only == state_composed, \
        (state_neither, state_dedup_only, state_sub_only, state_composed)
    print("PASS: test_composed_state_still_identical_to_neither_applied")


if __name__ == "__main__":
    test_dedup_then_substitute_beats_either_alone()
    test_dedup_removes_slot_substitution_would_have_also_fixed()
    test_composed_state_still_identical_to_neither_applied()
    print("\nALL SYNTHETIC COMPOSITION TESTS PASSED")
