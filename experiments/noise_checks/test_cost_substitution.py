"""
Zero-cost synthetic verification of apply_cost_substitution, run BEFORE any
real data/API check. Pure Python, no API calls, no CostBench data -- just
constructs minimal fake Tool objects to test the substitution logic itself
in isolation.
"""
import sys
sys.path.insert(0, '.')
from collections import namedtuple
from brts_core import apply_cost_substitution

Tool = namedtuple("Tool", ["input_types", "output_type", "cost"])


def test_substitutes_cheaper_same_output():
    tools = {
        "ExpensiveLocationFinder": Tool(input_types=[], output_type="LocationX", cost=50),
        "CheapLocationFinder": Tool(input_types=[], output_type="LocationX", cost=10),
        "Unrelated": Tool(input_types=[], output_type="Other", cost=5),
    }
    state = frozenset()
    visible = list(tools.keys())
    valid = ["ExpensiveLocationFinder"]

    substituted, subs = apply_cost_substitution(valid, state, tools, visible)

    assert substituted == ["CheapLocationFinder"], substituted
    assert len(subs) == 1
    assert subs[0]["slot"] == "ExpensiveLocationFinder"
    assert subs[0]["replaced_with"] == "CheapLocationFinder"
    assert subs[0]["cost_saved"] == 40
    assert len(substituted) == len(valid)  # batch size unchanged
    print("PASS: test_substitutes_cheaper_same_output")


def test_no_op_when_no_cheaper_alternative_exists():
    tools = {
        "OnlyLocationFinder": Tool(input_types=[], output_type="LocationX", cost=10),
        "Unrelated": Tool(input_types=[], output_type="Other", cost=5),
    }
    state = frozenset()
    visible = list(tools.keys())
    valid = ["OnlyLocationFinder"]

    substituted, subs = apply_cost_substitution(valid, state, tools, visible)

    assert substituted == valid, substituted
    assert subs == []
    print("PASS: test_no_op_when_no_cheaper_alternative_exists")


def test_no_op_when_cheaper_alternative_not_visible():
    # A cheaper tool with the same output_type exists in the FULL registry,
    # but is not in `visible` this round -- must not be substituted in,
    # since the spec is "already-visible, currently-satisfiable" only.
    tools = {
        "ExpensiveLocationFinder": Tool(input_types=[], output_type="LocationX", cost=50),
        "CheapButNotVisible": Tool(input_types=[], output_type="LocationX", cost=10),
    }
    state = frozenset()
    visible = ["ExpensiveLocationFinder"]  # cheaper one deliberately excluded
    valid = ["ExpensiveLocationFinder"]

    substituted, subs = apply_cost_substitution(valid, state, tools, visible)

    assert substituted == valid, substituted
    assert subs == []
    print("PASS: test_no_op_when_cheaper_alternative_not_visible")


def test_no_op_when_cheaper_alternative_not_currently_satisfiable():
    # Same output_type, cheaper, and visible -- but its own inputs aren't
    # satisfiable at this point in the batch, so it must NOT be substituted.
    tools = {
        "ExpensiveLocationFinder": Tool(input_types=[], output_type="LocationX", cost=50),
        "CheapButNeedsUnmetInput": Tool(input_types=["SomethingNotYetAvailable"], output_type="LocationX", cost=10),
    }
    state = frozenset()
    visible = list(tools.keys())
    valid = ["ExpensiveLocationFinder"]

    substituted, subs = apply_cost_substitution(valid, state, tools, visible)

    assert substituted == valid, substituted
    assert subs == []
    print("PASS: test_no_op_when_cheaper_alternative_not_currently_satisfiable")


def test_never_substitutes_a_tied_cost_alternative():
    # Equal cost -- no benefit, must not churn the batch for a no-op swap.
    tools = {
        "OriginalLocationFinder": Tool(input_types=[], output_type="LocationX", cost=10),
        "SameCostAlternative": Tool(input_types=[], output_type="LocationX", cost=10),
    }
    state = frozenset()
    visible = list(tools.keys())
    valid = ["OriginalLocationFinder"]

    substituted, subs = apply_cost_substitution(valid, state, tools, visible)

    assert substituted == valid, substituted
    assert subs == []
    print("PASS: test_never_substitutes_a_tied_cost_alternative")


def test_preserves_batch_size_and_order_with_multiple_slots():
    tools = {
        "ExpensiveA": Tool(input_types=[], output_type="TypeA", cost=50),
        "CheapA": Tool(input_types=[], output_type="TypeA", cost=5),
        "ExpensiveB": Tool(input_types=[], output_type="TypeB", cost=40),
        "CheapB": Tool(input_types=[], output_type="TypeB", cost=8),
        "OnlyC": Tool(input_types=[], output_type="TypeC", cost=20),
    }
    state = frozenset()
    visible = list(tools.keys())
    valid = ["ExpensiveA", "OnlyC", "ExpensiveB"]  # deliberately out-of-alpha order

    substituted, subs = apply_cost_substitution(valid, state, tools, visible)

    assert substituted == ["CheapA", "OnlyC", "CheapB"], substituted  # same order, same length
    assert len(substituted) == len(valid)
    assert len(subs) == 2
    print("PASS: test_preserves_batch_size_and_order_with_multiple_slots")


def test_state_and_round_count_invariant_under_substitution():
    # The actual safety property: apply_batch's resulting state must be
    # IDENTICAL whether or not substitution happened, since output_types
    # are guaranteed identical -- this is what guarantees round count can
    # never change.
    from brts_core import apply_batch
    tools = {
        "ExpensiveLocationFinder": Tool(input_types=[], output_type="LocationX", cost=50),
        "CheapLocationFinder": Tool(input_types=[], output_type="LocationX", cost=10),
    }
    state = frozenset({"Start"})
    visible = list(tools.keys())
    valid = ["ExpensiveLocationFinder"]

    substituted, subs = apply_cost_substitution(valid, state, tools, visible)
    assert subs  # sanity: a real substitution did happen in this case

    state_without_sub = apply_batch(state, tools, valid)
    state_with_sub = apply_batch(state, tools, substituted)
    assert state_without_sub == state_with_sub, (state_without_sub, state_with_sub)
    print("PASS: test_state_and_round_count_invariant_under_substitution")


def test_chain_aware_substitution_uses_incremental_available_types():
    # A cheaper alternative for a LATER slot depends on an EARLIER slot's
    # output -- confirms the pass evaluates "currently satisfiable" using
    # the incremental available_types set as it walks the batch, not just
    # the static pre-round state.
    tools = {
        "StepOne": Tool(input_types=[], output_type="TypeA", cost=10),
        "ExpensiveStepTwo": Tool(input_types=["TypeA"], output_type="TypeB", cost=50),
        "CheapStepTwo": Tool(input_types=["TypeA"], output_type="TypeB", cost=5),
    }
    state = frozenset()
    visible = list(tools.keys())
    valid = ["StepOne", "ExpensiveStepTwo"]

    substituted, subs = apply_cost_substitution(valid, state, tools, visible)

    assert substituted == ["StepOne", "CheapStepTwo"], substituted
    assert len(subs) == 1
    print("PASS: test_chain_aware_substitution_uses_incremental_available_types")


if __name__ == "__main__":
    test_substitutes_cheaper_same_output()
    test_no_op_when_no_cheaper_alternative_exists()
    test_no_op_when_cheaper_alternative_not_visible()
    test_no_op_when_cheaper_alternative_not_currently_satisfiable()
    test_never_substitutes_a_tied_cost_alternative()
    test_preserves_batch_size_and_order_with_multiple_slots()
    test_state_and_round_count_invariant_under_substitution()
    test_chain_aware_substitution_uses_incremental_available_types()
    print("\nALL SYNTHETIC TESTS PASSED")
