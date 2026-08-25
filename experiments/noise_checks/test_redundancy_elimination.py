"""
Stage 1 of the redundancy-elimination verification: zero-cost synthetic
tests, before any real spend or even any real-log scanning. Same pattern
as cost-substitution and state-caching: prove the mechanism's own claimed
properties in isolation first.
"""
import sys
sys.path.insert(0, '.')
from collections import namedtuple
from brts_core import apply_redundancy_elimination, apply_batch

Tool = namedtuple("Tool", ["input_types", "output_type", "cost"])


def test_drops_all_but_cheapest_same_output():
    tools = {
        "ExpensiveFinish": Tool(input_types=[], output_type="TravelTransportation", cost=123.31),
        "MidFinish": Tool(input_types=[], output_type="TravelTransportation", cost=103.56),
        "CheapFinish": Tool(input_types=[], output_type="TravelTransportation", cost=62.26),
        "Unrelated": Tool(input_types=[], output_type="Other", cost=5),
    }
    batch = ["ExpensiveFinish", "MidFinish", "CheapFinish", "Unrelated"]

    deduped, dropped = apply_redundancy_elimination(batch, tools)

    assert deduped == ["CheapFinish", "Unrelated"], deduped
    assert len(dropped) == 2
    dropped_names = {d["name"] for d in dropped}
    assert dropped_names == {"ExpensiveFinish", "MidFinish"}
    assert all(d["kept"] == "CheapFinish" for d in dropped)
    print("PASS: test_drops_all_but_cheapest_same_output")


def test_true_no_op_when_no_duplicate_output_types():
    tools = {
        "A": Tool(input_types=[], output_type="TypeA", cost=10),
        "B": Tool(input_types=[], output_type="TypeB", cost=20),
        "C": Tool(input_types=[], output_type="TypeC", cost=30),
    }
    batch = ["A", "B", "C"]
    deduped, dropped = apply_redundancy_elimination(batch, tools)
    assert deduped == batch
    assert dropped == []
    print("PASS: test_true_no_op_when_no_duplicate_output_types")


def test_cost_never_increases_and_can_decrease():
    tools = {
        "Expensive": Tool(input_types=[], output_type="TypeA", cost=50),
        "Cheap": Tool(input_types=[], output_type="TypeA", cost=10),
    }
    batch = ["Expensive", "Cheap"]
    deduped, dropped = apply_redundancy_elimination(batch, tools)
    original_cost = sum(tools[n].cost for n in batch)
    deduped_cost = sum(tools[n].cost for n in deduped)
    assert deduped_cost <= original_cost
    assert deduped_cost == 10 < original_cost == 60
    print("PASS: test_cost_never_increases_and_can_decrease")


def test_final_state_identical_with_or_without_dedup():
    """The key safety property: apply_batch's resulting state (a SET of
    output_types) must be IDENTICAL whether duplicates are dropped or not,
    since dropping only ever removes an ALREADY-represented output_type."""
    tools = {
        "ExpensiveFinish": Tool(input_types=[], output_type="TravelTransportation", cost=123.31),
        "CheapFinish": Tool(input_types=[], output_type="TravelTransportation", cost=62.26),
        "OtherTool": Tool(input_types=[], output_type="TravelLocation", cost=40),
    }
    batch = ["ExpensiveFinish", "CheapFinish", "OtherTool"]
    deduped, dropped = apply_redundancy_elimination(batch, tools)
    assert dropped  # sanity: a real drop did happen

    state = frozenset({"Start"})
    state_without_dedup = apply_batch(state, tools, batch)
    state_with_dedup = apply_batch(state, tools, deduped)
    assert state_without_dedup == state_with_dedup, (state_without_dedup, state_with_dedup)
    print("PASS: test_final_state_identical_with_or_without_dedup")


def test_known_real_case_seed2_round7():
    """The exact real case found live: 4 real tools, real costs (from the
    seed=2 registry), all producing TravelTransportation. Confirms the
    detection logic actually catches the concrete known-positive case
    before scanning the full logs for it."""
    tools = {
        "Transportation_Finish_from_Step1_6Steps": Tool(
            input_types=["TimeInfo", "TransportationCandidate_Raw"], output_type="TravelTransportation", cost=123.31),
        "Transportation_Finish_from_Step2_5Steps": Tool(
            input_types=["TransportationCandidate_L1_Available"], output_type="TravelTransportation", cost=103.56),
        "Transportation_Finish_from_Step3_4Steps": Tool(
            input_types=["TransportationCandidate_L2_Located"], output_type="TravelTransportation", cost=82.74),
        "Transportation_Finish_from_Step4_3Steps": Tool(
            input_types=["TransportationCandidate_L3_Reviewed"], output_type="TravelTransportation", cost=62.26),
    }
    batch = list(tools.keys())  # exact order as proposed in the real log
    deduped, dropped = apply_redundancy_elimination(batch, tools)

    assert deduped == ["Transportation_Finish_from_Step4_3Steps"], deduped
    assert len(dropped) == 3
    total_saved = sum(d["cost_saved"] for d in dropped)
    assert abs(total_saved - 309.61) < 0.01, total_saved
    print(f"PASS: test_known_real_case_seed2_round7 (real waste this round would have been: ${total_saved:.2f})")


def test_duplicate_identical_tool_name_collapses_to_one():
    """Edge case: the exact same tool name appears twice in the batch (not
    just same output_type, literally the same tool) -- must collapse to a
    single instance, not be treated as two separate 'ties'."""
    tools = {
        "SameTool": Tool(input_types=[], output_type="TypeA", cost=10),
    }
    batch = ["SameTool", "SameTool"]
    deduped, dropped = apply_redundancy_elimination(batch, tools)
    assert deduped == ["SameTool"], deduped
    assert len(dropped) == 1
    assert dropped[0]["name"] == "SameTool" and dropped[0]["kept"] == "SameTool"
    print("PASS: test_duplicate_identical_tool_name_collapses_to_one")


if __name__ == "__main__":
    test_drops_all_but_cheapest_same_output()
    test_true_no_op_when_no_duplicate_output_types()
    test_cost_never_increases_and_can_decrease()
    test_final_state_identical_with_or_without_dedup()
    test_known_real_case_seed2_round7()
    test_duplicate_identical_tool_name_collapses_to_one()
    print("\nALL SYNTHETIC TESTS PASSED")
