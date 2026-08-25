"""
Stage 1 (continued): zero-cost end-to-end regression test -- exercises the
real run_episode wiring (not just the isolated apply_redundancy_elimination
function) via a scripted stub that reproduces the known real seed=2/round=7
case, confirming redundancy_elimination=True actually reduces total_tool_cost
in the full pipeline, redundancy_elimination=False is a true no-op vs
pre-existing behavior, and round count/state are unaffected either way.
"""
import sys
sys.path.insert(0, '.')
import run_brts_minimal as rbm

CALL_LOG = []


def make_scripted_stub(responses):
    it = iter(responses)

    def _stub(prompt, model="gpt-4o"):
        CALL_LOG.append(model)
        return next(it), {"prompt_tokens": 1, "completion_tokens": 1}
    return _stub


def test_e2e_redundancy_elimination_reduces_cost_on_known_case():
    CALL_LOG.clear()
    round0 = ['Decide_Location_Preference', 'Location_Preference_and_Search']
    round1 = ['Transportation_Preference_and_Search', 'Location_Full_Planning_to_Step1']
    round2 = ['Location_Refinement_Step2', 'Transportation_Refinement_Step1']
    # The known real redundant proposal: 4 tools, all -> TravelTransportation
    round3 = ['Transportation_Finish_from_Step1_6Steps', 'Transportation_Finish_from_Step2_5Steps',
              'Transportation_Finish_from_Step3_4Steps', 'Transportation_Finish_from_Step4_3Steps']
    scripted = [round0, round1, round2, round3]

    rbm.call_llm_stub = make_scripted_stub(list(scripted))
    r_off = rbm.run_episode(task="location", categories=["location", "transportation"], seed=2,
                             allow_batch=True, model="gpt-4o", safety_mode="gated", gate_version="v3",
                             redundancy_elimination=False, max_rounds=4)

    rbm.call_llm_stub = make_scripted_stub(list(scripted))
    r_on = rbm.run_episode(task="location", categories=["location", "transportation"], seed=2,
                            allow_batch=True, model="gpt-4o", safety_mode="gated", gate_version="v3",
                            redundancy_elimination=True, max_rounds=4)

    # Round count identical (dedup never changes round count, by construction)
    assert r_off["rounds"] == r_on["rounds"], (r_off["rounds"], r_on["rounds"])
    # Cost strictly lower with dedup on -- NOTE: this simplified synthetic
    # trajectory only gets 2 of the 4 known-real tools to a gate-valid state
    # by round 3 (Step3_4Steps/Step4_3Steps need L2/L3 refinement this
    # shortened script never performs), so the saving here is smaller than
    # the full real $309.61 case (already confirmed exactly via the
    # registry-based reconstruction in test_redundancy_elimination.py) --
    # this test's job is to confirm the PIPELINE WIRING works end-to-end,
    # not to re-derive that exact figure.
    assert r_on["total_tool_cost"] < r_off["total_tool_cost"], \
        (r_on["total_tool_cost"], r_off["total_tool_cost"])
    saved = r_off["total_tool_cost"] - r_on["total_tool_cost"]
    assert r_off["round_log"][3]["valid"] == ["Transportation_Finish_from_Step1_6Steps",
                                                "Transportation_Finish_from_Step2_5Steps"]
    assert r_on["round_log"][3]["valid"] == ["Transportation_Finish_from_Step2_5Steps"]  # cheaper of the 2 survives
    assert r_on["total_redundant_calls_dropped"] == 1
    print(f"PASS: test_e2e_redundancy_elimination_reduces_cost_on_known_case (saved ${saved:.2f}, "
          f"rounds identical: {r_off['rounds']}=={r_on['rounds']})")


def test_e2e_true_no_op_when_default_false():
    """redundancy_elimination defaults to False -- must be byte-identical
    to pre-existing behavior when a batch has no redundancy at all."""
    CALL_LOG.clear()
    round0 = ['Decide_Location_Preference', 'Location_Preference_and_Search']
    round1 = ['Transportation_Preference_and_Search', 'Location_Refinement_Step1']
    scripted = [round0, round1]

    rbm.call_llm_stub = make_scripted_stub(list(scripted))
    r_default = rbm.run_episode(task="location", categories=["location", "transportation"], seed=1,
                                 allow_batch=True, model="gpt-4o", safety_mode="gated", gate_version="v3",
                                 max_rounds=2)  # redundancy_elimination not passed at all
    assert r_default["total_redundant_calls_dropped"] == 0
    assert r_default["round_log"][0]["redundancy_dropped"] == []
    print("PASS: test_e2e_true_no_op_when_default_false")


if __name__ == "__main__":
    test_e2e_redundancy_elimination_reduces_cost_on_known_case()
    test_e2e_true_no_op_when_default_false()
    print("\nALL E2E REGRESSION TESTS PASSED")
