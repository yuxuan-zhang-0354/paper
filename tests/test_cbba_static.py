import numpy as np

from uav_lifecycle.allocation_model import StaticAgent, StaticInstance, StaticTask
from uav_lifecycle.cbba_static import (
    AgentBundle,
    BundleEntry,
    CBBAResult,
    Method,
    Winner,
    build_bundle,
    run_static_cbba,
    validate_result,
)


def instance_for_warp(values=(10.0, 30.0, 20.0), deadline=20.0):
    return StaticInstance(
        agents=(StaticAgent(0, (0.0, 0.0), deadline, 3),),
        tasks=tuple(
            StaticTask(j, (float(j + 1), 0.0), 0.0, value) for j, value in enumerate(values)
        ),
        discount=0.9,
    )


def test_empty_bundle_warp_equals_raw_and_external_bids_never_increase():
    state = build_bundle(instance_for_warp(), 0, Method.JOHNSON_WARPED, (None, None, None))
    assert state.entries[0].external_bid == state.entries[0].raw_score
    external = [entry.external_bid for entry in state.entries]
    assert all(left >= right - 1e-12 for left, right in zip(external, external[1:]))


def test_warped_bid_controls_eligibility_but_raw_score_controls_choice():
    # Task 0 wins first. Thereafter task 1 has the larger raw score but is
    # ineligible after warping against threshold 9; task 2 remains eligible.
    winners = (None, Winner(1, 25.0), Winner(1, 1.0))
    state = build_bundle(instance_for_warp((100.0, 30.0, 20.0)), 0, Method.JOHNSON_WARPED, winners)
    assert state.bundle[:2] == (0, 2)


def test_full_rebuild_withdraws_self_winner_but_keeps_foreign_threshold():
    winners = (Winner(0, 1_000.0), Winner(1, 1_000.0), None)
    state = build_bundle(instance_for_warp(), 0, Method.JOHNSON_WARPED, winners)
    assert 0 in state.bundle
    assert 1 not in state.bundle


def test_rebuild_is_exactly_replayable():
    winners = (Winner(1, 2.0), None, None)
    first = build_bundle(instance_for_warp(), 0, Method.JOHNSON_WARPED, winners)
    second = build_bundle(instance_for_warp(), 0, Method.JOHNSON_WARPED, winners)
    assert first == second


def test_all_methods_return_valid_result_on_small_static_instance():
    instance = StaticInstance(
        agents=(
            StaticAgent(0, (-2.0, 0.0), 12.0, 2),
            StaticAgent(1, (4.0, 0.0), 12.0, 2),
        ),
        tasks=tuple(StaticTask(j, (float(j), 0.0), 0.5, 10.0 + j) for j in range(3)),
        discount=0.95,
    )
    for method in Method:
        result = run_static_cbba(instance, method)
        report = validate_result(instance, result)
        assert report["winner_conflicts"] == 0
        assert report["infeasible_paths"] == 0
        assert report["bundle_path_mismatches"] == 0
        if method is Method.JOHNSON_WARPED:
            assert result.status == "converged"
            assert report["warped_monotonicity_violations"] == 0
            assert report["replay_mismatches"] == 0


def test_surrogate_conservative_weight_filters_deadline():
    instance = StaticInstance(
        agents=(StaticAgent(0, (0.0, 0.0), 3.0, 3),),
        tasks=(StaticTask(0, (1.0, 0.0), 0.0, 10.0), StaticTask(1, (2.0, 0.0), 0.0, 20.0)),
        discount=0.95,
    )
    state = build_bundle(instance, 0, Method.SURROGATE, (None, None))
    assert state.bundle == (0,)
    np.testing.assert_allclose(state.entries[0].external_bid, 10.0 * 0.95)


def test_warped_monotonicity_metric_is_not_applied_to_raw_methods():
    instance = instance_for_warp()
    state = AgentBundle(
        0,
        (0, 1),
        (0, 1),
        (BundleEntry(0, 1.0, 1.0), BundleEntry(1, 2.0, 2.0)),
    )
    result = CBBAResult(
        Method.STANDARD_RAW,
        "converged",
        1,
        (Winner(0, 1.0), Winner(0, 2.0), None),
        (state,),
        3.0,
    )
    assert validate_result(instance, result)["warped_monotonicity_violations"] == 0
