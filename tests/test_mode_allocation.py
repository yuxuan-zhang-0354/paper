import numpy as np

from uav_lifecycle.belief import bda_kernel, recon_kernel
from uav_lifecycle.mode_allocation import (
    Mode,
    ModeAgent,
    ModeInstance,
    ModeTask,
    TickPathTiming,
    best_mode_insertion,
    evaluate_mode_path,
    mode_utilities,
)
from uav_lifecycle.rollout import RolloutParameters, action_values


PARAMS = RolloutParameters(100, 30, 0.4, 0.75, 4, 2, 1.5, 2, 6, 1, 0.02)
ZR = recon_kernel([[0.65, 0.15], [0.35, 0.85]], [[0.75, 0.25], [0.25, 0.75]])
ZB = bda_kernel([[0.92, 0.06], [0.08, 0.94]])


def instance(horizon=20.0, distance_budget=20.0, ammo=1):
    return ModeInstance(
        agents=(ModeAgent(0, (0.0, 0.0), horizon, distance_budget, ammo),),
        tasks_by_target=(
            (ModeTask(0, Mode.RECON, (1.0, 0.0), 4.0, 0, 20.0),
             ModeTask(0, Mode.ATTACK, (1.0, 0.0), 2.0, 1, 25.0),
             ModeTask(0, Mode.BDA, (1.0, 0.0), 1.5, 0, 15.0)),
            (ModeTask(1, Mode.RECON, (3.0, 0.0), 4.0, 0, 10.0),
             ModeTask(1, Mode.ATTACK, (3.0, 0.0), 2.0, 1, 12.0),
             ModeTask(1, Mode.BDA, (3.0, 0.0), 1.5, 0, 8.0)),
        ),
        beta=0.02,
        distance_cost=1.0,
        ammo_cost=2.0,
    )


def test_path_score_accounts_for_start_discount_travel_and_ammo():
    result = evaluate_mode_path(instance(), 0, ((0, Mode.ATTACK),))
    np.testing.assert_allclose(result.score, np.exp(-0.02) * 25.0 - 1.0 - 2.0)
    assert result.completion_time == 3.0
    assert result.distance == 1.0
    assert result.ammo_used == 1
    assert result.feasible


def test_horizon_range_ammo_and_target_uniqueness_are_hard_constraints():
    assert not evaluate_mode_path(instance(horizon=2.9), 0, ((0, Mode.ATTACK),)).feasible
    assert not evaluate_mode_path(instance(distance_budget=0.9), 0, ((0, Mode.RECON),)).feasible
    assert not evaluate_mode_path(instance(ammo=0), 0, ((0, Mode.ATTACK),)).feasible
    assert not evaluate_mode_path(instance(), 0, ((0, Mode.RECON), (0, Mode.BDA))).feasible


def test_best_insertion_uses_earliest_position_on_tie():
    flat = ModeInstance(
        agents=(ModeAgent(0, (0.0, 0.0), 10, 10, 0),),
        tasks_by_target=(
            (ModeTask(0, Mode.RECON, (0.0, 0.0), 0, 0, 1),),
            (ModeTask(1, Mode.RECON, (0.0, 0.0), 0, 0, 1),),
        ), beta=0, distance_cost=0, ammo_cost=0,
    )
    insertion = best_mode_insertion(flat, 0, ((0, Mode.RECON),), (1, Mode.RECON))
    assert insertion.position == 0


def test_utility_variants_share_validated_optimistic_values():
    belief = np.array([0.4, 0.1, 0.3, 0.2])
    optimistic = mode_utilities(belief, ZR, ZB, PARAMS, "optimistic")
    expected = action_values(belief, ZR, ZB, PARAMS)
    assert optimistic == {Mode.RECON: expected["recon"], Mode.ATTACK: expected["attack"], Mode.BDA: expected["bda"]}
    no_cont = mode_utilities(belief, ZR, ZB, PARAMS, "no_continuation")
    gated = mode_utilities(belief, ZR, ZB, PARAMS, "ammo_reachability_gate", {mode: False for mode in Mode})
    assert gated == no_cont
    assert no_cont[Mode.RECON] == -PARAMS.cost_r
    assert no_cont[Mode.BDA] == -PARAMS.cost_b


def test_tick_path_timing_uses_absolute_half_even_ticks_without_changing_soft_costs():
    dynamic = ModeInstance(
        agents=(ModeAgent(0, (0.0, 0.0), 999.0, 2.0, 1),),
        tasks_by_target=((ModeTask(0, Mode.ATTACK, (0.25, 0.0), 0.25, 1, 10.0),),),
        beta=1.0,
        distance_cost=2.0,
        ammo_cost=3.0,
        timing=TickPathTiming(epoch_tick=2, max_tick=3, tick_size=0.25, speed=1.0),
    )
    result = evaluate_mode_path(dynamic, 0, ((0, Mode.ATTACK),))
    assert result.start_ticks == (3,)
    assert result.completion_tick == 4
    assert result.start_times == (0.75,)
    assert result.completion_time == 1.0
    np.testing.assert_allclose(result.score, np.exp(-0.75) * 10.0 - 0.5 - 3.0)
    assert not result.feasible


def test_legacy_path_evaluation_fields_remain_unchanged_without_tick_timing():
    result = evaluate_mode_path(instance(), 0, ((0, Mode.ATTACK),))
    assert result.start_ticks is None
    assert result.completion_tick is None
    assert result.start_times == (1.0,)
    assert result.completion_time == 3.0
