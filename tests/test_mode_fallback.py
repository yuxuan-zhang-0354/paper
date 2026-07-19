from dataclasses import FrozenInstanceError, replace

import pytest

from uav_lifecycle.mode_allocation import TOL, Mode, ModeAgent, ModeInstance, ModeTask
from uav_lifecycle.mode_cbba import (
    ModeCBBAResult,
    ModeMethod,
    ScreenedTask,
    screen_modes,
)
from uav_lifecycle.mode_fallback import (
    FallbackIteration,
    active_screening,
    finalize_fallback_attempts,
    rank_mode_candidates,
    run_ranked_fallback,
    select_fallback_iteration,
)
import uav_lifecycle.mode_fallback as mode_fallback


def task(target_id, mode, utility, ammo=0):
    return ModeTask(target_id, mode, (0, 0), 0, ammo, utility)


def ranked_witness_instance():
    return ModeInstance(
        agents=(
            ModeAgent(0, (0, 0), 10, 10, 0),
            ModeAgent(1, (0, 0), 10, 10, 0),
        ),
        tasks_by_target=(
            (
                task(0, Mode.BDA, 5),
                task(0, Mode.ATTACK, 5),
                task(0, Mode.RECON, 5),
            ),
            (
                task(1, Mode.RECON, TOL),
                task(1, Mode.ATTACK, 10, ammo=1),
                task(1, Mode.BDA, -1),
            ),
        ),
        beta=0,
        distance_cost=0,
        ammo_cost=0,
    )


def test_rank_zero_exactly_matches_frozen_screening():
    instance = ranked_witness_instance()
    ranked = rank_mode_candidates(instance)
    frozen = screen_modes(instance)
    assert frozen
    assert tuple(
        None
        if not row
        else ScreenedTask(
            row[0].target_id,
            row[0].mode,
            row[0].witness_agent,
            row[0].witness_value,
        )
        for row in ranked
    ) == frozen


def test_rank_zero_frozen_equivalence_checks_witness_fields(monkeypatch):
    instance = ranked_witness_instance()
    frozen = screen_modes(instance)
    changed = replace(frozen[0], witness_agent=1)
    monkeypatch.setattr(
        mode_fallback,
        "screen_modes",
        lambda ignored: (changed, *frozen[1:]),
    )

    with pytest.raises(AssertionError):
        rank_mode_candidates(instance)


def test_ranking_keeps_only_positive_feasible_modes_and_fixed_ties():
    ranked = rank_mode_candidates(ranked_witness_instance())
    assert [item.mode for item in ranked[0]] == [Mode.RECON, Mode.ATTACK, Mode.BDA]
    assert ranked[1] == ()
    assert all(item.witness_value > TOL for row in ranked for item in row)
    assert all(item.witness_agent == 0 for item in ranked[0])


def test_active_screening_materializes_ranked_pointer_as_frozen_screened_task():
    rankings = rank_mode_candidates(ranked_witness_instance())
    screened = active_screening(rankings, (1, 0))
    assert screened == (ScreenedTask(0, Mode.ATTACK, 0, 5), None)
    with pytest.raises(FrozenInstanceError):
        screened[0].mode = Mode.BDA


@pytest.mark.parametrize("pointers", [(), (0,), (0, 0, 0), (-1, 0), (4, 0), (0, 1)])
def test_active_screening_rejects_invalid_pointer_vectors(pointers):
    rankings = rank_mode_candidates(ranked_witness_instance())
    with pytest.raises(ValueError):
        active_screening(rankings, pointers)


def no_orphan_instance():
    return ModeInstance(
        agents=(ModeAgent(0, (0, 0), 10, 10, 0),),
        tasks_by_target=((task(0, Mode.RECON, 5),),),
        beta=0,
        distance_cost=0,
        ammo_cost=0,
        instance_id="no-orphan",
    )


def test_no_orphan_stops_after_base_iteration():
    result = run_ranked_fallback(no_orphan_instance())
    assert len(result.iterations) == 1
    assert result.base_orphans == ()
    assert result.total_johnson_calls == 1


def second_mode_resolution_instance():
    return ModeInstance(
        agents=(ModeAgent(0, (0, 0), 10, 10, 1),),
        tasks_by_target=(
            (task(0, Mode.ATTACK, 10, ammo=1),),
            (
                task(1, Mode.ATTACK, 9, ammo=1),
                task(1, Mode.RECON, 8),
            ),
        ),
        beta=0,
        distance_cost=0,
        ammo_cost=0,
        instance_id="second-mode-resolution",
    )


def test_orphan_advances_to_second_mode_and_is_assigned():
    result = run_ranked_fallback(second_mode_resolution_instance())
    assert result.base_targets == (0, 1)
    assert result.base_orphans == (1,)
    assert result.resolved_targets == (1,)
    assert result.selected_assigned_modes[1] is Mode.RECON
    assert result.iterations[1].assigned_modes == (Mode.ATTACK, Mode.RECON)


def exhaustion_instance():
    return ModeInstance(
        agents=(ModeAgent(0, (0, 0), 10, 10, 1),),
        tasks_by_target=(
            (task(0, Mode.ATTACK, 10, ammo=1),),
            (task(1, Mode.ATTACK, 9, ammo=1),),
        ),
        beta=0,
        distance_cost=0,
        ammo_cost=0,
        instance_id="exhaustion",
    )


def test_exhausted_candidates_end_as_defer_but_remain_in_denominator():
    result = run_ranked_fallback(exhaustion_instance())
    assert result.base_targets == (0, 1)
    assert result.base_orphans == (1,)
    assert 1 in result.fallback_unresolved
    assert 1 in result.search_exhausted_targets
    assert result.iterations[-1].screening[1] is None


def test_base_orphan_remaining_unresolved_is_not_a_selected_defer_change():
    result = run_ranked_fallback(exhaustion_instance())

    assert result.base_orphans == (1,)
    assert result.fallback_unresolved == (1,)
    assert result.selected_defers == ()


def test_assigned_in_base_becoming_unassigned_is_a_selected_defer_change():
    instance = no_orphan_instance()
    rankings = rank_mode_candidates(instance)
    base = replace(
        synthetic_iteration(0, 1.0, (Mode.RECON,)),
        screening=(ScreenedTask(0, Mode.RECON, 0, 5),),
    )
    selected = replace(
        synthetic_iteration(1, 2.0, (None,)),
        screening=(None,),
        pointers=(1,),
    )

    result = finalize_fallback_attempts((base, selected), rankings)

    assert result.selected_iteration == 1
    assert result.selected_defers == (0,)


def orphan_exchange_instance():
    return ModeInstance(
        agents=(ModeAgent(0, (5, 0), 12, 10, 1),),
        tasks_by_target=(
            (
                ModeTask(0, Mode.RECON, (5, 5), 1, 0, 8),
                ModeTask(0, Mode.ATTACK, (5, 5), 0, 0, 3),
            ),
            (
                ModeTask(1, Mode.RECON, (8, 0), 0, 0, 5),
                ModeTask(1, Mode.ATTACK, (0, 5), 0, 1, 8),
            ),
            (
                ModeTask(2, Mode.RECON, (0, 0), 2, 1, 8),
                ModeTask(2, Mode.ATTACK, (5, 0), 0, 1, 5),
            ),
        ),
        beta=0.05,
        distance_cost=0,
        ammo_cost=0,
        instance_id="orphan-exchange",
    )


def test_new_orphan_created_by_switch_advances_in_later_iteration():
    result = run_ranked_fallback(orphan_exchange_instance())
    assert result.search_advances >= 2
    assert len(result.iterations) >= 3
    assert result.iterations[0].active_orphans == (2,)
    assert result.iterations[1].active_orphans == (1,)
    assert all(
        sum(right.pointers) > sum(left.pointers)
        for left, right in zip(result.iterations, result.iterations[1:])
    )


def two_orphan_instance():
    return ModeInstance(
        agents=(ModeAgent(0, (0, 0), 10, 10, 1),),
        tasks_by_target=(
            (task(0, Mode.ATTACK, 10, ammo=1),),
            (task(1, Mode.ATTACK, 9, ammo=1), task(1, Mode.RECON, 2)),
            (task(2, Mode.ATTACK, 8, ammo=1), task(2, Mode.RECON, 1)),
        ),
        beta=0,
        distance_cost=0,
        ammo_cost=0,
        instance_id="two-orphan",
    )


def test_multiple_orphans_advance_synchronously():
    result = run_ranked_fallback(two_orphan_instance())
    assert result.iterations[0].active_orphans == (1, 2)
    assert result.iterations[1].pointers == (0, 1, 1)


def lower_score_fallback_instance():
    return ModeInstance(
        agents=(
            ModeAgent(0, (0, 0), 8, 15, 2),
            ModeAgent(1, (8, 0), 12, 20, 3),
        ),
        tasks_by_target=(
            (
                ModeTask(0, Mode.RECON, (2, 0), 0, 2, 7),
                ModeTask(0, Mode.ATTACK, (8, 8), 0, 0, 7),
                ModeTask(0, Mode.BDA, (2, 0), 2, 2, 15),
            ),
            (
                ModeTask(1, Mode.RECON, (5, 5), 2, 0, 15),
                ModeTask(1, Mode.ATTACK, (0, 0), 0, 2, 15),
                ModeTask(1, Mode.BDA, (8, 8), 2, 1, 2),
            ),
            (
                ModeTask(2, Mode.RECON, (5, 0), 1, 0, 10),
                ModeTask(2, Mode.ATTACK, (8, 8), 0, 1, 2),
                ModeTask(2, Mode.BDA, (0, 0), 0, 2, 20),
            ),
            (
                ModeTask(3, Mode.RECON, (5, 5), 1, 2, 7),
                ModeTask(3, Mode.ATTACK, (0, 5), 0, 0, 7),
                ModeTask(3, Mode.BDA, (5, 0), 0, 0, 10),
            ),
        ),
        beta=0.1,
        distance_cost=0,
        ammo_cost=0.2,
        instance_id="lower-score-fallback",
    )


def test_lower_score_later_iteration_cannot_replace_base():
    result = run_ranked_fallback(lower_score_fallback_instance())
    assert result.iterations[1].result.true_score < result.iterations[0].result.true_score
    assert result.selected_iteration == 0


def synthetic_iteration(index, score, assigned_modes):
    cbba_result = ModeCBBAResult(
        ModeMethod.JOHNSON_WARPED,
        "converged",
        1,
        (None,) * len(assigned_modes),
        (),
        score,
        (),
    )
    return FallbackIteration(
        index,
        (0,) * len(assigned_modes),
        (None,) * len(assigned_modes),
        cbba_result,
        (),
        True,
        (),
        assigned_modes,
    )


def select_synthetic_iterations(scores):
    iterations = tuple(
        synthetic_iteration(i, score, (None,) if i else (Mode.RECON,))
        for i, score in enumerate(scores)
    )
    return select_fallback_iteration(iterations, (0,))


def test_near_max_tie_prefers_fewer_unresolved_without_tolerance_drift():
    selected = select_synthetic_iterations((10.0, 10.0 - 0.9e-12, 10.0 - 1.8e-12))
    assert selected.index == 0


def test_fallback_replay_is_fieldwise_identical():
    instance = second_mode_resolution_instance()
    first = run_ranked_fallback(instance)
    assert first == run_ranked_fallback(instance)
    assert all(item.result.method is ModeMethod.JOHNSON_WARPED for item in first.iterations)


def test_base_gate_failure_returns_invalid_result_without_selection_or_rates():
    instance = no_orphan_instance()
    legal = run_ranked_fallback(instance).iterations[0]
    failed = replace(
        legal,
        legal=False,
        gate_report=(("cycle_or_timeout", 1),),
    )

    result = finalize_fallback_attempts(
        (failed,), rank_mode_candidates(instance)
    )

    assert result.valid is False
    assert result.iterations == ()
    assert result.selected_iteration is None
    assert result.base_orphan_rate is None
    assert result.fallback_unresolved_rate is None
    assert result.base_gate_failures == (failed,)
    assert result.late_gate_failures == ()


def test_late_gate_failure_stops_and_preserves_earlier_legal_candidate():
    instance = second_mode_resolution_instance()
    completed = run_ranked_fallback(instance)
    base, resolved = completed.iterations
    failed = replace(
        resolved,
        legal=False,
        gate_report=(("cycle_or_timeout", 1),),
    )

    result = finalize_fallback_attempts(
        (base, failed, resolved), rank_mode_candidates(instance)
    )

    assert result.valid is True
    assert result.iterations == (base,)
    assert result.selected_iteration == 0
    assert result.selected_assigned_modes == base.assigned_modes
    assert result.base_gate_failures == ()
    assert result.late_gate_failures == (failed,)
    assert result.total_johnson_calls == 2
