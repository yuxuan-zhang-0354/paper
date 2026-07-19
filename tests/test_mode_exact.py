import itertools
from dataclasses import replace

import pytest

import numpy as np

from uav_lifecycle.mode_allocation import Mode, ModeAgent, ModeInstance, ModeTask, evaluate_mode_path
from uav_lifecycle.mode_exact import (
    ExactSearchBoundExceeded,
    profile_equivalent_count,
    solve_all_mode_exact,
    solve_fixed_mode_exact,
    validate_exact_audit,
)


def tiny_instance():
    return ModeInstance(
        agents=(
            ModeAgent(0, (0.0, 0.0), 10, 10, 1),
            ModeAgent(1, (4.0, 0.0), 10, 10, 0),
        ),
        tasks_by_target=(
            (ModeTask(0, Mode.RECON, (1.0, 0.0), 1, 0, 7), ModeTask(0, Mode.ATTACK, (1.0, 0.0), 1, 1, 12), ModeTask(0, Mode.BDA, (1.0, 0.0), 1, 0, 4)),
            (ModeTask(1, Mode.RECON, (3.0, 0.0), 1, 0, 9), ModeTask(1, Mode.ATTACK, (3.0, 0.0), 1, 1, 11), ModeTask(1, Mode.BDA, (3.0, 0.0), 1, 0, 5)),
        ), beta=0, distance_cost=0, ammo_cost=0,
    )


def test_all_mode_exact_matches_direct_two_agent_enumeration():
    instance = tiny_instance()
    options = [(), ((0, Mode.RECON),), ((0, Mode.ATTACK),), ((1, Mode.RECON),), ((1, Mode.ATTACK),)]
    direct = 0.0
    for p0, p1 in itertools.product(options, repeat=2):
        if {k[0] for k in p0} & {k[0] for k in p1}:
            continue
        e0, e1 = evaluate_mode_path(instance, 0, p0), evaluate_mode_path(instance, 1, p1)
        if e0.feasible and e1.feasible:
            direct = max(direct, e0.score + e1.score)
    np.testing.assert_allclose(solve_all_mode_exact(instance).score, direct)


def test_all_mode_is_never_below_fixed_mode():
    instance = tiny_instance()
    fixed = {0: Mode.RECON, 1: Mode.BDA}
    all_result = solve_all_mode_exact(instance)
    fixed_result = solve_fixed_mode_exact(instance, fixed)
    assert all_result.score >= fixed_result.score
    assert fixed_result.target_modes == (Mode.RECON, Mode.BDA)


def test_exact_tie_is_deterministic():
    instance = ModeInstance(
        agents=(ModeAgent(0, (0, 0), 10, 10, 0),),
        tasks_by_target=((ModeTask(0, Mode.RECON, (0, 0), 0, 0, 1), ModeTask(0, Mode.BDA, (0, 0), 0, 0, 1)),),
        beta=0, distance_cost=0, ammo_cost=0,
    )
    assert solve_all_mode_exact(instance).paths == (((0, Mode.RECON),),)


def test_exact_search_audit_counts_candidate_evaluations_and_dp_transitions():
    result = solve_all_mode_exact(tiny_instance())
    assert result.audit.candidate_path_evaluations == (24, 24)
    assert result.audit.dp_transitions == 13
    assert result.audit.total_count == 61
    assert result.audit.profile_equivalent_count == 67
    assert validate_exact_audit(result, operation_cap=98) == ()


def test_exact_search_bound_aborts_without_returning_an_incumbent():
    with pytest.raises(ExactSearchBoundExceeded) as error:
        solve_all_mode_exact(tiny_instance(), operation_cap=66)
    assert error.value.actual_count == 67
    assert error.value.operation_cap == 66


def test_exact_audit_detects_injected_over_cap_and_tie_key_mismatch():
    result = solve_all_mode_exact(tiny_instance())
    over_cap = replace(result.audit, profile_equivalent_count=99)
    assert validate_exact_audit(replace(result, audit=over_cap), 98) == ("search_bound_exceeded",)
    internal_only = replace(result.audit, dp_transitions=10_000)
    assert validate_exact_audit(replace(result, audit=internal_only), 98) == ()
    wrong_key = replace(result.audit, solution_key=())
    assert validate_exact_audit(replace(result, audit=wrong_key), 98) == ("tie_key_mismatch",)


def test_profile_equivalent_count_has_registered_base_cases():
    assert profile_equivalent_count(agent_count=1, target_count=1) == 4
    assert profile_equivalent_count(agent_count=3, target_count=0) == 1


def test_single_target_single_agent_solver_observes_four_profiles_under_cap_four():
    one = ModeInstance(
        agents=(ModeAgent(0, (0, 0), 10, 10, 1),),
        tasks_by_target=((
            ModeTask(0, Mode.RECON, (0, 0), 0, 0, 1),
            ModeTask(0, Mode.ATTACK, (0, 0), 0, 1, 2),
            ModeTask(0, Mode.BDA, (0, 0), 0, 0, 1),
        ),),
        beta=0,
        distance_cost=0,
        ammo_cost=0,
    )
    result = solve_all_mode_exact(one, operation_cap=4)
    assert result.audit.profile_equivalent_count == 4
    assert result.paths == (((0, Mode.ATTACK),),)


def test_empty_target_solver_counts_one_profile_even_with_internal_dp_transitions():
    empty = ModeInstance(
        agents=(ModeAgent(0, (0, 0), 1, 1, 0), ModeAgent(1, (0, 0), 1, 1, 0)),
        tasks_by_target=(),
        beta=0,
        distance_cost=0,
        ammo_cost=0,
    )
    result = solve_all_mode_exact(empty, operation_cap=1)
    assert result.audit.profile_equivalent_count == 1
    assert result.audit.total_count == 2
    assert result.paths == ((), ())


@pytest.mark.parametrize("agent_count", (1, 2, 3))
@pytest.mark.parametrize("target_count", range(6))
def test_profile_equivalent_count_respects_registered_cap(agent_count, target_count):
    from math import factorial

    count = profile_equivalent_count(agent_count, target_count)
    cap = factorial(target_count) * (1 + 3 * agent_count) ** target_count
    assert 1 <= count <= cap


def test_near_tie_uses_lexicographic_solution_key_exhaustively():
    near_tie = ModeInstance(
        agents=(ModeAgent(0, (0, 0), 10, 10, 0),),
        tasks_by_target=((
            ModeTask(0, Mode.RECON, (0, 0), 0, 0, 1.0),
            ModeTask(0, Mode.BDA, (0, 0), 0, 0, 1.0 + 0.5e-12),
        ),),
        beta=0,
        distance_cost=0,
        ammo_cost=0,
    )
    result = solve_all_mode_exact(near_tie)
    exhaustive = [
        (evaluate_mode_path(near_tie, 0, path).score, path)
        for path in ((), ((0, Mode.RECON),), ((0, Mode.BDA),))
    ]
    best_score = max(score for score, _ in exhaustive)
    tied = [path for score, path in exhaustive if best_score - score <= 1e-12]
    assert result.paths == (min(tied, key=lambda path: tuple((target, 0 if mode is Mode.RECON else 2) for target, mode in path)),)
