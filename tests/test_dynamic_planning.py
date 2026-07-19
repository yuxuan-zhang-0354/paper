from dataclasses import replace

import pytest

from uav_lifecycle.dynamic_planning import (
    PlannedPath,
    build_planning_problem,
    commit_ticks,
    exact_audit_gates,
    nearest_positive_order_key,
    plan_all_mode_exact,
    plan_johnson,
    plan_nearest_positive,
    run_exact_search,
    target_continuation_gates,
    validate_johnson_audit,
    validate_commit_candidates,
)
from uav_lifecycle.dynamic_types import (
    DynamicConfig,
    PublicAgent,
    PublicBusyAction,
    PublicSnapshot,
    PublicTarget,
    quantize_tick,
)
from uav_lifecycle.mode_allocation import Mode
from uav_lifecycle.mode_cbba import ModeAgentBundle, ModeCBBAResult, ModeMethod
from uav_lifecycle.mode_exact import solve_all_mode_exact


def snapshot(*, tick=0, targets=None, agents=None, locks=()):
    targets = targets or (
        PublicTarget(0, (1.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        PublicTarget(1, (4.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
    )
    agents = agents or (
        PublicAgent(0, (0.0, 0.0), 2.0, 20.0, None),
        PublicAgent(1, (5.0, 0.0), 2.0, 20.0, None),
    )
    return PublicSnapshot(tick, targets, agents, locks, (), ())


def test_continuation_gate_is_target_level_winner_independent_and_busy_excluded():
    config = DynamicConfig()
    finish = quantize_tick(5.0, config.tick_size)
    snap = snapshot(
        targets=(
            PublicTarget(0, (1.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            PublicTarget(1, (4.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
            PublicTarget(2, (3.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
        ),
        agents=(
            PublicAgent(0, (1.0, 0.0), 4.0, 100.0, PublicBusyAction(1, (4.0, 0.0), finish)),
            PublicAgent(1, (0.0, 0.0), 1.0, 3.0, None),
        ),
        locks=((1, 0),),
    )
    gates = target_continuation_gates(snap, config, quantize_tick(8.0, config.tick_size))
    assert gates[0] == {Mode.RECON: True, Mode.ATTACK: False, Mode.BDA: True}
    assert gates[2] == {Mode.RECON: False, Mode.ATTACK: False, Mode.BDA: True}
    assert 1 not in gates
    problem = build_planning_problem(snap, config, quantize_tick(8.0, config.tick_size))
    assert problem.global_agent_ids == (1,)
    assert problem.global_target_ids == (0, 2)
    assert problem.continuation_gates[0] == gates[0]


def test_compact_ids_are_stable_and_results_restore_global_ids():
    config = DynamicConfig()
    finish = quantize_tick(6.0, config.tick_size)
    snap = snapshot(
        targets=(
            PublicTarget(0, (9.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            PublicTarget(1, (1.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            PublicTarget(2, (3.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ),
        agents=(
            PublicAgent(0, (9.0, 0.0), 2.0, 20.0, PublicBusyAction(0, (9.0, 0.0), finish)),
            PublicAgent(1, (0.0, 0.0), 2.0, 20.0, None),
            PublicAgent(2, (4.0, 0.0), 2.0, 20.0, None),
        ),
        locks=((0, 0),),
    )
    result = plan_johnson(snap, config, quantize_tick(20.0, config.tick_size))
    assert {path.agent_id for path in result.paths} <= {1, 2}
    assert {target for path in result.paths for target, _ in path.tasks} <= {1, 2}
    assert all(target != 0 for path in result.paths for target, _ in path.tasks)


def test_nearest_positive_uses_frozen_pair_order_and_one_task_per_agent():
    config = DynamicConfig()
    result = plan_nearest_positive(snapshot(), config, quantize_tick(20.0, config.tick_size))
    assert all(len(path.tasks) == 1 for path in result.paths)
    assert len({path.agent_id for path in result.paths}) == len(result.paths)
    assert len({path.tasks[0][0] for path in result.paths}) == len(result.paths)
    assert result.positive_pair_count >= len(result.paths)


def test_nearest_positive_real_symmetric_tie_prefers_target_then_agent():
    config = DynamicConfig()
    symmetric = snapshot(
        targets=(
            PublicTarget(0, (1.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            PublicTarget(1, (-1.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ),
        agents=(
            PublicAgent(0, (0.0, 0.0), 2.0, 20.0, None),
            PublicAgent(1, (0.0, 0.0), 2.0, 20.0, None),
        ),
    )
    result = plan_nearest_positive(symmetric, config, quantize_tick(20.0, config.tick_size))
    assert tuple((path.agent_id, path.tasks[0][0]) for path in result.paths) == ((0, 0), (1, 1))


def test_exact_enforces_small_target_bound_and_deterministic_paths():
    config = DynamicConfig()
    snap = snapshot()
    first = plan_all_mode_exact(snap, config, quantize_tick(20.0, config.tick_size))
    assert first == plan_all_mode_exact(snap, config, quantize_tick(20.0, config.tick_size))
    assert first.search_count is not None and first.search_bound is not None
    assert first.search_count <= first.search_bound
    assigned = [target for path in first.paths for target, _ in path.tasks]
    assert len(assigned) == len(set(assigned))
    six = tuple(PublicTarget(i, (float(i), 0.0), (1.0, 0.0, 0.0, 0.0)) for i in range(6))
    with pytest.raises(ValueError, match="at most 5"):
        plan_all_mode_exact(snapshot(targets=six), config, quantize_tick(20.0, config.tick_size))


def test_commit_validation_is_whole_batch_and_pure():
    config = DynamicConfig()
    snap = snapshot()
    max_tick = quantize_tick(20.0, config.tick_size)
    before = snap
    duplicate = (
        PlannedPath(0, ((0, Mode.ATTACK),), 1.0, (quantize_tick(1.0, config.tick_size),), (quantize_tick(3.0, config.tick_size),)),
        PlannedPath(1, ((0, Mode.RECON),), 1.0, (quantize_tick(4.0, config.tick_size),), (quantize_tick(8.0, config.tick_size),)),
    )
    failures = validate_commit_candidates(snap, config, max_tick, duplicate)
    assert failures and failures[0].gate == "commit_batch"
    assert snap == before

    unavailable = (replace(duplicate[0], tasks=((0, Mode.ATTACK),), completion_ticks=(max_tick + 1,)),)
    assert validate_commit_candidates(snap, config, max_tick, unavailable)


def test_b1m_multileg_ticks_quantize_each_leg_and_land_on_horizon_boundary():
    config = DynamicConfig()
    snap = snapshot(
        targets=(
            PublicTarget(0, (1.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            PublicTarget(1, (2.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ),
        agents=(PublicAgent(0, (0.0, 0.0), 2.0, 2.0, None),),
    )
    max_tick = quantize_tick(5.0, config.tick_size)
    problem = build_planning_problem(snap, config, max_tick)
    recon_path = ((0, Mode.BDA), (1, Mode.BDA))
    local = problem.instance
    from uav_lifecycle.mode_allocation import evaluate_mode_path

    evaluation = evaluate_mode_path(local, 0, recon_path)
    assert evaluation.start_ticks == (
        quantize_tick(1.0, config.tick_size),
        quantize_tick(3.5, config.tick_size),
    )
    assert evaluation.completion_tick == max_tick
    assert evaluation.feasible


def test_predicted_first_leg_ticks_equal_shared_commit_tick_contract():
    config = DynamicConfig()
    snap = snapshot(tick=quantize_tick(0.5, config.tick_size))
    result = plan_nearest_positive(snap, config, quantize_tick(20.0, config.tick_size))
    first = result.paths[0]
    target_id, mode = first.tasks[0]
    expected = commit_ticks(
        snap.tick,
        snap.agents[first.agent_id].position,
        snap.targets[target_id].position,
        mode,
        config,
    )
    assert (first.start_ticks[0], first.completion_ticks[0]) == expected


def test_screening_is_frozen_once_per_johnson_call(monkeypatch):
    import uav_lifecycle.dynamic_planning as planning

    calls = 0
    real = planning.screen_modes

    def counted(instance):
        nonlocal calls
        calls += 1
        return real(instance)

    monkeypatch.setattr(planning, "screen_modes", counted)
    plan_johnson(snapshot(), DynamicConfig(), quantize_tick(20.0, DynamicConfig().tick_size))
    assert calls == 1


@pytest.mark.parametrize(
    ("status", "report", "orphans", "positive", "reason"),
    [
        ("cycle", {}, (), 0, "cycle"),
        ("timeout", {}, (), 0, "round_cap"),
        ("converged", {"winner_conflicts": 1}, (), 0, "winner_conflicts"),
        ("converged", {"infeasible_paths": 1}, (), 0, "infeasible_paths"),
        ("converged", {"bundle_path_mismatches": 1}, (), 0, "bundle_path_mismatches"),
        ("converged", {"warped_monotonicity_violations": 1}, (), 0, "warped_monotonicity_violations"),
        ("converged", {"replay_mismatches": 1}, (), 0, "replay_mismatches"),
        ("converged", {}, (), 1, "allocation_stall"),
    ],
)
def test_each_johnson_structural_failure_has_an_independent_gate(
    status, report, orphans, positive, reason
):
    result = ModeCBBAResult(ModeMethod.JOHNSON_WARPED, status, 3, (), (), 0.0, orphans, 2 if status == "cycle" else None)
    failures = validate_johnson_audit(7, result, report, positive)
    assert tuple(failure.reason for failure in failures) == (reason,)


def test_unassigned_positive_target_defers_without_structural_gate():
    result = ModeCBBAResult(
        ModeMethod.JOHNSON_WARPED,
        "converged",
        3,
        (),
        (ModeAgentBundle(0, (0,), ((0, Mode.ATTACK),), ()),),
        1.0,
        (1,),
    )
    assert validate_johnson_audit(7, result, {}, positive_pair_count=2) == ()


def test_raw_allocator_cycle_is_exposed_as_an_allocation_failure(monkeypatch):
    import uav_lifecycle.dynamic_planning as planning

    cycle = ModeCBBAResult(
        ModeMethod.STANDARD_RAW,
        "cycle",
        3,
        (),
        (),
        0.0,
        (),
        2,
    )
    monkeypatch.setattr(planning, "run_mode_cbba", lambda *args: cycle)
    monkeypatch.setattr(planning, "validate_mode_result", lambda *args: {})
    result = planning.plan_vanilla(
        snapshot(), DynamicConfig(), quantize_tick(20.0, DynamicConfig().tick_size)
    )
    assert result.status == "allocator_nonconvergence"
    assert result.positive_pair_count > 0
    assert tuple(gate.reason for gate in result.gates) == ("cycle",)


def test_b6_order_key_exercises_each_registered_tie_level():
    candidates = [
        (1.0, 4.0, 2, 0),
        (1.0, 5.0, 2, 1),
        (1.0, 5.0, 1, 1),
        (1.0, 5.0, 1, 0),
        (0.5, 1.0, 9, 9),
    ]
    assert sorted(candidates, key=lambda item: nearest_positive_order_key(*item)) == [
        (0.5, 1.0, 9, 9),
        (1.0, 5.0, 1, 0),
        (1.0, 5.0, 1, 1),
        (1.0, 5.0, 2, 1),
        (1.0, 4.0, 2, 0),
    ]


def test_cex_audit_enforces_bound_and_tie_key():
    problem = build_planning_problem(snapshot(), DynamicConfig(), quantize_tick(20.0, DynamicConfig().tick_size))
    result = solve_all_mode_exact(problem.instance)
    assert exact_audit_gates(0, result, 98) == ()
    assert exact_audit_gates(0, result, result.audit.profile_equivalent_count - 1)[0].reason == "search_bound_exceeded"
    from dataclasses import replace as dc_replace

    wrong = dc_replace(result, audit=dc_replace(result.audit, solution_key=()))
    assert exact_audit_gates(0, wrong, 98)[0].reason == "tie_key_mismatch"


def test_cex_injected_low_cap_returns_gate_and_no_incumbent():
    problem = build_planning_problem(snapshot(), DynamicConfig(), quantize_tick(20.0, DynamicConfig().tick_size))
    result, gates, actual = run_exact_search(problem.instance, tick=0, search_bound=66)
    assert result is None
    assert tuple(gate.reason for gate in gates) == ("search_bound_exceeded",)
    assert actual == 67


def test_commit_validation_isolates_each_rejection_reason_without_mutation():
    config = DynamicConfig()
    max_tick = quantize_tick(20.0, config.tick_size)
    base = snapshot()
    valid = plan_nearest_positive(base, config, max_tick).paths
    assert valid
    candidate = valid[0]

    cases = []
    duplicate = valid[:2] if len(valid) >= 2 else (candidate, replace(candidate, agent_id=1))
    duplicate = (duplicate[0], replace(duplicate[1], tasks=((duplicate[0].tasks[0][0], duplicate[1].tasks[0][1]),)))
    cases.append((base, duplicate, "duplicate_target"))

    attack = next(path for path in plan_all_mode_exact(base, config, max_tick).paths if any(mode is Mode.ATTACK for _, mode in path.tasks))
    no_ammo_agents = tuple(replace(agent, available_ammo=0.0) if agent.agent_id == attack.agent_id else agent for agent in base.agents)
    cases.append((replace(base, agents=no_ammo_agents), (attack,), "ammo_unavailable"))

    no_range_agents = tuple(replace(agent, available_distance=0.0) if agent.agent_id == candidate.agent_id else agent for agent in base.agents)
    cases.append((replace(base, agents=no_range_agents), (candidate,), "distance_unavailable"))
    cases.append((base, (replace(candidate, start_ticks=(candidate.start_ticks[0] + 1,)),), "tick_path_mismatch"))

    finish = quantize_tick(4.0, config.tick_size)
    busy = replace(base.agents[0], busy_action=PublicBusyAction(0, base.targets[0].position, finish))
    busy_snapshot = replace(base, agents=(busy, base.agents[1]), target_locks=((0, 0),))
    non_idle = replace(candidate, agent_id=0)
    cases.append((busy_snapshot, (non_idle,), "non_idle_winner"))

    for snap, batch, expected_reason in cases:
        before = snap
        failures = validate_commit_candidates(snap, config, max_tick, batch)
        assert tuple(failure.reason for failure in failures) == (expected_reason,)
        assert snap == before
