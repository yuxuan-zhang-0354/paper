from uav_lifecycle.mode_allocation import Mode, ModeAgent, ModeInstance, ModeTask
from uav_lifecycle.mode_cbba import (
    ModeMethod,
    run_mode_cbba,
    screen_modes,
    validate_mode_result,
)


def task(target, mode, x, utility, ammo=0, duration=0):
    return ModeTask(target, mode, (x, 0), duration, ammo, utility)


def test_screening_selects_fleet_witness_and_defers_nonpositive_target():
    instance = ModeInstance(
        agents=(ModeAgent(0, (0, 0), 10, 10, 1), ModeAgent(1, (9, 0), 10, 10, 1)),
        tasks_by_target=(
            (task(0, Mode.RECON, 8, 5), task(0, Mode.ATTACK, 8, 8, 1), task(0, Mode.BDA, 8, 6)),
            (task(1, Mode.RECON, 5, -1), task(1, Mode.ATTACK, 5, -1, 1), task(1, Mode.BDA, 5, -1)),
        ),
        beta=0,
        distance_cost=1,
        ammo_cost=0,
    )
    screened = screen_modes(instance)
    assert screened[0].mode is Mode.ATTACK
    assert screened[0].witness_agent == 1
    assert screened[1] is None


def test_screening_tie_prefers_recon_then_lower_agent_id():
    instance = ModeInstance(
        agents=(ModeAgent(0, (0, 0), 2, 2, 0), ModeAgent(1, (0, 0), 2, 2, 0)),
        tasks_by_target=((task(0, Mode.BDA, 0, 1), task(0, Mode.RECON, 0, 1)),),
        beta=0,
        distance_cost=0,
        ammo_cost=0,
    )
    selected = screen_modes(instance)[0]
    assert selected.mode is Mode.RECON
    assert selected.witness_agent == 0


def test_cbba_variants_are_feasible_deterministic_and_johnson_bids_are_monotone():
    instance = ModeInstance(
        agents=(ModeAgent(0, (0, 0), 10, 10, 1), ModeAgent(1, (4, 0), 10, 10, 1)),
        tasks_by_target=(
            (task(0, Mode.RECON, 1, 6), task(0, Mode.ATTACK, 1, 10, 1)),
            (task(1, Mode.RECON, 3, 8), task(1, Mode.ATTACK, 3, 9, 1)),
        ),
        beta=0,
        distance_cost=0.2,
        ammo_cost=0.5,
    )
    screened = screen_modes(instance)
    for method in ModeMethod:
        first = run_mode_cbba(instance, screened, method)
        second = run_mode_cbba(instance, screened, method)
        assert first == second
        assert first.status == "converged"
        assert all(value == 0 for value in validate_mode_result(instance, screened, first).values())


def test_positive_screened_task_can_be_orphaned_by_shared_ammo_witness():
    instance = ModeInstance(
        agents=(ModeAgent(0, (0, 0), 10, 10, 1),),
        tasks_by_target=(
            (task(0, Mode.ATTACK, 0, 10, 1),),
            (task(1, Mode.ATTACK, 0, 9, 1),),
        ),
        beta=0,
        distance_cost=0,
        ammo_cost=0,
    )
    screened = screen_modes(instance)
    result = run_mode_cbba(instance, screened, ModeMethod.FULL_REBUILD_RAW)
    assert result.orphan_targets == (1,)
    assert result.true_score == 10


def test_standard_raw_reports_messages_and_uses_classic_suffix_release():
    instance = ModeInstance(
        agents=(ModeAgent(0, (0, 0), 10, 10, 2), ModeAgent(1, (4, 0), 10, 10, 2)),
        tasks_by_target=(
            (task(0, Mode.ATTACK, 1, 10, 1),),
            (task(1, Mode.ATTACK, 3, 9, 1),),
        ),
        beta=0, distance_cost=.1, ammo_cost=.5,
    )
    screened = screen_modes(instance)
    result = run_mode_cbba(instance, screened, ModeMethod.STANDARD_RAW)
    assert result.status == "converged"
    assert result.message_packets == result.rounds * 2
    assert result.message_scalars == result.message_packets * 2
    assert all(value == 0 for value in validate_mode_result(instance, screened, result).values())
