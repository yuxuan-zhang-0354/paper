from dataclasses import asdict, fields, is_dataclass
from fractions import Fraction
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path

import pytest

from uav_lifecycle.dynamic_rng import DrawKey, canonical_key_bytes, categorical, uniform01
from uav_lifecycle.dynamic_scenarios import (
    AssertionId,
    D0Action,
    D1_CELLS,
    D0Fault,
    D0GateCode,
    D0Fixture,
    D0Run,
    D0Terminal,
    ExpectedPrivateRecord,
    ExpectedAbsence,
    ExpectedPublicRecord,
    FixtureAssertion,
    PublicPrecondition,
    PublicRecordKind,
    PrivateRecordKind,
    RejectionReason,
    ScriptedPolicyStep,
    d0_contract_digest,
    d1_draw_digest,
    dynamic_config_registry,
    dynamic_registry_digest,
    d0_scenarios,
    generate_d1_scenario,
    generate_d2_scenario,
)
from uav_lifecycle.dynamic_types import quantize_tick
from uav_lifecycle.attack import predict_attack
from uav_lifecycle.belief import bayes_update, bda_kernel, recon_kernel


PASSPORT_DIGEST = "314f923560a280221149613fffd7f51eb358150658e11bebf89640d9311cb57e"
APPROVED_CONFIG_ID = "recon_damage_plus_010_r2_a6_b3"
TRUTH_ORDER = (("H", "A"), ("H", "D"), ("L", "A"), ("L", "D"))
ARCHETYPES = (
    (0.00, 0.42, 0.24, 0.34),
    (1.00, 0.00, 0.00, 0.00),
    (0.26, 0.66, 0.00, 0.08),
    (0.00, 0.08, 0.06, 0.86),
)


def _key(cell_id: str, seed: int, namespace: str, entity_id: int, event_type: str, subdraw: int) -> DrawKey:
    return DrawKey(
        rng_version="sha256-u64-v1",
        experiment_id="dynamic-lifecycle-mainline-v2",
        generator_version="d1-generator-v1",
        cell_id=cell_id,
        within_cell_seed=seed,
        entity_namespace=namespace,
        entity_id=entity_id,
        event_type=event_type,
        occurrence_index=0,
        subdraw_index=subdraw,
    )


def _expected_coordinate(key: DrawKey) -> float:
    return float(Fraction(-6) + 12 * uniform01(key))


def _draw_digest(scenario: object) -> str:
    payload = {
        "scenario_id": scenario.scenario_id,
        "targets": [asdict(item) for item in scenario.targets],
        "private_targets": [asdict(item) for item in scenario.private_targets],
        "agents": [asdict(item) for item in scenario.agents],
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_registry_is_digest_pinned_to_material_passport_and_only_approved_config() -> None:
    source = Path("preregistration/first_batch.json").read_bytes()
    assert sha256(source).hexdigest() == PASSPORT_DIGEST
    assert dynamic_registry_digest() == PASSPORT_DIGEST
    registry = dynamic_config_registry()
    assert tuple(registry) == (APPROVED_CONFIG_ID,)
    assert registry[APPROVED_CONFIG_ID].config_id == APPROVED_CONFIG_ID
    with pytest.raises(TypeError):
        registry["new"] = registry[APPROVED_CONFIG_ID]  # type: ignore[index]


def test_d1_cells_are_exact_bound_resource_cartesian_product() -> None:
    actual = {(cell.agent_count, cell.target_count, cell.resource_tier) for cell in D1_CELLS}
    assert actual == {
        (n, m, tier)
        for n in (2, 3)
        for m in (3, 5)
        for tier in ("tight", "loose")
    }
    for cell in D1_CELLS:
        expected = (1.0, 16.0, 18.0) if cell.resource_tier == "tight" else (3.0, 32.0, 40.0)
        assert (cell.ammo_per_agent, cell.t_max, cell.range_per_agent) == expected
        assert cell.cell_id == f"N{cell.agent_count}-M{cell.target_count}-R{cell.resource_tier}"


def test_d1_has_exactly_160_stable_ids_with_reserved_namespace_separation() -> None:
    scenarios = [generate_d1_scenario(cell, index, APPROVED_CONFIG_ID) for cell in D1_CELLS for index in range(20)]
    ids = [scenario.scenario_id for scenario in scenarios]
    assert len(ids) == len(set(ids)) == 160
    assert all(value.startswith("D1-") for value in ids)
    assert all(not value.startswith(("D0-", "D2-")) for value in ids)
    with pytest.raises(ValueError, match="scenario_index"):
        generate_d1_scenario(D1_CELLS[0], 20, APPROVED_CONFIG_ID)


def test_d2_uses_frozen_new_ids_indices_and_rng_namespace() -> None:
    first = generate_d2_scenario(D1_CELLS[0], 1000, APPROVED_CONFIG_ID)
    last = generate_d2_scenario(D1_CELLS[-1], 1511, APPROVED_CONFIG_ID)
    assert first.scenario_id == f"D2-{D1_CELLS[0].cell_id}-S1000"
    assert last.scenario_id == f"D2-{D1_CELLS[-1].cell_id}-S1511"
    assert first.crn_namespace == "dynamic-lifecycle-mainline-v2/d2-generator-v1"
    assert first.crn_namespace != generate_d1_scenario(D1_CELLS[0], 0, APPROVED_CONFIG_ID).crn_namespace
    for index in (999, 1512):
        with pytest.raises(ValueError, match="scenario_index"):
            generate_d2_scenario(D1_CELLS[0], index, APPROVED_CONFIG_ID)


def test_generator_is_deterministic_and_has_no_calibration_run_input_or_state() -> None:
    signature = inspect.signature(generate_d1_scenario)
    assert tuple(signature.parameters) == ("cell", "scenario_index", "registered_config_id")
    assert all("run" not in field.name.lower() for field in fields(type(D1_CELLS[0])))
    forward = {
        (cell.cell_id, seed): _draw_digest(generate_d1_scenario(cell, seed, APPROVED_CONFIG_ID))
        for cell in D1_CELLS
        for seed in range(20)
    }
    reverse = {
        (cell.cell_id, seed): _draw_digest(generate_d1_scenario(cell, seed, APPROVED_CONFIG_ID))
        for cell in reversed(D1_CELLS)
        for seed in reversed(range(20))
    }
    assert forward == reverse
    serialized = json.dumps(sorted((cell_id, seed, digest) for (cell_id, seed), digest in forward.items()))
    assert not any(f"run_{run}" in serialized for run in range(3))


def test_unregistered_config_is_rejected_without_changing_generator_namespace() -> None:
    with pytest.raises(KeyError, match="unregistered"):
        generate_d1_scenario(D1_CELLS[0], 0, "later_unapproved_config")


def test_positions_use_exact_counter_uniform_mapping_and_strict_namespaces() -> None:
    cell = next(item for item in D1_CELLS if item.cell_id == "N2-M3-Rtight")
    seed = 7
    scenario = generate_d1_scenario(cell, seed, APPROVED_CONFIG_ID)
    keys: list[DrawKey] = []
    for target in scenario.targets:
        expected = tuple(
            _expected_coordinate(_key(cell.cell_id, seed, "target", target.target_id, "initial_position", axis))
            for axis in (0, 1)
        )
        keys.extend(_key(cell.cell_id, seed, "target", target.target_id, "initial_position", axis) for axis in (0, 1))
        assert target.position == expected
    for agent in scenario.agents:
        expected = tuple(
            _expected_coordinate(_key(cell.cell_id, seed, "agent", agent.agent_id, "initial_position", axis))
            for axis in (0, 1)
        )
        keys.extend(_key(cell.cell_id, seed, "agent", agent.agent_id, "initial_position", axis) for axis in (0, 1))
        assert agent.position == expected
    assert len({canonical_key_bytes(key) for key in keys}) == 2 * (cell.agent_count + cell.target_count)
    assert scenario.targets[0].position != scenario.agents[0].position


def test_beliefs_cycle_by_target_plus_seed_and_truth_is_each_beliefs_inverse_cdf() -> None:
    for cell in D1_CELLS:
        for seed in range(20):
            scenario = generate_d1_scenario(cell, seed, APPROVED_CONFIG_ID)
            for target, private in zip(scenario.targets, scenario.private_targets, strict=True):
                expected_belief = ARCHETYPES[(target.target_id + seed) % 4]
                assert target.belief == expected_belief
                assert abs(sum(target.belief) - 1.0) <= 1e-12
                truth_key = _key(cell.cell_id, seed, "target", target.target_id, "initial_truth", 0)
                assert (private.true_category, private.true_damage) == TRUTH_ORDER[categorical(truth_key, target.belief)]
                assert private.first_destroyed_paid is False


def test_resource_tier_is_linked_and_cell_membership_is_exact() -> None:
    for cell in D1_CELLS:
        scenario = generate_d1_scenario(cell, 0, APPROVED_CONFIG_ID)
        assert scenario.cell_id == cell.cell_id
        assert scenario.seed == 0
        assert scenario.t_max_tick == round(cell.t_max / dynamic_config_registry()[APPROVED_CONFIG_ID].tick_size)
        assert len(scenario.agents) == cell.agent_count
        assert len(scenario.targets) == cell.target_count
        for agent in scenario.agents:
            assert agent.ammo.available == cell.ammo_per_agent
            assert agent.ammo.total == cell.ammo_per_agent
            assert agent.distance.available == cell.range_per_agent
            assert agent.distance.total == cell.range_per_agent


EXPECTED_D0_NAMES = (
    "initial_wreck_zero_reward",
    "first_new_destroyed_paid_once",
    "continuous_attack_after_failure",
    "attack_ack_hides_outcome",
    "recon_joint_bayes",
    "bda_before_first_attack",
    "target_lock_prevents_overlap",
    "busy_uav_excluded_from_bid_and_gate",
    "simultaneous_completion_batch",
    "commit_next_releases_suffix",
    "cross_uav_sequential_handoff",
    "horizon_range_ammo_rejection",
    "completion_precedes_periodic_planning",
    "defer_reactivated_by_event",
    "no_positive_normal_termination",
    "counter_replay_shared_initial_truth",
    "busy_event_prevents_early_termination",
    "completion_at_horizon_settles_first",
    "positive_task_zero_commit_stall_gate",
    "public_private_event_nonleakage",
    "b1m_frozen_suffix_auto_next",
    "dynamic_absolute_discount_clock",
)


def test_d0_is_22_harness_consumable_structured_fixtures_not_executed_results() -> None:
    fixtures = d0_scenarios()
    assert tuple(fixture.name for fixture in fixtures) == EXPECTED_D0_NAMES
    assert len(fixtures) == len(set(fixture.name for fixture in fixtures)) == 22
    for fixture in fixtures:
        assert isinstance(fixture, D0Fixture)
        assert is_dataclass(fixture)
        assert fixture.scenario_id == f"D0-{fixture.name}"
        assert fixture.scenario.scenario_id == fixture.scenario_id
        assert fixture.scenario.t_max_tick > 0
        assert fixture.script
        assert fixture.expected_public_trace or fixture.runs[0].expected_absences
        assert fixture.expected_private_audit_trace or fixture.expected_gate is not None
        assert isinstance(fixture.expected_terminal, D0Terminal)
        assert fixture.focused_assertions
        assert all(isinstance(assertion, FixtureAssertion) for assertion in fixture.focused_assertions)
        assert all(isinstance(assertion.assertion_id, AssertionId) for assertion in fixture.focused_assertions)
        assert all(assertion.operands for assertion in fixture.focused_assertions)
        assert not hasattr(fixture, "actual_public_trace")
        assert not hasattr(fixture, "passed")
    assert len({fixture.focused_assertions for fixture in fixtures}) == 22


def test_d0_fixture_scenarios_scripts_keys_and_records_are_directly_validatable() -> None:
    fixtures = {fixture.name: fixture for fixture in d0_scenarios()}
    for fixture in fixtures.values():
        scenario = fixture.scenario
        assert all(isinstance(step, ScriptedPolicyStep) for step in fixture.script)
        assert all(isinstance(step.precondition, PublicPrecondition) for step in fixture.script)
        assert all(0 <= step.tick <= scenario.t_max_tick for step in fixture.script)
        assert all(step.agent_id is None or step.agent_id < len(scenario.agents) for step in fixture.script)
        assert all(step.target_id is None or step.target_id < len(scenario.targets) for step in fixture.script)
        assert all(step.action is not None or step.fault is not None for step in fixture.script)
        assert all(step.counter_key is None or isinstance(step.counter_key, DrawKey) for step in fixture.script)
        for step in fixture.script:
            if step.counter_key is not None:
                assert step.counter_key.cell_id == scenario.scenario_id
                assert step.counter_key.entity_namespace == "target"
                assert step.counter_key.entity_id == step.target_id
                assert step.action is not None and step.counter_key.event_type == step.action.value
                assert step.expected_uniform == uniform01(step.counter_key)
        assert all(isinstance(event, ExpectedPublicRecord) for event in fixture.expected_public_trace)
        assert all(isinstance(event, ExpectedPrivateRecord) for event in fixture.expected_private_audit_trace)
        assert all(event.tick <= scenario.t_max_tick for event in fixture.expected_public_trace)
        assert all(event.tick <= scenario.t_max_tick for event in fixture.expected_private_audit_trace)
        assert all(event.target_id < len(scenario.targets) for event in fixture.expected_public_trace)
        assert all(event.agent_id < len(scenario.agents) for event in fixture.expected_public_trace)
        assert all(event.target_id < len(scenario.targets) for event in fixture.expected_private_audit_trace)
        assert all(event.agent_id < len(scenario.agents) for event in fixture.expected_private_audit_trace)
        if fixture.expected_gate is not None:
            assert isinstance(fixture.expected_gate.gate, D0GateCode)

    ack_fixture = fixtures["attack_ack_hides_outcome"]
    ack = ack_fixture.expected_public_trace[0]
    forbidden = {"hit", "miss", "destroyed", "reward", "outcome", "physical_success", "draw", "truth"}
    assert forbidden.isdisjoint(field.name for field in fields(type(ack)))
    assert ack.observation is None
    assert fixtures["positive_task_zero_commit_stall_gate"].expected_gate.gate is D0GateCode.ALLOCATION_STALL  # type: ignore[union-attr]
    assert fixtures["no_positive_normal_termination"].expected_terminal is D0Terminal.NO_POSITIVE
    assert any(item.assertion_id is AssertionId.ABSOLUTE_DISCOUNT for item in fixtures["dynamic_absolute_discount_clock"].focused_assertions)


def test_public_canonical_digest_helpers_cover_all_fixtures_and_scenarios() -> None:
    fixtures = d0_scenarios()
    scenarios = tuple(
        generate_d1_scenario(cell, seed, APPROVED_CONFIG_ID)
        for cell in D1_CELLS
        for seed in range(20)
    )
    expected_d0 = "2c2cce4cbda0826083b6b4a04962e9c7f321d2be08b11f9ac8d34c51999f1fa0"
    expected_d1 = "1e63ee7c4aad7c796fd5d6380c4a3b1ff4bffb5e12b65b0bb0a3f46915a8737c"
    assert d0_contract_digest(fixtures) == d0_contract_digest(tuple(reversed(fixtures))) == expected_d0
    assert d1_draw_digest(scenarios) == d1_draw_digest(tuple(reversed(scenarios))) == expected_d1
    assert len(d0_contract_digest(fixtures)) == len(d1_draw_digest(scenarios)) == 64
    checkpoint = json.loads(Path("results/dynamic_mainline/checkpoints/task03.json").read_text())
    assert checkpoint["d0_contract_digest"] == d0_contract_digest(fixtures)
    assert checkpoint["d1_draw_digest"] == d1_draw_digest(scenarios)


def test_multi_event_d0_witnesses_encode_specific_order_agents_targets_and_faults() -> None:
    fixtures = {fixture.name: fixture for fixture in d0_scenarios()}

    paid_once = fixtures["first_new_destroyed_paid_once"]
    assert [step.action for step in paid_once.script] == [D0Action.ATTACK, D0Action.ATTACK]
    assert [event.realized_reward for event in paid_once.expected_private_audit_trace] == [100.0, 0.0]

    continuous = fixtures["continuous_attack_after_failure"]
    assert [step.physical_control.attack_success for step in continuous.script] == [False, True]
    assert [event.physical_success for event in continuous.expected_private_audit_trace] == [False, True]

    locked = fixtures["target_lock_prevents_overlap"]
    assert [(step.agent_id, step.target_id) for step in locked.script] == [(0, 0), (1, 0)]
    assert locked.script[1].fault is D0Fault.REJECT_LOCKED
    assert locked.script[1].precondition.locked_targets == (0,)

    batch = fixtures["simultaneous_completion_batch"]
    assert {(event.agent_id, event.target_id) for event in batch.expected_public_trace} == {(0, 0), (1, 1)}
    assert len({event.tick for event in batch.expected_public_trace}) == 1

    handoff = fixtures["cross_uav_sequential_handoff"]
    assert [(step.agent_id, step.action) for step in handoff.script] == [(0, D0Action.RECON), (1, D0Action.ATTACK)]
    assert handoff.script[1].tick > handoff.script[0].tick

    rejected = fixtures["horizon_range_ammo_rejection"]
    assert {step.fault for step in rejected.script} == {
        D0Fault.REJECT_HORIZON,
        D0Fault.REJECT_RANGE,
        D0Fault.REJECT_AMMO,
    }

    nonleak = fixtures["public_private_event_nonleakage"]
    assert [event.kind for event in nonleak.expected_public_trace] == [
        PublicRecordKind.OBSERVATION,
        PublicRecordKind.ATTACK_ACK,
    ]

    b1m = fixtures["b1m_frozen_suffix_auto_next"]
    assert len(b1m.script) == 2 and all(step.committed for step in b1m.script)

    discounted = fixtures["dynamic_absolute_discount_clock"]
    assert len(discounted.expected_private_audit_trace) == 2
    assert discounted.expected_private_audit_trace[1].tick > discounted.expected_private_audit_trace[0].tick


def test_negative_and_scheduler_d0_witnesses_do_not_fabricate_action_completions() -> None:
    fixtures = {fixture.name: fixture for fixture in d0_scenarios()}

    busy = fixtures["busy_uav_excluded_from_bid_and_gate"]
    assert busy.script[-1].precondition.busy_agents == (0,)
    assert any(step.fault is D0Fault.EXCLUDE_BUSY for step in busy.script)

    periodic = fixtures["completion_precedes_periodic_planning"]
    assert [event.kind for event in periodic.expected_public_trace] == [
        PublicRecordKind.ATTACK_ACK,
        PublicRecordKind.SCHEDULER,
    ]
    assert periodic.expected_public_trace[0].tick == periodic.expected_public_trace[1].tick
    assert periodic.expected_public_trace[0].event_id < periodic.expected_public_trace[1].event_id

    deferred = fixtures["defer_reactivated_by_event"]
    assert len(deferred.script) >= 2
    assert deferred.script[0].fault is D0Fault.FORCE_DEFER
    assert deferred.script[2].tick > deferred.script[0].tick

    no_positive = fixtures["no_positive_normal_termination"]
    assert [event.kind for event in no_positive.expected_public_trace] == [PublicRecordKind.SCHEDULER, PublicRecordKind.TERMINATION]
    assert no_positive.expected_private_audit_trace[-1].kind is PrivateRecordKind.TERMINATION

    stalled = fixtures["positive_task_zero_commit_stall_gate"]
    assert [event.kind for event in stalled.expected_public_trace] == [PublicRecordKind.SCHEDULER]
    assert all(event.kind is not PublicRecordKind.ATTACK_ACK for event in stalled.expected_public_trace)

    at_horizon = fixtures["completion_at_horizon_settles_first"]
    assert at_horizon.expected_public_trace[0].tick == at_horizon.scenario.t_max_tick


def test_every_d0_run_recomputes_legal_timeline_from_geometry_and_duration() -> None:
    config = dynamic_config_registry()[APPROVED_CONFIG_ID]
    duration = {
        D0Action.RECON: config.recon_duration,
        D0Action.ATTACK: config.attack_duration,
        D0Action.BDA: config.bda_duration,
    }
    for fixture in d0_scenarios():
        assert fixture.runs and all(isinstance(run, D0Run) for run in fixture.runs)
        for run in fixture.runs:
            agent_position = {agent.agent_id: agent.position for agent in run.scenario.agents}
            agent_free = {agent.agent_id: 0 for agent in run.scenario.agents}
            target_free = {target.target_id: 0 for target in run.scenario.targets}
            for index, step in enumerate(run.script):
                if not step.committed:
                    assert step.start_tick is step.finish_tick is None
                    assert step.counter_key is step.expected_uniform is None
                    assert any(item.step_index == index for item in run.expected_absences)
                    continue
                assert step.action is not None and step.agent_id is not None and step.target_id is not None
                assert step.fault is None
                assert step.counter_key is not None and step.expected_uniform == uniform01(step.counter_key)
                assert step.commit_tick >= agent_free[step.agent_id]
                assert step.commit_tick >= target_free[step.target_id]
                origin = agent_position[step.agent_id]
                destination = run.scenario.targets[step.target_id].position
                travel_tick = quantize_tick(math.hypot(destination[0] - origin[0], destination[1] - origin[1]) / config.speed, config.tick_size)
                assert step.start_tick == step.commit_tick + travel_tick
                assert step.finish_tick == step.start_tick + quantize_tick(duration[step.action], config.tick_size)
                assert step.finish_tick <= run.scenario.t_max_tick
                completions = [event for event in run.expected_public_trace if event.source_step_index == index and event.kind in {PublicRecordKind.OBSERVATION, PublicRecordKind.ATTACK_ACK}]
                audits = [event for event in run.expected_private_audit_trace if event.source_step_index == index and event.kind is PrivateRecordKind.COMPLETION]
                assert len(completions) == len(audits) == 1
                assert completions[0].tick == audits[0].tick == step.finish_tick
                assert audits[0].draw == step.expected_uniform
                agent_free[step.agent_id] = step.finish_tick
                target_free[step.target_id] = step.finish_tick
                agent_position[step.agent_id] = destination


def test_rejected_steps_have_typed_audits_absences_and_never_read_counter() -> None:
    fixture = next(item for item in d0_scenarios() if item.name == "horizon_range_ammo_rejection")
    run = fixture.runs[0]
    assert {event.rejection_reason for event in run.expected_private_audit_trace} == {
        RejectionReason.HORIZON,
        RejectionReason.RANGE,
        RejectionReason.AMMO,
    }
    assert all(not step.committed and step.counter_key is None and step.expected_uniform is None for step in run.script)
    assert not any(event.kind in {PublicRecordKind.OBSERVATION, PublicRecordKind.ATTACK_ACK} for event in run.expected_public_trace)
    assert len(run.expected_absences) == 9
    assert all(isinstance(item, ExpectedAbsence) for item in run.expected_absences)
    config = dynamic_config_registry()[APPROVED_CONFIG_ID]
    horizon, range_step, ammo_step = run.script
    horizon_target = run.scenario.targets[horizon.target_id].position  # type: ignore[index]
    horizon_origin = run.scenario.agents[horizon.agent_id].position  # type: ignore[index]
    horizon_finish = horizon.commit_tick + quantize_tick(
        math.hypot(horizon_target[0] - horizon_origin[0], horizon_target[1] - horizon_origin[1]) / config.speed
        + config.attack_duration,
        config.tick_size,
    )
    assert horizon_finish > run.scenario.t_max_tick
    range_target = run.scenario.targets[range_step.target_id].position  # type: ignore[index]
    range_origin = run.scenario.agents[range_step.agent_id].position  # type: ignore[index]
    assert math.hypot(range_target[0] - range_origin[0], range_target[1] - range_origin[1]) > run.scenario.agents[range_step.agent_id].distance.available  # type: ignore[index]
    assert run.scenario.agents[ammo_step.agent_id].ammo.available < 1.0  # type: ignore[index]


def test_ack_nonleakage_is_two_worlds_with_equal_public_and_different_private() -> None:
    fixture = next(item for item in d0_scenarios() if item.name == "attack_ack_hides_outcome")
    assert len(fixture.runs) == 2
    success, failure = fixture.runs
    assert success.scenario.targets == failure.scenario.targets
    assert success.expected_public_trace == failure.expected_public_trace
    assert success.expected_private_audit_trace != failure.expected_private_audit_trace
    assert success.expected_private_audit_trace[0].physical_success is True
    assert failure.expected_private_audit_trace[0].physical_success is False


def test_counter_replay_has_two_method_runs_with_fieldwise_shared_initial_state() -> None:
    fixture = next(item for item in d0_scenarios() if item.name == "counter_replay_shared_initial_truth")
    assert {run.method_id for run in fixture.runs} == {"P", "B1m"}
    left, right = fixture.runs
    assert left.scenario == right.scenario
    assert left.script == right.script
    assert left.expected_public_trace == right.expected_public_trace
    assert left.expected_private_audit_trace == right.expected_private_audit_trace


def test_busy_event_and_b1m_suffix_encode_full_intermediate_and_terminal_records() -> None:
    fixtures = {item.name: item for item in d0_scenarios()}
    busy = fixtures["busy_event_prevents_early_termination"].runs[0]
    scheduler = next(event for event in busy.expected_public_trace if event.kind is PublicRecordKind.SCHEDULER)
    assert scheduler.queue_depth == 1 and scheduler.idle_positive_tasks == 0 and scheduler.commits == 0
    completion = next(event for event in busy.expected_public_trace if event.kind is PublicRecordKind.ATTACK_ACK)
    termination = next(event for event in busy.expected_public_trace if event.kind is PublicRecordKind.TERMINATION)
    assert scheduler.tick < completion.tick == termination.tick

    b1m = fixtures["b1m_frozen_suffix_auto_next"].runs[0]
    assert len(b1m.script) == 2 and all(step.committed for step in b1m.script)
    assert b1m.script[1].commit_tick >= b1m.script[0].finish_tick
    assert len([event for event in b1m.expected_public_trace if event.kind is PublicRecordKind.ATTACK_ACK]) == 2
    assert len([event for event in b1m.expected_private_audit_trace if event.kind is PrivateRecordKind.COMPLETION]) == 2
    assert b1m.expected_public_trace[-1].kind is PublicRecordKind.TERMINATION


def test_all_committed_d0_outcomes_observations_and_rewards_follow_recorded_crn() -> None:
    config = dynamic_config_registry()[APPROVED_CONFIG_ID]
    recon_observations = (("H", "A"), ("H", "D"), ("L", "A"), ("L", "D"))
    for fixture in d0_scenarios():
        for run in fixture.runs:
            paid_targets: set[int] = set()
            for step_index, step in enumerate(run.script):
                if not step.committed:
                    continue
                assert step.counter_key is not None and step.expected_uniform == uniform01(step.counter_key)
                audit = next(
                    event for event in run.expected_private_audit_trace
                    if event.source_step_index == step_index and event.kind is PrivateRecordKind.COMPLETION
                )
                public = next(
                    event for event in run.expected_public_trace
                    if event.source_step_index == step_index and event.kind in {PublicRecordKind.ATTACK_ACK, PublicRecordKind.OBSERVATION}
                )
                if step.action is D0Action.ATTACK:
                    probability = config.attack_success_high if audit.true_category == "H" else config.attack_success_low
                    expected_success = audit.damage_before == "A" and step.expected_uniform < Fraction.from_float(probability)
                    assert audit.physical_success is expected_success
                    expected_after = "D" if expected_success else audit.damage_before
                    expected_reward = (config.value_high if audit.true_category == "H" else config.value_low) if expected_success else 0.0
                    assert audit.damage_after == expected_after
                    assert audit.realized_reward == expected_reward
                    expected_first_payment = expected_success and audit.target_id not in paid_targets
                    assert audit.first_destroyed_payment is expected_first_payment
                    if expected_first_payment:
                        paid_targets.add(audit.target_id)
                    assert public.observation is None
                elif step.action is D0Action.RECON:
                    class_col = 0 if audit.true_category == "H" else 1
                    damage_col = 0 if audit.damage_before == "A" else 1
                    probabilities = tuple(
                        config.recon_category_matrix[row // 2][class_col]
                        * config.recon_damage_matrix[row % 2][damage_col]
                        for row in range(4)
                    )
                    assert public.observation == recon_observations[categorical(step.counter_key, probabilities)]
                    assert audit.realized_reward == 0.0
                elif step.action is D0Action.BDA:
                    damage_col = 0 if audit.damage_before == "A" else 1
                    probabilities = tuple(config.bda_damage_matrix[row][damage_col] for row in range(2))
                    assert public.observation == ("A", "D")[categorical(step.counter_key, probabilities)]
                    assert audit.realized_reward == 0.0


def test_all_d0_preconditions_replay_public_belief_completion_first() -> None:
    config = dynamic_config_registry()[APPROVED_CONFIG_ID]
    recon_z = recon_kernel(config.recon_category_matrix, config.recon_damage_matrix)
    bda_z = bda_kernel(config.bda_damage_matrix)
    recon_order = (("H", "A"), ("H", "D"), ("L", "A"), ("L", "D"))
    for fixture in d0_scenarios():
        for run in fixture.runs:
            beliefs = [target.belief for target in run.scenario.targets]
            pending: list[tuple[int, int, D0Action, object]] = []
            for step_index, step in enumerate(run.script):
                ready = sorted((item for item in pending if item[0] <= step.commit_tick), key=lambda item: item[0])
                for finish_tick, target_id, action, observation in ready:
                    if action is D0Action.ATTACK:
                        beliefs[target_id] = tuple(predict_attack(beliefs[target_id], config.attack_success_high, config.attack_success_low))
                    elif action is D0Action.RECON:
                        beliefs[target_id] = tuple(bayes_update(beliefs[target_id], recon_z, recon_order.index(observation)))
                    else:
                        beliefs[target_id] = tuple(bayes_update(beliefs[target_id], bda_z, ("A", "D").index(observation)))
                    pending.remove((finish_tick, target_id, action, observation))
                assert step.precondition.beliefs == tuple(beliefs)
                if step.committed:
                    public = next(event for event in run.expected_public_trace if event.source_step_index == step_index)
                    pending.append((step.finish_tick, step.target_id, step.action, public.observation))  # type: ignore[arg-type]
                    expected = list(beliefs)
                    if step.action is D0Action.ATTACK:
                        expected[step.target_id] = tuple(predict_attack(expected[step.target_id], config.attack_success_high, config.attack_success_low))  # type: ignore[index]
                    elif step.action is D0Action.RECON:
                        expected[step.target_id] = tuple(bayes_update(expected[step.target_id], recon_z, recon_order.index(public.observation)))  # type: ignore[index,arg-type]
                    else:
                        expected[step.target_id] = tuple(bayes_update(expected[step.target_id], bda_z, ("A", "D").index(public.observation)))  # type: ignore[index,arg-type]
                    assert public.posterior_belief == expected[step.target_id]  # type: ignore[index]


def test_recon_fixture_and_busy_exclusion_have_typed_operands() -> None:
    fixtures = {item.name: item for item in d0_scenarios()}
    recon = fixtures["recon_joint_bayes"]
    recon_posterior = recon.expected_public_trace[0].posterior_belief
    assert recon_posterior is not None
    assert any(operand.name.value == "expected_posterior" and operand.value == recon_posterior for operand in recon.focused_assertions[0].operands)

    busy = fixtures["busy_uav_excluded_from_bid_and_gate"]
    scheduler = next(event for event in busy.expected_public_trace if event.kind is PublicRecordKind.SCHEDULER)
    assert scheduler.idle_positive_tasks == 1 and scheduler.commits == 1
    committed = [step for step in busy.script if step.committed]
    assert [(step.agent_id, step.target_id) for step in committed] == [(0, 0), (1, 1)]
    agent1_step_index = busy.script.index(committed[1])
    assert committed[1].precondition.busy_agents == (0,)
    assert committed[1].precondition.locked_targets == (0,)
    assert any(
        event.source_step_index == agent1_step_index and event.kind is PublicRecordKind.ATTACK_ACK
        for event in busy.expected_public_trace
    )
    assert any(
        event.source_step_index == agent1_step_index and event.kind is PrivateRecordKind.COMPLETION
        for event in busy.expected_private_audit_trace
    )
    assert busy.expected_gate is None
    operands = {operand.name.value: operand.value for operand in busy.focused_assertions[0].operands}
    assert operands["expected_bidders"] == (1,)
    assert operands["expected_gate_agents"] == (1,)
