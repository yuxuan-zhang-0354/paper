from dataclasses import FrozenInstanceError, asdict, fields
import inspect
import math
from typing import get_args

import pytest

from uav_lifecycle.dynamic_rng import DrawKey
from uav_lifecycle.dynamic_types import (
    ActionCommit,
    CompletionEvent,
    DynamicConfig,
    DynamicScenario,
    EpisodeRecord,
    EpisodeResult,
    GateFailure,
    InternalAgentState,
    PrivateAuditEvent,
    PrivateTarget,
    PublicAgent,
    PublicActionAck,
    PublicBusyAction,
    PublicCompletion,
    PublicObservation,
    PublicPolicy,
    PublicSnapshot,
    PublicTarget,
    ResourceLedger,
    private_truth_digest,
    public_digest,
    quantize_tick,
    validate_runtime_invariants,
)


def _target(target_id: int = 0, belief: tuple[float, ...] = (0.4, 0.1, 0.2, 0.3)) -> PublicTarget:
    return PublicTarget(target_id, (3.0, 4.0), belief)


def _commit(agent_id: int = 0, target_id: int = 0, ordinal: int = 0) -> ActionCommit:
    return ActionCommit(
        agent_id=agent_id,
        target_id=target_id,
        mode="attack",
        origin=(0.0, 0.0),
        destination=(3.0, 4.0),
        travel=5.0,
        reserved_ammo=1.0,
        reserved_distance=5.0,
        commit_tick=0,
        start_tick=50,
        finish_tick=70,
        ordinal=ordinal,
    )


def _internal(agent_id: int = 0, commit: ActionCommit | None = None) -> InternalAgentState:
    return InternalAgentState(
        agent_id=agent_id,
        position=(0.0, 0.0),
        ammo=ResourceLedger(2.0 if commit is None else 1.0, 0.0 if commit is None else 1.0, 0.0),
        distance=ResourceLedger(10.0 if commit is None else 5.0, 0.0 if commit is None else 5.0, 0.0),
        initial_ammo_total=2.0,
        initial_distance_total=10.0,
        active_action=commit,
    )


def _record(**changes: object) -> EpisodeRecord:
    values: dict[str, object] = {
        "scenario_id": "s0",
        "cell_id": "c0",
        "method": "greedy",
        "initial_truth_digest": "a" * 64,
        "final_truth_digest": "b" * 64,
        "public_initial_digest": "c" * 64,
        "termination": "normal",
        "status": "ok",
        "event_count": 2,
        "action_count": 1,
        "replan_count": 1,
        "final_beliefs": ((0.4, 0.1, 0.2, 0.3),),
        "destroyed_value": 100.0,
        "service_cost": 6.0,
        "distance_cost": 0.5,
        "ammo_cost": 0.5,
        "realized_utility": 93.0,
        "normalized_utility": 0.93,
        "gross_scenario_value": 100.0,
        "distance_consumed": 5.0,
        "ammo_consumed": 1.0,
        "makespan": 7.0,
        "first_destroyed_value": 100.0,
        "invalid_attack_count": 0,
        "initial_wreck_attack_count": 0,
        "recon_count": 0,
        "bda_count": 0,
        "continuous_attack_count": 0,
        "handoff_count": 0,
        "orphan_count": 0,
        "stall_count": 0,
        "cbba_round_count": 0,
        "final_joint_brier_score": 0.1,
        "allocator_gates": (),
        "replay_audit": "verified",
    }
    values.update(changes)
    return EpisodeRecord(**values)  # type: ignore[arg-type]


def test_quantize_tick_uses_binary64_half_even() -> None:
    assert quantize_tick(1.25, 0.5) == 2
    assert quantize_tick(1.75, 0.5) == 4


def test_quantize_tick_distinguishes_binary64_neighbors_of_tie() -> None:
    tie = 1.25
    assert quantize_tick(math.nextafter(tie, 0.0), 0.5) == 2
    assert quantize_tick(math.nextafter(tie, math.inf), 0.5) == 3


def test_hypot_distance_and_horizon_use_same_tick_contract() -> None:
    tick_size = 1e-10
    assert quantize_tick(math.hypot(3.0, 4.0), tick_size) == 50_000_000_000
    assert quantize_tick(300.0, tick_size) == 3_000_000_000_000


@pytest.mark.parametrize(
    ("value", "tick_size"),
    [(-1.0, 1.0), (math.inf, 1.0), (math.nan, 1.0), (1.0, 0.0), (1.0, -1.0), (1.0, math.inf), (1.0, math.nan)],
)
def test_quantize_tick_rejects_invalid_inputs(value: float, tick_size: float) -> None:
    with pytest.raises(ValueError):
        quantize_tick(value, tick_size)


@pytest.mark.parametrize(
    "values",
    [(-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (math.inf, 0.0, 0.0), (math.nan, 0.0, 0.0)],
)
def test_ledger_rejects_negative_or_nonfinite_fields(values: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        ResourceLedger(*values)


def test_ledger_reserve_and_consume_are_atomic_conservative_transitions() -> None:
    initial = ResourceLedger(10.0, 0.0, 2.0)
    reserved = initial.reserve(3.0)
    completed = reserved.consume(3.0)
    assert initial == ResourceLedger(10.0, 0.0, 2.0)
    assert reserved == ResourceLedger(7.0, 3.0, 2.0)
    assert completed == ResourceLedger(7.0, 0.0, 5.0)
    assert initial.total == reserved.total == completed.total == 12.0


def test_failed_ledger_transitions_do_not_mutate_or_allow_release() -> None:
    ledger = ResourceLedger(2.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        ledger.reserve(3.0)
    for amount in (-1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            ledger.reserve(amount)
    committed = ledger.reserve(2.0)
    completed = committed.consume(2.0)
    with pytest.raises(ValueError):
        completed.consume(2.0)
    with pytest.raises(FrozenInstanceError):
        committed.reserved = 0.0  # type: ignore[misc]
    assert not hasattr(committed, "cancel")


def test_dynamic_config_contains_every_frozen_parameter_and_serializes() -> None:
    config = DynamicConfig()
    assert asdict(config) == {
        "config_id": "recon_damage_plus_010_r2_a6_b3",
        "value_high": 100.0,
        "value_low": 30.0,
        "attack_success_high": 0.4,
        "attack_success_low": 0.75,
        "recon_duration": 4.0,
        "attack_duration": 2.0,
        "bda_duration": 1.5,
        "minimum_duration": 1.5,
        "recon_service_cost": 2.0,
        "attack_service_cost": 6.0,
        "bda_service_cost": 3.0,
        "discount_rate": 0.02,
        "distance_cost_rate": 0.1,
        "ammo_cost_rate": 0.5,
        "speed": 1.0,
        "recon_category_matrix": ((0.65, 0.15), (0.35, 0.85)),
        "recon_damage_matrix": ((0.85, 0.15), (0.15, 0.85)),
        "bda_damage_matrix": ((0.92, 0.06), (0.08, 0.94)),
        "tick_size": 1e-10,
    }
    with pytest.raises(FrozenInstanceError):
        config.speed = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"value_high": 101.0}, "frozen matched-model"),
        ({"attack_success_high": 1.01}, "probability"),
        ({"attack_duration": 0.0}, "positive"),
        ({"minimum_duration": 0.0}, "positive"),
        ({"recon_category_matrix": ((0.5, 0.5), (0.5, 0.5))}, "frozen matched-model"),
    ],
)
def test_dynamic_config_rejects_any_override_of_approved_model(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DynamicConfig(**changes)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "belief",
    [(-0.1, 0.2, 0.3, 0.6), (0.2, 0.2, 0.2, 0.2), (math.nan, 0.0, 0.0, 1.0), (0.5, 0.5, 0.0)],
)
def test_public_target_requires_four_element_probability_simplex(belief: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        _target(belief=belief)


def test_public_projection_exposes_only_available_resources_and_allowed_busy_fields() -> None:
    busy = PublicBusyAction(target_id=0, destination=(3.0, 4.0), finish_tick=70)
    agent = PublicAgent(0, (0.0, 0.0), available_ammo=1.0, available_distance=5.0, busy_action=busy)
    assert {field.name for field in fields(agent)} == {
        "agent_id", "position", "available_ammo", "available_distance", "busy_action"
    }
    assert {field.name for field in fields(busy)} == {"target_id", "destination", "finish_tick"}


def test_public_types_and_policy_protocol_have_no_forbidden_information_fields() -> None:
    forbidden = {
        "scenario_id", "cell_id", "seed", "crn_namespace", "truth", "true_category", "true_damage",
        "hit", "miss", "destroyed", "physical_success", "realized_reward", "reward", "private_digest",
        "future_draw", "draw", "rng",
    }
    for public_type in (PublicSnapshot, PublicObservation, PublicActionAck):
        assert forbidden.isdisjoint(field.name for field in fields(public_type))
    assert set(get_args(PublicCompletion)) == {PublicObservation, PublicActionAck}
    decision = inspect.signature(PublicPolicy.decide)
    assert list(decision.parameters) == ["self", "snapshot"]
    assert decision.parameters["snapshot"].annotation in {PublicSnapshot, "PublicSnapshot"}
    assert decision.return_annotation == "tuple[tuple[int, int, Mode], ...]"


def test_public_snapshot_validates_contiguous_ids_busy_state_and_lock_ownership() -> None:
    busy = PublicBusyAction(0, (3.0, 4.0), 70)
    valid = PublicSnapshot(
        tick=0,
        targets=(_target(),),
        agents=(PublicAgent(0, (0.0, 0.0), 1.0, 5.0, busy),),
        target_locks=((0, 0),),
        observations=(),
        acknowledgements=(),
    )
    assert valid.target_locks == ((0, 0),)
    with pytest.raises(ValueError, match="contiguous"):
        PublicSnapshot(0, (_target(1),), valid.agents, ((0, 0),), (), ())
    with pytest.raises(ValueError, match="lock"):
        PublicSnapshot(0, valid.targets, valid.agents, (), (), ())
    with pytest.raises(ValueError, match="finish"):
        PublicBusyAction(0, (3.0, 4.0), 0)


def test_public_observation_and_attack_ack_have_disjoint_validated_payloads() -> None:
    recon = PublicObservation(0, 60, 0, 0, "recon", ("H", "A"))
    bda = PublicObservation(1, 70, 0, 0, "bda", "D")
    ack = PublicActionAck(2, 80, 0, 0, "attack")
    assert recon.observation == ("H", "A")
    assert bda.observation == "D"
    assert {field.name for field in fields(ack)} == {"event_id", "tick", "target_id", "agent_id", "mode"}
    with pytest.raises(ValueError, match="Recon"):
        PublicObservation(0, 60, 0, 0, "recon", ("unknown", "A"))
    with pytest.raises(ValueError, match="BDA"):
        PublicObservation(0, 60, 0, 0, "bda", ("H", "A"))
    with pytest.raises(ValueError, match="Observation"):
        PublicObservation(0, 60, 0, 0, "attack", "A")
    with pytest.raises(ValueError, match="Attack"):
        PublicActionAck(0, 60, 0, 0, "recon")
    with pytest.raises(TypeError):
        PublicActionAck(0, 60, 0, 0, "attack", "destroyed")  # type: ignore[call-arg]


def test_public_snapshot_rejects_observation_ack_collection_mixing() -> None:
    target = (_target(),)
    agents = (PublicAgent(0, (0.0, 0.0), 2.0, 10.0, None),)
    observation = PublicObservation(0, 60, 0, 0, "recon", ("H", "A"))
    ack = PublicActionAck(0, 70, 0, 0, "attack")
    PublicSnapshot(70, target, agents, (), (observation,), (ack,))
    with pytest.raises(TypeError, match="observations"):
        PublicSnapshot(70, target, agents, (), (ack,), ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="acknowledgements"):
        PublicSnapshot(70, target, agents, (), (), (observation,))  # type: ignore[arg-type]


def test_internal_agent_validates_idle_busy_and_resource_conservation() -> None:
    assert _internal().active_action is None
    assert _internal(commit=_commit()).active_action is not None
    with pytest.raises(ValueError, match="conservation"):
        InternalAgentState(0, (0.0, 0.0), ResourceLedger(1, 0, 0), ResourceLedger(10, 0, 0), 2, 10, None)
    with pytest.raises(ValueError, match="agent"):
        _internal(agent_id=1, commit=_commit(agent_id=0))


def test_runtime_validator_enforces_one_action_per_uav_target_and_event_tie_keys() -> None:
    commit = _commit()
    event = CompletionEvent.from_action(commit)
    assert event.heap_key == (70, 0, 0)
    validate_runtime_invariants((_internal(commit=commit),), ((0, 0),), (event,))
    duplicate_target = _commit(agent_id=1, target_id=0, ordinal=1)
    with pytest.raises(ValueError, match="target"):
        validate_runtime_invariants(
            (_internal(commit=commit), _internal(agent_id=1, commit=duplicate_target)),
            ((0, 0),),
            (event, CompletionEvent.from_action(duplicate_target)),
        )
    with pytest.raises(ValueError, match="event"):
        validate_runtime_invariants((_internal(commit=commit),), ((0, 0),), ())


def test_scenario_validates_canonical_public_private_and_agent_ids() -> None:
    scenario = DynamicScenario(
        scenario_id="s0",
        cell_id="c0",
        seed=42,
        crn_namespace="pilot",
        targets=(_target(),),
        private_targets=(PrivateTarget(0, "H", "A", False),),
        agents=(_internal(),),
        t_max_tick=100,
    )
    assert scenario.seed == 42 and scenario.t_max_tick == 100
    with pytest.raises(ValueError, match="t_max_tick"):
        DynamicScenario("s0", "c0", 42, "pilot", (_target(),), (PrivateTarget(0, "H", "A", False),), (_internal(),), 0)
    with pytest.raises(ValueError, match="private"):
        DynamicScenario("s0", "c0", 42, "pilot", (_target(),), (PrivateTarget(1, "H", "A", False),), (_internal(),), 100)


def test_public_and_private_digests_are_canonical_and_information_separated() -> None:
    def project_public(context: DynamicScenario) -> PublicSnapshot:
        public_agents = tuple(
            PublicAgent(agent.agent_id, agent.position, agent.ammo.available, agent.distance.available, None)
            for agent in context.agents
        )
        return PublicSnapshot(0, context.targets, public_agents, (), (), ())

    context_a = DynamicScenario(
        "s0", "c0", 1, "pilot", (_target(),), (PrivateTarget(0, "H", "A", False),), (_internal(),), 100
    )
    context_b = DynamicScenario(
        "s0", "c0", 999, "different-private-namespace", (_target(),),
        (PrivateTarget(0, "L", "D", False),), (_internal(),), 100
    )
    snapshot_a = project_public(context_a)
    snapshot_b = project_public(context_b)
    assert asdict(snapshot_a) == asdict(snapshot_b)
    assert private_truth_digest(context_a.private_targets) != private_truth_digest(context_b.private_targets)
    assert public_digest(snapshot_a) == public_digest(snapshot_b)
    assert len(public_digest(snapshot_a)) == 64


def test_public_and_private_events_are_distinct_types_and_collections() -> None:
    public = PublicActionAck(0, 70, 0, 0, "attack")
    key = DrawKey("v", "e", "g", "c", 0, "target", 0, "attack", 0, 0)
    private = PrivateAuditEvent(
        0, 70, 0, 0, "attack", 0.2, "H", "A", "D", True, 100.0,
        False, False, key,
    )
    result = EpisodeResult(_record(), (public,), (private,), (), (_commit(),))
    assert result.public_events == (public,)
    assert result.private_audit_events == (private,)
    assert type(public) is not type(private)
    with pytest.raises(TypeError):
        EpisodeResult(_record(), (private,), (private,), (), (_commit(),))  # type: ignore[arg-type]


def test_episode_record_has_complete_episode_method_schema_and_validates_decomposition() -> None:
    record = _record(allocator_gates=(GateFailure("atomic_commit", 0, "duplicate target", (("target", "0"),)),))
    serialized = asdict(record)
    required = {
        "scenario_id", "cell_id", "method", "initial_truth_digest", "final_truth_digest",
        "public_initial_digest", "termination", "status", "event_count", "action_count", "replan_count",
        "final_beliefs", "destroyed_value", "service_cost", "distance_cost", "ammo_cost", "realized_utility",
        "normalized_utility", "gross_scenario_value", "distance_consumed", "ammo_consumed", "makespan",
        "first_destroyed_value", "invalid_attack_count", "initial_wreck_attack_count", "recon_count", "bda_count",
        "continuous_attack_count", "handoff_count", "orphan_count", "stall_count", "cbba_round_count",
        "final_joint_brier_score", "allocator_gates", "replay_audit",
    }
    assert required == set(serialized)
    with pytest.raises(ValueError, match="decomposition"):
        _record(realized_utility=92.0)
    with pytest.raises(ValueError, match="normalized"):
        _record(normalized_utility=0.92)


def test_event_and_action_ordinals_and_ticks_are_validated() -> None:
    with pytest.raises(ValueError, match="ordinal"):
        _commit(ordinal=-1)
    with pytest.raises(ValueError, match="later"):
        ActionCommit(0, 0, "attack", (0.0, 0.0), (1.0, 0.0), 1.0, 1.0, 1.0, 2, 2, 2, 0)
    with pytest.raises(ValueError, match="mode"):
        ActionCommit(0, 0, "secret", (0.0, 0.0), (1.0, 0.0), 1.0, 0.0, 1.0, 0, 1, 2, 0)
