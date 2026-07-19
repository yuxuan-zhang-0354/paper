"""Immutable public/private boundary types for the dynamic lifecycle simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from decimal import Decimal, ROUND_HALF_EVEN
from enum import Enum
import hashlib
import json
import math
from typing import Any, Protocol

from .dynamic_rng import DrawKey


Position = tuple[float, float]
Belief = tuple[float, float, float, float]
Mode = str
_MODES = frozenset({"recon", "attack", "bda"})
_CATEGORIES = frozenset({"H", "L"})
_DAMAGE_STATES = frozenset({"A", "D"})


class PlanningClock(str, Enum):
    """Public scheduling contract for policy invocation epochs."""

    EVENT_DRIVEN = "event_driven"
    PERIODIC = "periodic"
    B1M_ONE_SHOT = "b1m_one_shot"
    ONE_SHOT = "b1m_one_shot"  # Backward-compatible alias; reserved for B1m.


def _require_nonnegative_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _validate_position(position: Position, name: str = "position") -> None:
    if len(position) != 2 or any(not math.isfinite(value) for value in position):
        raise ValueError(f"{name} must contain two finite coordinates")


def _validate_belief(belief: tuple[float, ...], name: str = "belief") -> None:
    if len(belief) != 4:
        raise ValueError(f"{name} must have four components")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in belief):
        raise ValueError(f"{name} must contain finite probabilities")
    if not math.isclose(math.fsum(belief), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} must lie on the probability simplex")


def _validate_contiguous_ids(items: tuple[Any, ...], label: str) -> None:
    ids = tuple(item.target_id if hasattr(item, "target_id") else item.agent_id for item in items)
    if ids != tuple(range(len(items))):
        raise ValueError(f"{label} IDs must be contiguous, canonical, and ordered from zero")


def _validate_mode(mode: Mode) -> None:
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}")


def quantize_tick(value: float, tick_size: float) -> int:
    """Quantize a nonnegative binary64 value using frozen half-even semantics."""

    _require_nonnegative_finite(value, "value")
    if not math.isfinite(tick_size) or tick_size <= 0:
        raise ValueError("tick_size must be finite and positive")
    scaled = value / tick_size
    if not math.isfinite(scaled):
        raise ValueError("value/tick_size must be finite")
    return int(Decimal.from_float(scaled).to_integral_value(rounding=ROUND_HALF_EVEN))


@dataclass(frozen=True, slots=True)
class ResourceLedger:
    """Immutable available/reserved/consumed resource accounting."""

    available: float
    reserved: float
    consumed: float

    def __post_init__(self) -> None:
        _require_nonnegative_finite(self.available, "available")
        _require_nonnegative_finite(self.reserved, "reserved")
        _require_nonnegative_finite(self.consumed, "consumed")

    @property
    def total(self) -> float:
        return math.fsum((self.available, self.reserved, self.consumed))

    def reserve(self, amount: float) -> ResourceLedger:
        """Atomically move an available amount into committed reservation."""

        _require_nonnegative_finite(amount, "amount")
        if amount > self.available:
            raise ValueError("cannot reserve more than is available")
        return replace(self, available=self.available - amount, reserved=self.reserved + amount)

    def consume(self, amount: float) -> ResourceLedger:
        """Atomically complete a commitment by consuming its reservation."""

        _require_nonnegative_finite(amount, "amount")
        if amount > self.reserved:
            raise ValueError("cannot consume more than is reserved")
        return replace(self, reserved=self.reserved - amount, consumed=self.consumed + amount)


def _validate_dynamic_config_values(config: object) -> None:
    for field_name in (
        "value_high", "value_low", "attack_success_high", "attack_success_low",
        "recon_duration", "attack_duration", "bda_duration", "minimum_duration",
        "recon_service_cost", "attack_service_cost", "bda_service_cost", "discount_rate",
        "distance_cost_rate", "ammo_cost_rate", "speed", "tick_size",
    ):
        _require_nonnegative_finite(getattr(config, field_name), field_name)
    if not 0.0 <= getattr(config, "attack_success_high") <= 1.0 or not 0.0 <= getattr(config, "attack_success_low") <= 1.0:
        raise ValueError("attack success probability must lie in [0, 1]")
    if any(
        getattr(config, name) <= 0.0
        for name in ("recon_duration", "attack_duration", "bda_duration", "minimum_duration")
    ):
        raise ValueError("action durations and minimum_duration must be positive")
    if getattr(config, "speed") <= 0 or getattr(config, "tick_size") <= 0:
        raise ValueError("speed and tick_size must be positive")
    for name in ("recon_category_matrix", "recon_damage_matrix", "bda_damage_matrix"):
        matrix = getattr(config, name)
        if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
            raise ValueError(f"{name} must be 2 by 2")
        for column in zip(*matrix, strict=True):
            if any(not math.isfinite(value) or value < 0 for value in column) or not math.isclose(
                math.fsum(column), 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"{name} columns must be probability simplexes")


@dataclass(frozen=True, slots=True)
class DynamicConfig:
    config_id: str = "recon_damage_plus_010_r2_a6_b3"
    value_high: float = 100.0
    value_low: float = 30.0
    attack_success_high: float = 0.40
    attack_success_low: float = 0.75
    recon_duration: float = 4.0
    attack_duration: float = 2.0
    bda_duration: float = 1.5
    minimum_duration: float = 1.5
    recon_service_cost: float = 2.0
    attack_service_cost: float = 6.0
    bda_service_cost: float = 3.0
    discount_rate: float = 0.02
    distance_cost_rate: float = 0.10
    ammo_cost_rate: float = 0.50
    speed: float = 1.0
    recon_category_matrix: tuple[tuple[float, float], tuple[float, float]] = ((0.65, 0.15), (0.35, 0.85))
    recon_damage_matrix: tuple[tuple[float, float], tuple[float, float]] = ((0.85, 0.15), (0.15, 0.85))
    bda_damage_matrix: tuple[tuple[float, float], tuple[float, float]] = ((0.92, 0.06), (0.08, 0.94))
    tick_size: float = 1e-10

    def __post_init__(self) -> None:
        _validate_dynamic_config_values(self)
        approved = {
            "config_id": "recon_damage_plus_010_r2_a6_b3",
            "value_high": 100.0,
            "value_low": 30.0,
            "attack_success_high": 0.40,
            "attack_success_low": 0.75,
            "recon_duration": 4.0,
            "attack_duration": 2.0,
            "bda_duration": 1.5,
            "minimum_duration": 1.5,
            "recon_service_cost": 2.0,
            "attack_service_cost": 6.0,
            "bda_service_cost": 3.0,
            "discount_rate": 0.02,
            "distance_cost_rate": 0.10,
            "ammo_cost_rate": 0.50,
            "speed": 1.0,
            "recon_category_matrix": ((0.65, 0.15), (0.35, 0.85)),
            "recon_damage_matrix": ((0.85, 0.15), (0.15, 0.85)),
            "bda_damage_matrix": ((0.92, 0.06), (0.08, 0.94)),
            "tick_size": 1e-10,
        }
        changed = tuple(name for name, expected in approved.items() if getattr(self, name) != expected)
        if changed:
            raise ValueError(f"frozen matched-model fields cannot be overridden: {', '.join(changed)}")


@dataclass(frozen=True, slots=True)
class ExperimentalDynamicConfig(DynamicConfig):
    """D3-only preference configuration; D2 still uses the frozen base class."""

    experiment_profile: str = "balanced"

    def __post_init__(self) -> None:
        _validate_dynamic_config_values(self)
        if not self.config_id.startswith("d3-") or not self.experiment_profile:
            raise ValueError("experimental configurations require a d3 config ID and profile")


@dataclass(frozen=True, slots=True)
class EnvironmentModel:
    """Simulator-private physical model, separate from the planner's beliefs."""

    attack_success_high: float
    attack_success_low: float
    recon_category_matrix: tuple[tuple[float, float], tuple[float, float]]
    recon_damage_matrix: tuple[tuple[float, float], tuple[float, float]]
    bda_damage_matrix: tuple[tuple[float, float], tuple[float, float]]
    model_id: str = "nominal"

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be nonempty")
        proxy = type("EnvironmentProxy", (), {
            "value_high": 1.0, "value_low": 1.0,
            "attack_success_high": self.attack_success_high,
            "attack_success_low": self.attack_success_low,
            "recon_duration": 1.0, "attack_duration": 1.0,
            "bda_duration": 1.0, "minimum_duration": 1.0,
            "recon_service_cost": 0.0, "attack_service_cost": 0.0,
            "bda_service_cost": 0.0, "discount_rate": 0.0,
            "distance_cost_rate": 0.0, "ammo_cost_rate": 0.0,
            "speed": 1.0, "tick_size": 1.0,
            "recon_category_matrix": self.recon_category_matrix,
            "recon_damage_matrix": self.recon_damage_matrix,
            "bda_damage_matrix": self.bda_damage_matrix,
        })()
        _validate_dynamic_config_values(proxy)

    @classmethod
    def from_config(cls, config: DynamicConfig) -> "EnvironmentModel":
        return cls(
            config.attack_success_high, config.attack_success_low,
            config.recon_category_matrix, config.recon_damage_matrix,
            config.bda_damage_matrix,
        )


@dataclass(frozen=True, slots=True)
class PublicTarget:
    target_id: int
    position: Position
    belief: Belief

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.target_id, "target_id")
        _validate_position(self.position)
        _validate_belief(self.belief)


@dataclass(frozen=True, slots=True)
class PublicBusyAction:
    target_id: int
    destination: Position
    finish_tick: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.target_id, "target_id")
        _validate_position(self.destination, "destination")
        _require_nonnegative_int(self.finish_tick, "finish_tick")
        if self.finish_tick == 0:
            raise ValueError("finish_tick for a busy action must be positive")


@dataclass(frozen=True, slots=True)
class PublicAgent:
    agent_id: int
    position: Position
    available_ammo: float
    available_distance: float
    busy_action: PublicBusyAction | None

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.agent_id, "agent_id")
        _validate_position(self.position)
        _require_nonnegative_finite(self.available_ammo, "available_ammo")
        _require_nonnegative_finite(self.available_distance, "available_distance")


@dataclass(frozen=True, slots=True)
class PublicObservation:
    event_id: int
    tick: int
    target_id: int
    agent_id: int
    mode: Mode
    observation: tuple[str, str] | str

    def __post_init__(self) -> None:
        for name in ("event_id", "tick", "target_id", "agent_id"):
            _require_nonnegative_int(getattr(self, name), name)
        if self.mode == "recon":
            if (
                not isinstance(self.observation, tuple)
                or len(self.observation) != 2
                or self.observation[0] not in _CATEGORIES
                or self.observation[1] not in _DAMAGE_STATES
            ):
                raise ValueError("Recon observation must be a valid (category, damage) pair")
        elif self.mode == "bda":
            if not isinstance(self.observation, str) or self.observation not in _DAMAGE_STATES:
                raise ValueError("BDA observation must be a valid damage label")
        else:
            raise ValueError("PublicObservation mode must be recon or bda")


@dataclass(frozen=True, slots=True)
class PublicActionAck:
    event_id: int
    tick: int
    target_id: int
    agent_id: int
    mode: Mode

    def __post_init__(self) -> None:
        for name in ("event_id", "tick", "target_id", "agent_id"):
            _require_nonnegative_int(getattr(self, name), name)
        if self.mode != "attack":
            raise ValueError("PublicActionAck is valid only for Attack")


PublicCompletion = PublicObservation | PublicActionAck


@dataclass(frozen=True, slots=True)
class PublicSnapshot:
    tick: int
    targets: tuple[PublicTarget, ...]
    agents: tuple[PublicAgent, ...]
    target_locks: tuple[tuple[int, int], ...]
    observations: tuple[PublicObservation, ...]
    acknowledgements: tuple[PublicActionAck, ...]

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.tick, "tick")
        _validate_contiguous_ids(self.targets, "target")
        _validate_contiguous_ids(self.agents, "agent")
        if any(not isinstance(item, PublicObservation) for item in self.observations):
            raise TypeError("observations accept only PublicObservation")
        if any(not isinstance(item, PublicActionAck) for item in self.acknowledgements):
            raise TypeError("acknowledgements accept only PublicActionAck")
        lock_map = dict(self.target_locks)
        if len(lock_map) != len(self.target_locks):
            raise ValueError("each target may have only one lock")
        busy_locks: dict[int, int] = {}
        for agent in self.agents:
            if agent.busy_action is not None:
                target_id = agent.busy_action.target_id
                if target_id in busy_locks:
                    raise ValueError("one active action per target is required")
                if target_id >= len(self.targets):
                    raise ValueError("busy action references unknown target")
                if agent.busy_action.finish_tick <= self.tick:
                    raise ValueError("busy action finish_tick must be later than snapshot tick")
                busy_locks[target_id] = agent.agent_id
        if lock_map != busy_locks:
            raise ValueError("target lock ownership must exactly match busy actions")


class PublicPolicy(Protocol):
    """Information-safe policy boundary: the policy receives public state only."""

    planning_clock: PlanningClock

    def decide(self, snapshot: PublicSnapshot) -> tuple[tuple[int, int, Mode], ...]: ...


@dataclass(frozen=True, slots=True)
class PrivateTarget:
    target_id: int
    true_category: str
    true_damage: str
    first_destroyed_paid: bool

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.target_id, "target_id")
        if self.true_category not in _CATEGORIES:
            raise ValueError("true_category must be H or L")
        if self.true_damage not in _DAMAGE_STATES:
            raise ValueError("true_damage must be A or D")
        if self.first_destroyed_paid and self.true_damage != "D":
            raise ValueError("first-destroyed payment requires destroyed truth")


@dataclass(frozen=True, slots=True)
class ActionCommit:
    agent_id: int
    target_id: int
    mode: Mode
    origin: Position
    destination: Position
    travel: float
    reserved_ammo: float
    reserved_distance: float
    commit_tick: int
    start_tick: int
    finish_tick: int
    ordinal: int

    def __post_init__(self) -> None:
        for name in ("agent_id", "target_id", "commit_tick", "start_tick", "finish_tick", "ordinal"):
            _require_nonnegative_int(getattr(self, name), name)
        _validate_mode(self.mode)
        _validate_position(self.origin, "origin")
        _validate_position(self.destination, "destination")
        for name in ("travel", "reserved_ammo", "reserved_distance"):
            _require_nonnegative_finite(getattr(self, name), name)
        if self.start_tick < self.commit_tick or self.finish_tick <= self.start_tick:
            raise ValueError("action start cannot precede commit and finish must be strictly later")
        if self.reserved_distance != self.travel:
            raise ValueError("reserved_distance must equal travel")
        expected_ammo = 1.0 if self.mode == "attack" else 0.0
        if self.reserved_ammo != expected_ammo:
            raise ValueError("reserved_ammo must match action mode")


@dataclass(frozen=True, slots=True)
class InternalAgentState:
    agent_id: int
    position: Position
    ammo: ResourceLedger
    distance: ResourceLedger
    initial_ammo_total: float
    initial_distance_total: float
    active_action: ActionCommit | None

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.agent_id, "agent_id")
        _validate_position(self.position)
        _require_nonnegative_finite(self.initial_ammo_total, "initial_ammo_total")
        _require_nonnegative_finite(self.initial_distance_total, "initial_distance_total")
        if not math.isclose(self.ammo.total, self.initial_ammo_total, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("ammo conservation violated")
        if not math.isclose(self.distance.total, self.initial_distance_total, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("distance conservation violated")
        if self.active_action is None:
            if self.ammo.reserved != 0 or self.distance.reserved != 0:
                raise ValueError("idle agent cannot retain reserved resources")
        else:
            if self.active_action.agent_id != self.agent_id:
                raise ValueError("active action agent does not own internal state")
            if self.ammo.reserved != self.active_action.reserved_ammo or self.distance.reserved != self.active_action.reserved_distance:
                raise ValueError("active action reservation must equal ledger reservation")


@dataclass(frozen=True, slots=True)
class CompletionEvent:
    finish_tick: int
    target_id: int
    agent_id: int
    action_ordinal: int

    def __post_init__(self) -> None:
        for name in ("finish_tick", "target_id", "agent_id", "action_ordinal"):
            _require_nonnegative_int(getattr(self, name), name)

    @property
    def heap_key(self) -> tuple[int, int, int]:
        return (self.finish_tick, self.target_id, self.agent_id)

    @classmethod
    def from_action(cls, action: ActionCommit) -> CompletionEvent:
        return cls(action.finish_tick, action.target_id, action.agent_id, action.ordinal)


@dataclass(frozen=True, slots=True)
class PrivateAuditEvent:
    event_id: int
    tick: int
    target_id: int
    agent_id: int
    mode: Mode
    draw: float
    true_category: str
    damage_before: str
    damage_after: str
    physical_success: bool
    realized_reward: float
    invalid_attack: bool
    initial_wreck_attack: bool
    counter_key: DrawKey

    def __post_init__(self) -> None:
        for name in ("event_id", "tick", "target_id", "agent_id"):
            _require_nonnegative_int(getattr(self, name), name)
        _validate_mode(self.mode)
        if not math.isfinite(self.draw) or not 0.0 <= self.draw < 1.0:
            raise ValueError("draw must lie in [0, 1)")
        if self.true_category not in _CATEGORIES or self.damage_before not in _DAMAGE_STATES or self.damage_after not in _DAMAGE_STATES:
            raise ValueError("private truth labels are invalid")
        if not math.isfinite(self.realized_reward):
            raise ValueError("realized_reward must be finite")
        if not isinstance(self.counter_key, DrawKey):
            raise TypeError("counter_key must be a DrawKey")


@dataclass(frozen=True, slots=True)
class GateFailure:
    gate: str
    tick: int
    reason: str
    details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.gate or not self.reason:
            raise ValueError("gate and reason must be nonempty")
        _require_nonnegative_int(self.tick, "tick")


@dataclass(frozen=True, slots=True)
class DynamicScenario:
    scenario_id: str
    cell_id: str
    seed: int
    crn_namespace: str
    targets: tuple[PublicTarget, ...]
    private_targets: tuple[PrivateTarget, ...]
    agents: tuple[InternalAgentState, ...]
    t_max_tick: int

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.cell_id or not self.crn_namespace:
            raise ValueError("scenario, cell, and CRN namespace must be nonempty")
        _require_nonnegative_int(self.seed, "seed")
        _require_nonnegative_int(self.t_max_tick, "t_max_tick")
        if self.t_max_tick == 0:
            raise ValueError("t_max_tick must be positive")
        _validate_contiguous_ids(self.targets, "target")
        _validate_contiguous_ids(self.private_targets, "private target")
        _validate_contiguous_ids(self.agents, "agent")
        if tuple(item.target_id for item in self.targets) != tuple(item.target_id for item in self.private_targets):
            raise ValueError("private target IDs must match public target IDs")


def validate_runtime_invariants(
    agents: tuple[InternalAgentState, ...],
    target_locks: tuple[tuple[int, int], ...],
    completion_events: tuple[CompletionEvent, ...],
) -> None:
    """Validate cross-object busy, lock, event, and one-active-action invariants."""

    _validate_contiguous_ids(agents, "agent")
    actions = tuple(agent.active_action for agent in agents if agent.active_action is not None)
    target_ids = tuple(action.target_id for action in actions)
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("one active action per target is required")
    lock_map = dict(target_locks)
    if len(lock_map) != len(target_locks) or lock_map != {action.target_id: action.agent_id for action in actions}:
        raise ValueError("target lock ownership must exactly match active actions")
    expected = sorted((CompletionEvent.from_action(action) for action in actions), key=lambda item: item.heap_key)
    actual = sorted(completion_events, key=lambda item: item.heap_key)
    if actual != expected or len({event.heap_key for event in actual}) != len(actual):
        raise ValueError("completion event set and event heap tie keys must uniquely match active actions")


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    scenario_id: str
    cell_id: str
    method: str
    initial_truth_digest: str
    final_truth_digest: str
    public_initial_digest: str
    termination: str
    status: str
    event_count: int
    action_count: int
    replan_count: int
    final_beliefs: tuple[Belief, ...]
    destroyed_value: float
    service_cost: float
    distance_cost: float
    ammo_cost: float
    realized_utility: float
    normalized_utility: float
    gross_scenario_value: float
    distance_consumed: float
    ammo_consumed: float
    makespan: float
    first_destroyed_value: float
    invalid_attack_count: int
    initial_wreck_attack_count: int
    recon_count: int
    bda_count: int
    continuous_attack_count: int
    handoff_count: int
    orphan_count: int
    stall_count: int
    cbba_round_count: int
    final_joint_brier_score: float
    allocator_gates: tuple[GateFailure, ...]
    replay_audit: str

    def __post_init__(self) -> None:
        for name in ("scenario_id", "cell_id", "method", "termination", "status", "replay_audit"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        for name in ("initial_truth_digest", "final_truth_digest", "public_initial_digest"):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in (
            "event_count", "action_count", "replan_count", "invalid_attack_count", "initial_wreck_attack_count",
            "recon_count", "bda_count", "continuous_attack_count", "handoff_count", "orphan_count", "stall_count",
            "cbba_round_count",
        ):
            _require_nonnegative_int(getattr(self, name), name)
        for index, belief in enumerate(self.final_beliefs):
            _validate_belief(belief, f"final_beliefs[{index}]")
        for name in (
            "destroyed_value", "service_cost", "distance_cost", "ammo_cost", "gross_scenario_value",
            "distance_consumed", "ammo_consumed", "makespan", "first_destroyed_value", "final_joint_brier_score",
        ):
            _require_nonnegative_finite(getattr(self, name), name)
        if not math.isfinite(self.realized_utility) or not math.isfinite(self.normalized_utility):
            raise ValueError("utility values must be finite")
        expected_utility = self.destroyed_value - self.service_cost - self.distance_cost - self.ammo_cost
        if not math.isclose(self.realized_utility, expected_utility, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("utility decomposition does not equal realized_utility")
        if self.gross_scenario_value <= 0:
            raise ValueError("gross_scenario_value must be positive")
        if not math.isclose(
            self.normalized_utility,
            self.realized_utility / self.gross_scenario_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("normalized utility is inconsistent")
        if self.first_destroyed_value != self.destroyed_value:
            raise ValueError("first_destroyed_value must equal the destruction reward component")
        if not 0.0 <= self.final_joint_brier_score <= 2.0:
            raise ValueError("final_joint_brier_score must lie in [0, 2]")
        if any(not isinstance(failure, GateFailure) for failure in self.allocator_gates):
            raise TypeError("allocator_gates accept only GateFailure")


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    record: EpisodeRecord
    public_events: tuple[PublicCompletion, ...]
    private_audit_events: tuple[PrivateAuditEvent, ...]
    gate_failures: tuple[GateFailure, ...]
    actions: tuple[ActionCommit, ...]

    def __post_init__(self) -> None:
        collections = (
            (self.public_events, PublicCompletion, "public_events"),
            (self.private_audit_events, PrivateAuditEvent, "private_audit_events"),
            (self.gate_failures, GateFailure, "gate_failures"),
            (self.actions, ActionCommit, "actions"),
        )
        for values, expected_type, name in collections:
            if any(not isinstance(value, expected_type) for value in values):
                raise TypeError(f"{name} contains the wrong event or record type")
        for values, attribute, name in (
            (self.public_events, "event_id", "public event"),
            (self.private_audit_events, "event_id", "private event"),
        ):
            ids = tuple(getattr(value, attribute) for value in values)
            if ids != tuple(range(len(values))):
                raise ValueError(f"{name} IDs must be contiguous and ordered")
        ordinal_groups: dict[tuple[int, Mode], list[int]] = {}
        for action in self.actions:
            ordinal_groups.setdefault((action.target_id, action.mode), []).append(action.ordinal)
        if any(ordinals != list(range(len(ordinals))) for ordinals in ordinal_groups.values()):
            raise ValueError("action ordinals must be contiguous within each target/mode key")


def _canonical_json(value: object) -> bytes:
    def jsonable(item: object) -> object:
        if is_dataclass(item) and not isinstance(item, type):
            return {key: jsonable(child) for key, child in asdict(item).items()}
        if isinstance(item, (tuple, list)):
            return [jsonable(child) for child in item]
        if isinstance(item, dict):
            return {str(key): jsonable(child) for key, child in item.items()}
        return item

    return json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def public_digest(snapshot: PublicSnapshot) -> str:
    """Return the stable digest of public state only."""

    if not isinstance(snapshot, PublicSnapshot):
        raise TypeError("public_digest accepts only PublicSnapshot")
    return hashlib.sha256(_canonical_json(snapshot)).hexdigest()


def private_truth_digest(targets: tuple[PrivateTarget, ...]) -> str:
    """Return the stable evaluator-only digest of canonical private truth."""

    if any(not isinstance(target, PrivateTarget) for target in targets):
        raise TypeError("private_truth_digest accepts only PrivateTarget values")
    _validate_contiguous_ids(targets, "private target")
    return hashlib.sha256(_canonical_json(targets)).hexdigest()
