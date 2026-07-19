"""Digest-pinned D0 witness contracts and deterministic D1 scenario generation.

D0 values in this module are specifications for the future simulator tests.  They
are deliberately expectations, not records claiming that a simulator has run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from .dynamic_rng import DrawKey, categorical, uniform01
from .attack import predict_attack
from .belief import bayes_update, bda_kernel, recon_kernel
from .dynamic_types import (
    DynamicConfig,
    DynamicScenario,
    InternalAgentState,
    PrivateTarget,
    PublicTarget,
    ResourceLedger,
    quantize_tick,
)


_RNG_VERSION = "sha256-u64-v1"
_EXPERIMENT_ID = "dynamic-lifecycle-mainline-v2"
_GENERATOR_VERSION = "d1-generator-v1"
_D2_GENERATOR_VERSION = "d2-generator-v1"
_APPROVED_CONFIG_ID = "recon_damage_plus_010_r2_a6_b3"
_PASSPORT_DIGEST = "314f923560a280221149613fffd7f51eb358150658e11bebf89640d9311cb57e"
_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "preregistration" / "first_batch.json"

_BELIEF_ARCHETYPES = (
    (0.00, 0.42, 0.24, 0.34),  # Recon
    (1.00, 0.00, 0.00, 0.00),  # Attack
    (0.26, 0.66, 0.00, 0.08),  # BDA
    (0.00, 0.08, 0.06, 0.86),  # Defer
)
_TRUTH_ORDER = (("H", "A"), ("H", "D"), ("L", "A"), ("L", "D"))


@dataclass(frozen=True, slots=True)
class D1Cell:
    """One bound D1 cell; resource dimensions are not independent factors."""

    agent_count: int
    target_count: int
    resource_tier: Literal["tight", "loose"]
    ammo_per_agent: float
    t_max: float
    range_per_agent: float

    @property
    def cell_id(self) -> str:
        return f"N{self.agent_count}-M{self.target_count}-R{self.resource_tier}"


def _cell(agent_count: int, target_count: int, resource_tier: Literal["tight", "loose"]) -> D1Cell:
    if resource_tier == "tight":
        return D1Cell(agent_count, target_count, resource_tier, 1.0, 16.0, 18.0)
    return D1Cell(agent_count, target_count, resource_tier, 3.0, 32.0, 40.0)


D1_CELLS = tuple(
    _cell(agent_count, target_count, resource_tier)
    for agent_count in (2, 3)
    for target_count in (3, 5)
    for resource_tier in ("tight", "loose")
)


class D0Action(str, Enum):
    RECON = "recon"
    ATTACK = "attack"
    BDA = "bda"


class D0Fault(str, Enum):
    FORCE_ATTACK_SUCCESS = "force_attack_success"
    FORCE_ATTACK_FAILURE = "force_attack_failure"
    REJECT_LOCKED = "reject_locked"
    EXCLUDE_BUSY = "exclude_busy"
    RELEASE_SUFFIX = "release_suffix"
    REJECT_HORIZON = "reject_horizon"
    REJECT_RANGE = "reject_range"
    REJECT_AMMO = "reject_ammo"
    FORCE_DEFER = "force_defer"
    ZERO_COMMIT = "zero_commit"
    AUTO_NEXT = "auto_next"
    REPLAY = "replay"
    SCHEDULER_PROBE = "scheduler_probe"


class PublicRecordKind(str, Enum):
    OBSERVATION = "observation"
    ATTACK_ACK = "attack_ack"
    SCHEDULER = "scheduler"
    TERMINATION = "termination"


class PrivateRecordKind(str, Enum):
    COMPLETION = "completion"
    ALLOCATION = "allocation"
    SCHEDULER = "scheduler"
    TERMINATION = "termination"
    REPLAY = "replay"
    REJECTION = "rejection"


class D0Terminal(str, Enum):
    NORMAL = "normal"
    NO_POSITIVE = "no_positive"
    HORIZON = "horizon"
    GATE_FAILURE = "gate_failure"


class D0GateCode(str, Enum):
    ALLOCATION_STALL = "allocation-stall"


class GateReason(str, Enum):
    POSITIVE_TASK_ZERO_COMMIT = "positive_task_zero_commit"


class RejectionReason(str, Enum):
    HORIZON = "horizon"
    RANGE = "range"
    AMMO = "ammo"
    TARGET_LOCKED = "target_locked"
    BUSY_AGENT = "busy_agent"
    SUFFIX_RELEASED = "suffix_released"
    DEFERRED = "deferred"
    ALLOCATION_STALL = "allocation_stall"


class AbsenceKind(str, Enum):
    NO_COMMIT = "no_commit"
    NO_COMPLETION = "no_completion"
    NO_COUNTER_READ = "no_counter_read"
    NO_EARLY_TERMINATION = "no_early_termination"


class AssertionId(str, Enum):
    INITIAL_WRECK_ZERO_REWARD = "initial_wreck_zero_reward"
    FIRST_DESTROYED_PAID_ONCE = "first_destroyed_paid_once"
    CONTINUOUS_ATTACK = "continuous_attack"
    ACK_NONLEAKAGE = "ack_nonleakage"
    RECON_JOINT_BAYES = "recon_joint_bayes"
    BDA_BEFORE_ATTACK = "bda_before_attack"
    TARGET_LOCK = "target_lock"
    BUSY_EXCLUSION = "busy_exclusion"
    COMPLETION_BATCH = "completion_batch"
    COMMIT_NEXT = "commit_next"
    HANDOFF = "handoff"
    HARD_REJECTION = "hard_rejection"
    COMPLETION_BEFORE_PLANNING = "completion_before_planning"
    DEFER_REACTIVATION = "defer_reactivation"
    NO_POSITIVE = "no_positive"
    COUNTER_REPLAY = "counter_replay"
    BUSY_PREVENTS_TERMINATION = "busy_prevents_termination"
    HORIZON_SETTLEMENT = "horizon_settlement"
    ALLOCATION_STALL = "allocation_stall"
    EVENT_NONLEAKAGE = "event_nonleakage"
    B1M_AUTO_NEXT = "b1m_auto_next"
    ABSOLUTE_DISCOUNT = "absolute_discount"


class OperandName(str, Enum):
    EXPECTED_COUNT = "expected_count"
    EXPECTED_REWARD = "expected_reward"
    EXPECTED_BOOL = "expected_bool"
    EXPECTED_TICK = "expected_tick"
    TOLERANCE = "tolerance"
    EXPECTED_POSTERIOR = "expected_posterior"
    EXPECTED_BIDDERS = "expected_bidders"
    EXPECTED_GATE_AGENTS = "expected_gate_agents"


@dataclass(frozen=True, slots=True)
class PublicPrecondition:
    tick: int
    idle_agents: tuple[int, ...]
    busy_agents: tuple[int, ...]
    locked_targets: tuple[int, ...]
    beliefs: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True, slots=True)
class PhysicalStateControl:
    true_category: Literal["H", "L"]
    damage_before: Literal["A", "D"]
    attack_success: bool | None
    observation: tuple[str, str] | str | None


@dataclass(frozen=True, slots=True)
class ScriptedPolicyStep:
    commit_tick: int
    precondition: PublicPrecondition
    agent_id: int | None
    target_id: int | None
    action: D0Action | None
    fault: D0Fault | None
    counter_key: DrawKey | None
    expected_uniform: Fraction | None
    physical_control: PhysicalStateControl | None
    committed: bool
    start_tick: int | None
    finish_tick: int | None

    @property
    def tick(self) -> int:
        return self.commit_tick


@dataclass(frozen=True, slots=True)
class ExpectedPublicRecord:
    kind: PublicRecordKind
    event_id: int
    tick: int
    target_id: int
    agent_id: int
    mode: D0Action | None
    observation: tuple[str, str] | str | None
    source_step_index: int | None = None
    queue_depth: int | None = None
    idle_positive_tasks: int | None = None
    commits: int | None = None
    posterior_belief: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class ExpectedPrivateRecord:
    kind: PrivateRecordKind
    event_id: int
    tick: int
    target_id: int
    agent_id: int
    mode: D0Action | None
    true_category: Literal["H", "L"]
    damage_before: Literal["A", "D"]
    damage_after: Literal["A", "D"]
    draw: Fraction | None
    physical_success: bool | None
    realized_reward: float
    source_step_index: int | None = None
    rejection_reason: RejectionReason | None = None
    first_destroyed_payment: bool = False


@dataclass(frozen=True, slots=True)
class ExpectedAbsence:
    kind: AbsenceKind
    step_index: int


@dataclass(frozen=True, slots=True)
class ExpectedGate:
    gate: D0GateCode
    tick: int
    reason: GateReason


@dataclass(frozen=True, slots=True)
class AssertionOperand:
    name: OperandName
    value: int | float | bool | tuple[int, ...] | tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class FixtureAssertion:
    assertion_id: AssertionId
    operands: tuple[AssertionOperand, ...]


@dataclass(frozen=True, slots=True)
class D0Run:
    run_id: str
    method_id: str
    scenario: DynamicScenario
    script: tuple[ScriptedPolicyStep, ...]
    expected_public_trace: tuple[ExpectedPublicRecord, ...]
    expected_private_audit_trace: tuple[ExpectedPrivateRecord, ...]
    expected_absences: tuple[ExpectedAbsence, ...]
    expected_terminal: D0Terminal
    terminal_tick: int
    expected_gate: ExpectedGate | None


@dataclass(frozen=True, slots=True)
class D0Fixture:
    """Non-executed, directly consumable input/expectation contract for a harness."""

    name: str
    scenario_id: str
    runs: tuple[D0Run, ...]
    focused_assertions: tuple[FixtureAssertion, ...]

    @property
    def scenario(self) -> DynamicScenario:
        return self.runs[0].scenario

    @property
    def script(self) -> tuple[ScriptedPolicyStep, ...]:
        return self.runs[0].script

    @property
    def expected_public_trace(self) -> tuple[ExpectedPublicRecord, ...]:
        return self.runs[0].expected_public_trace

    @property
    def expected_private_audit_trace(self) -> tuple[ExpectedPrivateRecord, ...]:
        return self.runs[0].expected_private_audit_trace

    @property
    def expected_terminal(self) -> D0Terminal:
        return self.runs[0].expected_terminal

    @property
    def expected_gate(self) -> ExpectedGate | None:
        return self.runs[0].expected_gate


def _d0_key(name: str, seed: int, target_id: int, action: D0Action, ordinal: int) -> DrawKey:
    return DrawKey(
        _RNG_VERSION, _EXPERIMENT_ID, _GENERATOR_VERSION, f"D0-{name}", seed,
        "target", target_id, action.value, ordinal, 0,
    )


def _scenario(
    name: str,
    *,
    t_max: float = 16.0,
    target_positions: tuple[tuple[float, float], ...] = ((2.0, 0.0), (0.0, 2.0)),
    target_truth: tuple[tuple[Literal["H", "L"], Literal["A", "D"]], ...] = (("H", "A"), ("L", "A")),
    target_beliefs: tuple[tuple[float, float, float, float], ...] | None = None,
    agent_positions: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.0, 0.0)),
    ammo: tuple[float, ...] = (3.0, 3.0),
    distance: tuple[float, ...] = (40.0, 40.0),
    seed: int = 0,
) -> DynamicScenario:
    config = DynamicConfig()
    beliefs = (
        tuple((0.4, 0.1, 0.2, 0.3) for _ in target_positions)
        if target_beliefs is None
        else target_beliefs
    )
    if len(beliefs) != len(target_positions):
        raise ValueError("target_beliefs must match target_positions")
    targets = tuple(
        PublicTarget(target_id, position, beliefs[target_id])
        for target_id, position in enumerate(target_positions)
    )
    private = tuple(
        PrivateTarget(target_id, truth[0], truth[1], False)
        for target_id, truth in enumerate(target_truth)
    )
    agents = tuple(
        InternalAgentState(
            agent_id, position, ResourceLedger(ammo[agent_id], 0.0, 0.0),
            ResourceLedger(distance[agent_id], 0.0, 0.0), ammo[agent_id], distance[agent_id], None,
        )
        for agent_id, position in enumerate(agent_positions)
    )
    scenario_id = f"D0-{name}"
    return DynamicScenario(
        scenario_id, scenario_id, seed, "d0-fixture-v2", targets, private, agents,
        quantize_tick(t_max, config.tick_size),
    )


@dataclass(frozen=True, slots=True)
class _StepSpec:
    action: D0Action | None
    agent_id: int | None
    target_id: int | None
    commit_tick: int
    physical: PhysicalStateControl | None = None
    committed: bool = True
    fault: D0Fault | None = None
    rejection: RejectionReason | None = None
    busy_agents: tuple[int, ...] = ()
    locked_targets: tuple[int, ...] = ()
    scheduler_state: tuple[int, int, int] | None = None


def _run(
    name: str,
    run_id: str,
    method_id: str,
    scenario: DynamicScenario,
    specs: tuple[_StepSpec, ...],
    terminal: D0Terminal = D0Terminal.NORMAL,
    gate: ExpectedGate | None = None,
    *,
    include_terminal_record: bool = False,
) -> D0Run:
    config = DynamicConfig()
    durations = {
        D0Action.RECON: config.recon_duration,
        D0Action.ATTACK: config.attack_duration,
        D0Action.BDA: config.bda_duration,
    }
    positions = {agent.agent_id: agent.position for agent in scenario.agents}
    ordinals: dict[tuple[int, D0Action], int] = {}
    script: list[ScriptedPolicyStep] = []
    public: list[ExpectedPublicRecord] = []
    private: list[ExpectedPrivateRecord] = []
    absences: list[ExpectedAbsence] = []
    beliefs = [target.belief for target in scenario.targets]
    pending_beliefs: list[tuple[int, int, tuple[float, float, float, float]]] = []
    truth_category = {target.target_id: target.true_category for target in scenario.private_targets}
    truth_damage = {target.target_id: target.true_damage for target in scenario.private_targets}
    paid_targets: set[int] = set()
    recon_z = recon_kernel(config.recon_category_matrix, config.recon_damage_matrix)
    bda_z = bda_kernel(config.bda_damage_matrix)
    recon_observations = (("H", "A"), ("H", "D"), ("L", "A"), ("L", "D"))
    event_id = 0
    for step_index, spec in enumerate(specs):
        ready = sorted((item for item in pending_beliefs if item[0] <= spec.commit_tick), key=lambda item: item[0])
        for item in ready:
            _, target_id, posterior = item
            beliefs[target_id] = posterior
            pending_beliefs.remove(item)
        idle = tuple(agent.agent_id for agent in scenario.agents if agent.agent_id not in spec.busy_agents)
        precondition = PublicPrecondition(
            spec.commit_tick, idle, spec.busy_agents, spec.locked_targets,
            tuple(beliefs),
        )
        if not spec.committed:
            step = ScriptedPolicyStep(
                spec.commit_tick, precondition, spec.agent_id, spec.target_id, spec.action,
                spec.fault, None, None, spec.physical, False, None, None,
            )
            script.append(step)
            if spec.rejection is not None:
                target_id = 0 if spec.target_id is None else spec.target_id
                agent_id = 0 if spec.agent_id is None else spec.agent_id
                truth = scenario.private_targets[target_id]
                private.append(ExpectedPrivateRecord(
                    PrivateRecordKind.REJECTION, event_id, spec.commit_tick, target_id, agent_id,
                    spec.action, truth.true_category, truth.true_damage, truth.true_damage,
                    None, None, 0.0, step_index, spec.rejection,
                ))
                event_id += 1
            if spec.scheduler_state is not None:
                queue_depth, positive, commits = spec.scheduler_state
                public.append(ExpectedPublicRecord(
                    PublicRecordKind.SCHEDULER, event_id, spec.commit_tick, 0, 0, None, None,
                    step_index, queue_depth, positive, commits,
                ))
                event_id += 1
            absences.extend(ExpectedAbsence(kind, step_index) for kind in (
                AbsenceKind.NO_COMMIT, AbsenceKind.NO_COMPLETION, AbsenceKind.NO_COUNTER_READ,
            ))
            continue

        if spec.action is None or spec.agent_id is None or spec.target_id is None or spec.physical is None:
            raise ValueError("committed D0 step requires action, agent, target, and physical control")
        origin = positions[spec.agent_id]
        destination = scenario.targets[spec.target_id].position
        travel = math.hypot(destination[0] - origin[0], destination[1] - origin[1]) / config.speed
        start_tick = spec.commit_tick + quantize_tick(travel, config.tick_size)
        finish_tick = start_tick + quantize_tick(durations[spec.action], config.tick_size)
        ordinal_key = (spec.target_id, spec.action)
        ordinal = ordinals.get(ordinal_key, 0)
        ordinals[ordinal_key] = ordinal + 1
        key = _d0_key(name, scenario.seed, spec.target_id, spec.action, ordinal)
        draw = uniform01(key)
        if spec.physical.true_category != truth_category[spec.target_id] or spec.physical.damage_before != truth_damage[spec.target_id]:
            raise ValueError("D0 physical control must match the canonical current private state")
        observation: tuple[str, str] | str | None = None
        physical_success: bool | None = None
        after: Literal["A", "D"] = spec.physical.damage_before
        reward = 0.0
        first_destroyed_payment = False
        if spec.action is D0Action.ATTACK:
            probability = config.attack_success_high if spec.physical.true_category == "H" else config.attack_success_low
            physical_success = after == "A" and draw < Fraction.from_float(probability)
            if physical_success:
                after = "D"
                reward = config.value_high if spec.physical.true_category == "H" else config.value_low
                first_destroyed_payment = spec.target_id not in paid_targets
                paid_targets.add(spec.target_id)
            posterior_array = predict_attack(beliefs[spec.target_id], config.attack_success_high, config.attack_success_low)
        elif spec.action is D0Action.RECON:
            class_column = 0 if spec.physical.true_category == "H" else 1
            damage_column = 0 if spec.physical.damage_before == "A" else 1
            probabilities = tuple(
                config.recon_category_matrix[row // 2][class_column]
                * config.recon_damage_matrix[row % 2][damage_column]
                for row in range(4)
            )
            observation_index = categorical(key, probabilities)
            observation = recon_observations[observation_index]
            posterior_array = bayes_update(beliefs[spec.target_id], recon_z, observation_index)
        else:
            damage_column = 0 if spec.physical.damage_before == "A" else 1
            probabilities = tuple(config.bda_damage_matrix[row][damage_column] for row in range(2))
            observation_index = categorical(key, probabilities)
            observation = ("A", "D")[observation_index]
            posterior_array = bayes_update(beliefs[spec.target_id], bda_z, observation_index)
        posterior = tuple(float(value) for value in posterior_array)
        physical = PhysicalStateControl(
            spec.physical.true_category, spec.physical.damage_before, physical_success, observation,
        )
        step = ScriptedPolicyStep(
            spec.commit_tick, precondition, spec.agent_id, spec.target_id, spec.action,
            None, key, draw, physical, True, start_tick, finish_tick,
        )
        script.append(step)
        public_kind = PublicRecordKind.ATTACK_ACK if spec.action is D0Action.ATTACK else PublicRecordKind.OBSERVATION
        public.append(ExpectedPublicRecord(
            public_kind, event_id, finish_tick, spec.target_id, spec.agent_id,
            spec.action, observation, step_index, posterior_belief=posterior,
        ))
        private.append(ExpectedPrivateRecord(
            PrivateRecordKind.COMPLETION, event_id, finish_tick, spec.target_id, spec.agent_id,
            spec.action, spec.physical.true_category, spec.physical.damage_before, after,
            draw, physical_success, reward, step_index, None, first_destroyed_payment,
        ))
        event_id += 1
        positions[spec.agent_id] = destination
        truth_damage[spec.target_id] = after
        pending_beliefs.append((finish_tick, spec.target_id, posterior))

    completion_ticks = tuple(step.finish_tick for step in script if step.finish_tick is not None)
    terminal_tick = max(completion_ticks, default=0)
    if terminal is D0Terminal.HORIZON:
        terminal_tick = scenario.t_max_tick
    if include_terminal_record:
        public.append(ExpectedPublicRecord(
            PublicRecordKind.TERMINATION, event_id, terminal_tick, 0, 0, None, None, None, 0, 0, 0,
        ))
        truth = scenario.private_targets[0]
        private.append(ExpectedPrivateRecord(
            PrivateRecordKind.TERMINATION, event_id, terminal_tick, 0, 0, None,
            truth.true_category, truth.true_damage, truth.true_damage, None, None, 0.0,
        ))
    public_rank = {
        PublicRecordKind.OBSERVATION: 0,
        PublicRecordKind.ATTACK_ACK: 0,
        PublicRecordKind.SCHEDULER: 1,
        PublicRecordKind.TERMINATION: 2,
    }
    public = [
        replace(record, event_id=index)
        for index, record in enumerate(sorted(public, key=lambda record: (record.tick, public_rank[record.kind])))
    ]
    private_rank = {
        PrivateRecordKind.COMPLETION: 0,
        PrivateRecordKind.REJECTION: 1,
        PrivateRecordKind.ALLOCATION: 1,
        PrivateRecordKind.SCHEDULER: 2,
        PrivateRecordKind.REPLAY: 2,
        PrivateRecordKind.TERMINATION: 3,
    }
    private = [
        replace(record, event_id=index)
        for index, record in enumerate(sorted(private, key=lambda record: (record.tick, private_rank[record.kind])))
    ]
    return D0Run(
        run_id, method_id, scenario, tuple(script), tuple(public), tuple(private),
        tuple(absences), terminal, terminal_tick, gate,
    )


def _assertion(
    assertion_id: AssertionId,
    count: int,
    reward: float,
    extras: tuple[AssertionOperand, ...] = (),
) -> tuple[FixtureAssertion, ...]:
    return (FixtureAssertion(assertion_id, (
        AssertionOperand(OperandName.EXPECTED_COUNT, count),
        AssertionOperand(OperandName.EXPECTED_REWARD, reward),
        AssertionOperand(OperandName.TOLERANCE, 1e-10),
    ) + extras),)


def _fixture(name: str, runs: tuple[D0Run, ...], assertion_id: AssertionId, count: int = 1, reward: float = 0.0) -> D0Fixture:
    derived_reward = sum(
        event.realized_reward
        for event in runs[0].expected_private_audit_trace
        if event.kind is PrivateRecordKind.COMPLETION
    )
    extras: tuple[AssertionOperand, ...] = ()
    if assertion_id is AssertionId.RECON_JOINT_BAYES:
        posterior = next(event.posterior_belief for event in runs[0].expected_public_trace if event.kind is PublicRecordKind.OBSERVATION)
        if posterior is None:
            raise ValueError("Recon fixture requires a concrete posterior")
        extras = (AssertionOperand(OperandName.EXPECTED_POSTERIOR, posterior),)
    elif assertion_id is AssertionId.BUSY_EXCLUSION:
        extras = (
            AssertionOperand(OperandName.EXPECTED_BIDDERS, (1,)),
            AssertionOperand(OperandName.EXPECTED_GATE_AGENTS, (1,)),
        )
    return D0Fixture(name, f"D0-{name}", runs, _assertion(assertion_id, count, derived_reward, extras))


def _attack(category: Literal["H", "L"] = "H", damage: Literal["A", "D"] = "A") -> PhysicalStateControl:
    return PhysicalStateControl(category, damage, None, None)


def _sense(category: Literal["H", "L"] = "H", damage: Literal["A", "D"] = "A") -> PhysicalStateControl:
    return PhysicalStateControl(category, damage, None, None)


def _find_attack_seed(name: str, category: Literal["H", "L"], outcomes: tuple[bool, ...]) -> int:
    config = DynamicConfig()
    probability = config.attack_success_high if category == "H" else config.attack_success_low
    threshold = Fraction.from_float(probability)
    for seed in range(100_000):
        actual = tuple(uniform01(_d0_key(name, seed, 0, D0Action.ATTACK, ordinal)) < threshold for ordinal in range(len(outcomes)))
        if actual == outcomes:
            return seed
    raise RuntimeError(f"no D0 counter context found for {name}: {outcomes}")


def _build_initial_wreck_zero_reward() -> D0Fixture:
    name = "initial_wreck_zero_reward"
    scenario = _scenario(name, target_truth=(("H", "D"), ("L", "A")))
    return _fixture(name, (_run(name, "main", "P", scenario, (_StepSpec(D0Action.ATTACK, 0, 0, 0, _attack(damage="D")),)),), AssertionId.INITIAL_WRECK_ZERO_REWARD)


def _build_first_new_destroyed_paid_once() -> D0Fixture:
    name = "first_new_destroyed_paid_once"
    scenario = _scenario(name, seed=_find_attack_seed(name, "H", (True,)))
    first = _StepSpec(D0Action.ATTACK, 0, 0, 0, _attack())
    first_finish = quantize_tick(4.0, DynamicConfig().tick_size)
    second = _StepSpec(D0Action.ATTACK, 0, 0, first_finish, _attack(damage="D"))
    return _fixture(name, (_run(name, "main", "P", scenario, (first, second)),), AssertionId.FIRST_DESTROYED_PAID_ONCE, 2, 100.0)


def _build_continuous_attack_after_failure() -> D0Fixture:
    name = "continuous_attack_after_failure"
    scenario = _scenario(name, seed=_find_attack_seed(name, "H", (False, True)))
    first_finish = quantize_tick(4.0, DynamicConfig().tick_size)
    specs = (_StepSpec(D0Action.ATTACK, 0, 0, 0, _attack()), _StepSpec(D0Action.ATTACK, 0, 0, first_finish, _attack()))
    return _fixture(name, (_run(name, "main", "P", scenario, specs),), AssertionId.CONTINUOUS_ATTACK, 2, 100.0)


def _build_attack_ack_hides_outcome() -> D0Fixture:
    name = "attack_ack_hides_outcome"
    success_scenario = _scenario(name, seed=_find_attack_seed(name, "H", (True,)))
    failure_seed = next(seed for seed in range(100_000) if seed != success_scenario.seed and not (uniform01(_d0_key(name, seed, 0, D0Action.ATTACK, 0)) < Fraction.from_float(DynamicConfig().attack_success_high)))
    failure_scenario = _scenario(name, seed=failure_seed)
    success = _run(name, "success_world", "P", success_scenario, (_StepSpec(D0Action.ATTACK, 0, 0, 0, _attack()),))
    failure = _run(name, "failure_world", "P", failure_scenario, (_StepSpec(D0Action.ATTACK, 0, 0, 0, _attack()),))
    return _fixture(name, (success, failure), AssertionId.ACK_NONLEAKAGE)


def _build_recon_joint_bayes() -> D0Fixture:
    name = "recon_joint_bayes"
    scenario = _scenario(name)
    spec = _StepSpec(D0Action.RECON, 0, 0, 0, _sense())
    return _fixture(name, (_run(name, "main", "P", scenario, (spec,)),), AssertionId.RECON_JOINT_BAYES)


def _build_bda_before_first_attack() -> D0Fixture:
    name = "bda_before_first_attack"
    scenario = _scenario(name)
    spec = _StepSpec(D0Action.BDA, 0, 0, 0, _sense())
    return _fixture(name, (_run(name, "main", "P", scenario, (spec,)),), AssertionId.BDA_BEFORE_ATTACK)


def _build_target_lock_prevents_overlap() -> D0Fixture:
    name = "target_lock_prevents_overlap"
    scenario = _scenario(name)
    specs = (
        _StepSpec(D0Action.ATTACK, 0, 0, 0, _attack()),
        _StepSpec(D0Action.ATTACK, 1, 0, 1, None, False, D0Fault.REJECT_LOCKED, RejectionReason.TARGET_LOCKED, (0,), (0,)),
    )
    return _fixture(name, (_run(name, "main", "P", scenario, specs),), AssertionId.TARGET_LOCK)


def _build_busy_uav_excluded_from_bid_and_gate() -> D0Fixture:
    name = "busy_uav_excluded_from_bid_and_gate"
    scenario = _scenario(name)
    specs = (
        _StepSpec(D0Action.ATTACK, 0, 0, 0, _attack()),
        _StepSpec(None, 0, None, 1, None, False, D0Fault.EXCLUDE_BUSY, RejectionReason.BUSY_AGENT, (0,), (0,), (1, 1, 1)),
        _StepSpec(D0Action.ATTACK, 1, 1, 1, _attack("L"), True, None, None, (0,), (0,)),
    )
    return _fixture(name, (_run(name, "main", "P", scenario, specs),), AssertionId.BUSY_EXCLUSION)


def _build_simultaneous_completion_batch() -> D0Fixture:
    name = "simultaneous_completion_batch"
    scenario = _scenario(name)
    specs = (_StepSpec(D0Action.ATTACK, 0, 0, 0, _attack()), _StepSpec(D0Action.ATTACK, 1, 1, 0, _attack("L")))
    return _fixture(name, (_run(name, "main", "P", scenario, specs),), AssertionId.COMPLETION_BATCH, 2, 130.0)


def _build_commit_next_releases_suffix() -> D0Fixture:
    name = "commit_next_releases_suffix"
    scenario = _scenario(name)
    specs = (
        _StepSpec(D0Action.ATTACK, 0, 0, 0, _attack()),
        _StepSpec(D0Action.ATTACK, 0, 1, 0, None, False, D0Fault.RELEASE_SUFFIX),
    )
    return _fixture(name, (_run(name, "main", "P", scenario, specs),), AssertionId.COMMIT_NEXT)


def _build_cross_uav_sequential_handoff() -> D0Fixture:
    name = "cross_uav_sequential_handoff"
    scenario = _scenario(name)
    recon_finish = quantize_tick(6.0, DynamicConfig().tick_size)
    specs = (
        _StepSpec(D0Action.RECON, 0, 0, 0, _sense()),
        _StepSpec(D0Action.ATTACK, 1, 0, recon_finish, _attack()),
    )
    return _fixture(name, (_run(name, "main", "P", scenario, specs),), AssertionId.HANDOFF, 2, 100.0)


def _build_horizon_range_ammo_rejection() -> D0Fixture:
    name = "horizon_range_ammo_rejection"
    scenario = _scenario(
        name, target_positions=((2.0, 0.0), (10.0, 0.0), (0.0, 2.0)),
        target_truth=(("H", "A"), ("L", "A"), ("H", "A")),
        agent_positions=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)), ammo=(3.0, 3.0, 0.0), distance=(40.0, 1.0, 40.0),
    )
    specs = (
        _StepSpec(D0Action.ATTACK, 0, 0, quantize_tick(15.0, DynamicConfig().tick_size), None, False, D0Fault.REJECT_HORIZON, RejectionReason.HORIZON),
        _StepSpec(D0Action.ATTACK, 1, 1, 0, None, False, D0Fault.REJECT_RANGE, RejectionReason.RANGE),
        _StepSpec(D0Action.ATTACK, 2, 2, 0, None, False, D0Fault.REJECT_AMMO, RejectionReason.AMMO),
    )
    return _fixture(name, (_run(name, "main", "P", scenario, specs, D0Terminal.NO_POSITIVE),), AssertionId.HARD_REJECTION, 3)


def _build_completion_precedes_periodic_planning() -> D0Fixture:
    name = "completion_precedes_periodic_planning"
    scenario = _scenario(name)
    run = _run(name, "main", "P", scenario, (_StepSpec(D0Action.ATTACK, 0, 0, 0, _attack()),))
    finish = run.script[0].finish_tick or 0
    scheduler = ExpectedPublicRecord(PublicRecordKind.SCHEDULER, 1, finish, 0, 0, None, None, None, 0, 0, 0)
    run = replace(run, expected_public_trace=run.expected_public_trace + (scheduler,))
    return _fixture(name, (run,), AssertionId.COMPLETION_BEFORE_PLANNING)


def _build_defer_reactivated_by_event() -> D0Fixture:
    name = "defer_reactivated_by_event"
    scenario = _scenario(
        name,
        target_positions=((8.0, 0.0), (6.0, 0.0)),
        target_truth=(("L", "A"), ("H", "D")),
        target_beliefs=((0.0, 0.0, 0.25, 0.75), (0.26, 0.66, 0.0, 0.08)),
        agent_positions=((0.0, 0.0),),
        ammo=(3.0,),
        distance=(40.0,),
    )
    bda_finish = quantize_tick(7.5, DynamicConfig().tick_size)
    recon_finish = quantize_tick(13.5, DynamicConfig().tick_size)
    specs = (
        _StepSpec(None, None, 0, 0, None, False, D0Fault.FORCE_DEFER),
        _StepSpec(D0Action.BDA, 0, 1, 0, _sense("H", "D")),
        _StepSpec(D0Action.RECON, 0, 0, bda_finish, _sense("L", "A")),
        _StepSpec(D0Action.ATTACK, 0, 0, recon_finish, _attack("L")),
    )
    return _fixture(
        name, (_run(name, "main", "P", scenario, specs),),
        AssertionId.DEFER_REACTIVATION, 3, 30.0,
    )


def _build_no_positive_normal_termination() -> D0Fixture:
    name = "no_positive_normal_termination"
    scenario = _scenario(name)
    spec = _StepSpec(
        None, None, 0, 0, None, False, D0Fault.FORCE_DEFER,
        scheduler_state=(0, 0, 0),
    )
    return _fixture(name, (_run(name, "main", "P", scenario, (spec,), D0Terminal.NO_POSITIVE, include_terminal_record=True),), AssertionId.NO_POSITIVE)


def _build_counter_replay_shared_initial_truth() -> D0Fixture:
    name = "counter_replay_shared_initial_truth"
    scenario = _scenario(name)
    specs = (_StepSpec(D0Action.RECON, 0, 0, 0, _sense()),)
    return _fixture(name, (_run(name, "P", "P", scenario, specs), _run(name, "B1m", "B1m", scenario, specs)), AssertionId.COUNTER_REPLAY)


def _build_busy_event_prevents_early_termination() -> D0Fixture:
    name = "busy_event_prevents_early_termination"
    scenario = _scenario(name)
    specs = (
        _StepSpec(D0Action.ATTACK, 0, 0, 0, _attack()),
        _StepSpec(None, None, None, 1, None, False, D0Fault.SCHEDULER_PROBE, None, (0,), (0,), (1, 0, 0)),
    )
    run = _run(name, "main", "P", scenario, specs, include_terminal_record=True)
    run = replace(run, expected_absences=run.expected_absences + (ExpectedAbsence(AbsenceKind.NO_EARLY_TERMINATION, 1),))
    return _fixture(name, (run,), AssertionId.BUSY_PREVENTS_TERMINATION)


def _build_completion_at_horizon_settles_first() -> D0Fixture:
    name = "completion_at_horizon_settles_first"
    scenario = _scenario(name, t_max=4.0)
    run = _run(name, "main", "P", scenario, (_StepSpec(D0Action.ATTACK, 0, 0, 0, _attack()),), D0Terminal.HORIZON)
    return _fixture(name, (run,), AssertionId.HORIZON_SETTLEMENT, 1, 100.0)


def _build_positive_task_zero_commit_stall_gate() -> D0Fixture:
    name = "positive_task_zero_commit_stall_gate"
    scenario = _scenario(name)
    gate = ExpectedGate(D0GateCode.ALLOCATION_STALL, 0, GateReason.POSITIVE_TASK_ZERO_COMMIT)
    spec = _StepSpec(
        None, None, 0, 0, None, False, D0Fault.ZERO_COMMIT,
        scheduler_state=(0, 1, 0),
    )
    return _fixture(name, (_run(name, "main", "P", scenario, (spec,), D0Terminal.GATE_FAILURE, gate),), AssertionId.ALLOCATION_STALL)


def _build_public_private_event_nonleakage() -> D0Fixture:
    name = "public_private_event_nonleakage"
    scenario = _scenario(name)
    recon_finish = quantize_tick(6.0, DynamicConfig().tick_size)
    specs = (
        _StepSpec(D0Action.RECON, 0, 0, 0, _sense()),
        _StepSpec(D0Action.ATTACK, 0, 0, recon_finish, _attack()),
    )
    return _fixture(name, (_run(name, "main", "P", scenario, specs),), AssertionId.EVENT_NONLEAKAGE, 2, 100.0)


def _build_b1m_frozen_suffix_auto_next() -> D0Fixture:
    name = "b1m_frozen_suffix_auto_next"
    scenario = _scenario(
        name,
        agent_positions=((0.0, 0.0),),
        ammo=(3.0,),
        distance=(40.0,),
    )
    first_finish = quantize_tick(4.0, DynamicConfig().tick_size)
    specs = (
        _StepSpec(D0Action.ATTACK, 0, 1, 0, _attack("L")),
        _StepSpec(D0Action.ATTACK, 0, 0, first_finish, _attack()),
    )
    run = _run(
        name, "main", "B1m", scenario, specs,
        D0Terminal.NORMAL, include_terminal_record=True,
    )
    return _fixture(name, (run,), AssertionId.B1M_AUTO_NEXT, 2, 130.0)


def _build_dynamic_absolute_discount_clock() -> D0Fixture:
    name = "dynamic_absolute_discount_clock"
    scenario = _scenario(name)
    later_commit = quantize_tick(6.0, DynamicConfig().tick_size)
    specs = (_StepSpec(D0Action.ATTACK, 0, 0, 0, _attack()), _StepSpec(D0Action.ATTACK, 1, 1, later_commit, _attack("L")))
    return _fixture(name, (_run(name, "main", "P", scenario, specs),), AssertionId.ABSOLUTE_DISCOUNT, 2, 130.0)


_D0_BUILDERS = (
    _build_initial_wreck_zero_reward,
    _build_first_new_destroyed_paid_once,
    _build_continuous_attack_after_failure,
    _build_attack_ack_hides_outcome,
    _build_recon_joint_bayes,
    _build_bda_before_first_attack,
    _build_target_lock_prevents_overlap,
    _build_busy_uav_excluded_from_bid_and_gate,
    _build_simultaneous_completion_batch,
    _build_commit_next_releases_suffix,
    _build_cross_uav_sequential_handoff,
    _build_horizon_range_ammo_rejection,
    _build_completion_precedes_periodic_planning,
    _build_defer_reactivated_by_event,
    _build_no_positive_normal_termination,
    _build_counter_replay_shared_initial_truth,
    _build_busy_event_prevents_early_termination,
    _build_completion_at_horizon_settles_first,
    _build_positive_task_zero_commit_stall_gate,
    _build_public_private_event_nonleakage,
    _build_b1m_frozen_suffix_auto_next,
    _build_dynamic_absolute_discount_clock,
)


_D0_FIXTURES = tuple(builder() for builder in _D0_BUILDERS)


def dynamic_registry_digest() -> str:
    """Hash the Material Passport's registry bytes and reject local drift."""

    digest = sha256(_REGISTRY_PATH.read_bytes()).hexdigest()
    if digest != _PASSPORT_DIGEST:
        raise RuntimeError(f"probability registry digest mismatch: expected {_PASSPORT_DIGEST}, got {digest}")
    return digest


def dynamic_config_registry() -> Mapping[str, DynamicConfig]:
    """Return the immutable registry containing only the approved matched model."""

    dynamic_registry_digest()
    config = DynamicConfig()
    return MappingProxyType({config.config_id: config})


def d0_scenarios() -> tuple[D0Fixture, ...]:
    """Return all 22 immutable, non-executed D0 witness contracts."""

    return _D0_FIXTURES


def _draw_key(
    cell: D1Cell,
    scenario_index: int,
    namespace: Literal["target", "agent"],
    entity_id: int,
    event_type: str,
    subdraw_index: int,
    generator_version: str = _GENERATOR_VERSION,
) -> DrawKey:
    return DrawKey(
        rng_version=_RNG_VERSION,
        experiment_id=_EXPERIMENT_ID,
        generator_version=generator_version,
        cell_id=cell.cell_id,
        within_cell_seed=scenario_index,
        entity_namespace=namespace,
        entity_id=entity_id,
        event_type=event_type,
        occurrence_index=0,
        subdraw_index=subdraw_index,
    )


def _coordinate(key: DrawKey) -> float:
    # Form the affine map exactly, then perform the sole binary64 conversion.
    return float(Fraction(-6) + 12 * uniform01(key))


def _position(
    cell: D1Cell,
    scenario_index: int,
    namespace: Literal["target", "agent"],
    entity_id: int,
    generator_version: str = _GENERATOR_VERSION,
) -> tuple[float, float]:
    return tuple(
        _coordinate(_draw_key(
            cell, scenario_index, namespace, entity_id, "initial_position", axis,
            generator_version,
        ))
        for axis in (0, 1)
    )  # type: ignore[return-value]


def _generate_scenario(
    cell: D1Cell,
    scenario_index: int,
    registered_config_id: str,
    *,
    stage: str,
    generator_version: str,
) -> DynamicScenario:
    registry = dynamic_config_registry()
    if registered_config_id not in registry:
        raise KeyError(f"unregistered dynamic config ID: {registered_config_id}")

    targets: list[PublicTarget] = []
    private_targets: list[PrivateTarget] = []
    for target_id in range(cell.target_count):
        belief = _BELIEF_ARCHETYPES[(target_id + scenario_index) % len(_BELIEF_ARCHETYPES)]
        targets.append(PublicTarget(
            target_id,
            _position(cell, scenario_index, "target", target_id, generator_version),
            belief,
        ))
        truth_key = _draw_key(
            cell, scenario_index, "target", target_id, "initial_truth", 0,
            generator_version,
        )
        category, damage = _TRUTH_ORDER[categorical(truth_key, belief)]
        private_targets.append(PrivateTarget(target_id, category, damage, False))

    agents = tuple(
        InternalAgentState(
            agent_id=agent_id,
            position=_position(cell, scenario_index, "agent", agent_id, generator_version),
            ammo=ResourceLedger(cell.ammo_per_agent, 0.0, 0.0),
            distance=ResourceLedger(cell.range_per_agent, 0.0, 0.0),
            initial_ammo_total=cell.ammo_per_agent,
            initial_distance_total=cell.range_per_agent,
            active_action=None,
        )
        for agent_id in range(cell.agent_count)
    )
    return DynamicScenario(
        scenario_id=f"{stage}-{cell.cell_id}-S{scenario_index:0{2 if stage == 'D1' else 4}d}",
        cell_id=cell.cell_id,
        seed=scenario_index,
        crn_namespace=f"{_EXPERIMENT_ID}/{generator_version}",
        targets=tuple(targets),
        private_targets=tuple(private_targets),
        agents=agents,
        t_max_tick=quantize_tick(cell.t_max, registry[registered_config_id].tick_size),
    )


def generate_d1_scenario(
    cell: D1Cell,
    scenario_index: int,
    registered_config_id: str,
) -> DynamicScenario:
    """Generate one of the frozen 20 D1 scenarios in a registered cell."""

    if cell not in D1_CELLS:
        raise ValueError("cell is not a registered D1 cell")
    if isinstance(scenario_index, bool) or not isinstance(scenario_index, int) or not 0 <= scenario_index < 20:
        raise ValueError("scenario_index must be an integer in [0, 20)")
    return _generate_scenario(
        cell, scenario_index, registered_config_id,
        stage="D1", generator_version=_GENERATOR_VERSION,
    )


def generate_d2_scenario(
    cell: D1Cell,
    scenario_index: int,
    registered_config_id: str,
) -> DynamicScenario:
    """Generate one frozen D2 confirmation scenario."""

    if cell not in D1_CELLS:
        raise ValueError("cell is not a registered D2 cell")
    if (
        isinstance(scenario_index, bool)
        or not isinstance(scenario_index, int)
        or not 1000 <= scenario_index < 1512
    ):
        raise ValueError("scenario_index must be an integer in [1000, 1512)")
    return _generate_scenario(
        cell, scenario_index, registered_config_id,
        stage="D2", generator_version=_D2_GENERATOR_VERSION,
    )


def _canonical_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if is_dataclass(value):
        return _canonical_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    return value


def _canonical_digest(value: object) -> str:
    payload = json.dumps(_canonical_value(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def d0_contract_digest(fixtures: tuple[D0Fixture, ...] | None = None) -> str:
    """Digest all canonical D0 input and expectation fields in fixture-name order."""

    selected = _D0_FIXTURES if fixtures is None else fixtures
    return _canonical_digest(tuple(sorted(selected, key=lambda fixture: fixture.name)))


def d1_draw_digest(scenarios: tuple[DynamicScenario, ...]) -> str:
    """Digest generated D1 IDs, geometry, beliefs, truth, resources, and horizon."""

    return _canonical_digest(tuple(sorted(scenarios, key=lambda scenario: scenario.scenario_id)))
