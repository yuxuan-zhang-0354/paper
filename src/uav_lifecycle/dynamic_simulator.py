"""Deterministic integer-tick simulator for the dynamic lifecycle experiment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import math
from typing import Iterable

from .attack import predict_attack
from .belief import bayes_update, bda_kernel, recon_kernel
from .dynamic_planning import PlannedPath, commit_ticks, validate_commit_candidates
from .dynamic_rng import DrawKey, categorical, uniform01
from .dynamic_types import (
    ActionCommit,
    Belief,
    CompletionEvent,
    DynamicConfig,
    DynamicScenario,
    EnvironmentModel,
    EpisodeRecord,
    EpisodeResult,
    GateFailure,
    InternalAgentState,
    Mode,
    PlanningClock,
    PrivateAuditEvent,
    PrivateTarget,
    PublicActionAck,
    PublicAgent,
    PublicBusyAction,
    PublicCompletion,
    PublicObservation,
    PublicSnapshot,
    PublicTarget,
    private_truth_digest,
    public_digest,
    quantize_tick,
    validate_runtime_invariants,
)


_RNG_VERSION = "sha256-u64-v1"
_EXPERIMENT_ID = "dynamic-lifecycle-mainline-v2"
_D1_GENERATOR_VERSION = "d1-generator-v1"
_MODES = frozenset({"recon", "attack", "bda"})
_OBSERVATIONS = (("H", "A"), ("H", "D"), ("L", "A"), ("L", "D"))


@dataclass(frozen=True, slots=True)
class SimulatorState:
    """Complete simulator-owned state; only ``snapshot`` may cross to a policy."""

    scenario_id: str
    cell_id: str
    seed: int
    crn_namespace: str
    tick: int
    targets: tuple[PublicTarget, ...]
    private_targets: tuple[PrivateTarget, ...]
    initial_private_targets: tuple[PrivateTarget, ...]
    agents: tuple[InternalAgentState, ...]
    target_locks: tuple[tuple[int, int], ...]
    completion_events: tuple[CompletionEvent, ...]
    ordinals: tuple[tuple[int, Mode, int], ...]
    public_events: tuple[PublicCompletion, ...] = ()
    private_audit_events: tuple[PrivateAuditEvent, ...] = ()
    actions: tuple[ActionCommit, ...] = ()
    gates: tuple[GateFailure, ...] = ()

    def snapshot(self) -> PublicSnapshot:
        public_agents = tuple(
            PublicAgent(
                agent.agent_id,
                agent.position,
                agent.ammo.available,
                agent.distance.available,
                None
                if agent.active_action is None
                else PublicBusyAction(
                    agent.active_action.target_id,
                    agent.active_action.destination,
                    agent.active_action.finish_tick,
                ),
            )
            for agent in self.agents
        )
        return PublicSnapshot(
            self.tick,
            self.targets,
            public_agents,
            self.target_locks,
            tuple(event for event in self.public_events if isinstance(event, PublicObservation)),
            tuple(event for event in self.public_events if isinstance(event, PublicActionAck)),
        )


@dataclass(frozen=True, slots=True)
class CommitBatchResult:
    state: SimulatorState
    committed: tuple[ActionCommit, ...]
    gates: tuple[GateFailure, ...]


@dataclass(frozen=True, slots=True)
class CompletionBatchResult:
    state: SimulatorState
    public_events: tuple[PublicCompletion, ...]
    private_events: tuple[PrivateAuditEvent, ...]
    gates: tuple[GateFailure, ...]


@dataclass(frozen=True, slots=True)
class UtilityEvaluation:
    destroyed_value: float
    service_cost: float
    distance_cost: float
    ammo_cost: float
    realized_utility: float
    normalized_utility: float
    gross_scenario_value: float
    distance_consumed: float
    ammo_consumed: float


def initialize_state(scenario: DynamicScenario) -> SimulatorState:
    """Create fresh simulator-owned state from one immutable scenario."""

    if not isinstance(scenario, DynamicScenario):
        raise TypeError("scenario must be DynamicScenario")
    state = SimulatorState(
        scenario.scenario_id,
        scenario.cell_id,
        scenario.seed,
        scenario.crn_namespace,
        0,
        scenario.targets,
        scenario.private_targets,
        scenario.private_targets,
        scenario.agents,
        (),
        (),
        (),
    )
    validate_runtime_invariants(state.agents, state.target_locks, state.completion_events)
    return state


def _mode_value(mode: object) -> str:
    value = getattr(mode, "value", mode)
    if value not in _MODES:
        raise ValueError("unknown action mode")
    return str(value)


def _first_decisions(
    candidates: tuple[object, ...],
) -> tuple[tuple[int, int, Mode], ...]:
    decisions: list[tuple[int, int, Mode]] = []
    for candidate in candidates:
        if isinstance(candidate, PlannedPath):
            if candidate.tasks:
                target_id, mode = candidate.tasks[0]
                decisions.append((candidate.agent_id, target_id, _mode_value(mode)))
            continue
        if not isinstance(candidate, tuple) or len(candidate) != 3:
            raise TypeError("commit candidates must be PlannedPath or (agent, target, mode)")
        agent_id, target_id, mode = candidate
        if isinstance(agent_id, bool) or not isinstance(agent_id, int):
            raise TypeError("agent ID must be an integer")
        if isinstance(target_id, bool) or not isinstance(target_id, int):
            raise TypeError("target ID must be an integer")
        decisions.append((agent_id, target_id, _mode_value(mode)))
    return tuple(decisions)


def commit_batch(
    state: SimulatorState,
    candidates: tuple[object, ...],
    config: DynamicConfig,
    max_tick: int,
    *,
    tick: int | None = None,
) -> CommitBatchResult:
    """Validate an entire winner batch, then atomically commit only each first leg."""

    commit_tick = state.tick if tick is None else tick
    if commit_tick < state.tick or commit_tick > max_tick:
        raise ValueError("commit tick is outside the active scheduler interval")
    snapshot = replace(state, tick=commit_tick).snapshot()
    planned = tuple(item for item in candidates if isinstance(item, PlannedPath))
    if planned and len(planned) != len(candidates):
        raise TypeError("a commit batch cannot mix planned and direct candidates")
    if planned:
        gates = validate_commit_candidates(snapshot, config, max_tick, planned)
        if gates:
            return CommitBatchResult(state, (), gates)
    decisions = _first_decisions(candidates)
    target_ids = tuple(target for _, target, _ in decisions)
    if len(target_ids) != len(set(target_ids)):
        gate = GateFailure("commit_batch", commit_tick, "duplicate_target")
        return CommitBatchResult(state, (), (gate,))
    agent_ids = tuple(agent for agent, _, _ in decisions)
    if len(agent_ids) != len(set(agent_ids)):
        gate = GateFailure("commit_batch", commit_tick, "duplicate_agent")
        return CommitBatchResult(state, (), (gate,))

    locks = dict(state.target_locks)
    failures: list[GateFailure] = []
    prepared: list[tuple[int, int, str, float, int, int]] = []
    for agent_id, target_id, mode in decisions:
        if not 0 <= agent_id < len(state.agents):
            failures.append(GateFailure("commit_batch", commit_tick, "unknown_agent"))
            continue
        if not 0 <= target_id < len(state.targets):
            failures.append(GateFailure("commit_batch", commit_tick, "unknown_target"))
            continue
        agent = state.agents[agent_id]
        target = state.targets[target_id]
        if agent.active_action is not None:
            failures.append(GateFailure("commit_batch", commit_tick, "busy_agent"))
            continue
        if target_id in locks:
            failures.append(GateFailure("commit_batch", commit_tick, "target_locked"))
            continue
        travel = math.hypot(
            target.position[0] - agent.position[0], target.position[1] - agent.position[1]
        )
        start_tick, finish_tick = commit_ticks(
            commit_tick, agent.position, target.position, mode, config
        )
        ammo = 1.0 if mode == "attack" else 0.0
        if finish_tick > max_tick:
            failures.append(GateFailure("commit_batch", commit_tick, "horizon"))
        elif travel > agent.distance.available:
            failures.append(GateFailure("commit_batch", commit_tick, "range"))
        elif ammo > agent.ammo.available:
            failures.append(GateFailure("commit_batch", commit_tick, "ammo"))
        else:
            prepared.append((agent_id, target_id, mode, travel, start_tick, finish_tick))
    if failures:
        return CommitBatchResult(state, (), tuple(failures))

    agents = list(state.agents)
    ordinal_map = {(target, mode): value for target, mode, value in state.ordinals}
    actions: list[ActionCommit] = []
    events = list(state.completion_events)
    for agent_id, target_id, mode, travel, start_tick, finish_tick in prepared:
        key = (target_id, mode)
        ordinal = ordinal_map.get(key, 0)
        ordinal_map[key] = ordinal + 1
        ammo = 1.0 if mode == "attack" else 0.0
        action = ActionCommit(
            agent_id,
            target_id,
            mode,
            agents[agent_id].position,
            state.targets[target_id].position,
            travel,
            ammo,
            travel,
            commit_tick,
            start_tick,
            finish_tick,
            ordinal,
        )
        old = agents[agent_id]
        agents[agent_id] = replace(
            old,
            ammo=old.ammo.reserve(ammo),
            distance=old.distance.reserve(travel),
            active_action=action,
        )
        locks[target_id] = agent_id
        actions.append(action)
        events.append(CompletionEvent.from_action(action))
    new_state = replace(
        state,
        tick=commit_tick,
        agents=tuple(agents),
        target_locks=tuple(sorted(locks.items())),
        completion_events=tuple(sorted(events, key=lambda event: event.heap_key)),
        ordinals=tuple(sorted((target, mode, value) for (target, mode), value in ordinal_map.items())),
        actions=state.actions + tuple(actions),
    )
    validate_runtime_invariants(
        new_state.agents, new_state.target_locks, new_state.completion_events
    )
    return CommitBatchResult(new_state, tuple(actions), ())


def advance_public_belief(
    belief: Belief,
    mode: Mode,
    observation: tuple[str, str] | str | None,
    config: DynamicConfig,
    *,
    tick: int = 0,
) -> tuple[Belief, GateFailure | None]:
    """Apply exactly one public completion update, guarding zero evidence."""

    try:
        if mode == "attack":
            updated = predict_attack(
                belief, config.attack_success_high, config.attack_success_low
            )
        elif mode == "recon":
            if observation not in _OBSERVATIONS:
                raise ValueError("invalid Recon observation")
            kernel = recon_kernel(
                config.recon_category_matrix, config.recon_damage_matrix
            )
            updated = bayes_update(belief, kernel, _OBSERVATIONS.index(observation))
        elif mode == "bda":
            if observation not in ("A", "D"):
                raise ValueError("invalid BDA observation")
            kernel = bda_kernel(config.bda_damage_matrix)
            updated = bayes_update(belief, kernel, ("A", "D").index(observation))
        else:
            raise ValueError("unknown action mode")
    except ValueError as error:
        if "zero predictive probability" not in str(error):
            raise
        return belief, GateFailure("belief", tick, "zero_evidence")
    result = tuple(float(value) for value in updated)
    if any(not math.isfinite(value) or value < 0.0 for value in result) or not math.isclose(
        math.fsum(result), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        return belief, GateFailure("belief", tick, "simplex")
    return result, None  # type: ignore[return-value]


def _draw_key(state: SimulatorState, action: ActionCommit) -> DrawKey:
    if "/" in state.crn_namespace:
        experiment_id, generator_version = state.crn_namespace.split("/", 1)
    else:  # D0 fixtures predate scenario-owned runtime namespaces.
        experiment_id, generator_version = _EXPERIMENT_ID, _D1_GENERATOR_VERSION
    return DrawKey(
        _RNG_VERSION,
        experiment_id,
        generator_version,
        state.cell_id,
        state.seed,
        "target",
        action.target_id,
        action.mode,
        action.ordinal,
        0,
    )


def _observation(
    state: SimulatorState,
    action: ActionCommit,
    truth: PrivateTarget,
    environment: EnvironmentModel,
) -> tuple[tuple[str, str] | str | None, Fraction, DrawKey]:
    key = _draw_key(state, action)
    draw = uniform01(key)
    category_column = 0 if truth.true_category == "H" else 1
    damage_column = 0 if truth.true_damage == "A" else 1
    if action.mode == "recon":
        probabilities = tuple(
            environment.recon_category_matrix[index // 2][category_column]
            * environment.recon_damage_matrix[index % 2][damage_column]
            for index in range(4)
        )
        return _OBSERVATIONS[categorical(key, probabilities)], draw, key
    if action.mode == "bda":
        probabilities = tuple(
            environment.bda_damage_matrix[index][damage_column] for index in range(2)
        )
        return ("A", "D")[categorical(key, probabilities)], draw, key
    return None, draw, key


def complete_event_batch(
    state: SimulatorState,
    events: Iterable[CompletionEvent],
    config: DynamicConfig,
    environment_model: EnvironmentModel | None = None,
) -> CompletionBatchResult:
    """Settle one exact-tick completion batch in deterministic target/agent order."""

    environment = EnvironmentModel.from_config(config) if environment_model is None else environment_model
    batch = tuple(sorted(events, key=lambda event: (event.target_id, event.agent_id)))
    if not batch:
        raise ValueError("completion batch must not be empty")
    tick = batch[0].finish_tick
    if any(event.finish_tick != tick for event in batch):
        raise ValueError("all completion events in a batch must share one tick")
    queued = set(state.completion_events)
    due = {event for event in queued if event.finish_tick == tick}
    if set(batch) != due or len(batch) != len(set(batch)):
        raise ValueError("completion batch must contain every event due at its tick exactly once")
    agents = list(state.agents)
    for event in batch:
        if event not in queued:
            raise ValueError("completion event is not active")
        action = agents[event.agent_id].active_action
        if action is None or CompletionEvent.from_action(action) != event:
            raise ValueError("completion event does not match an active action")

    targets = list(state.targets)
    private_targets = list(state.private_targets)
    locks = dict(state.target_locks)
    public: list[PublicCompletion] = []
    private: list[PrivateAuditEvent] = []
    gates: list[GateFailure] = []
    for event in batch:
        agent = agents[event.agent_id]
        action = agent.active_action
        if action is None:  # guarded above; keeps static typing explicit
            raise RuntimeError("missing active action")
        truth = private_targets[action.target_id]
        before = truth.true_damage
        observation, draw, counter_key = _observation(state, action, truth, environment)
        after = before
        success = False
        raw_reward = 0.0
        if action.mode == "attack" and before == "A":
            probability = (
                environment.attack_success_high
                if truth.true_category == "H"
                else environment.attack_success_low
            )
            success = draw < Fraction.from_float(probability)
            if success:
                after = "D"
                if not truth.first_destroyed_paid:
                    raw_reward = (
                        config.value_high if truth.true_category == "H" else config.value_low
                    )
        paid = truth.first_destroyed_paid or raw_reward > 0.0
        private_targets[action.target_id] = replace(
            truth, true_damage=after, first_destroyed_paid=paid
        )
        belief, gate = advance_public_belief(
            targets[action.target_id].belief,
            action.mode,
            observation,
            config,
            tick=tick,
        )
        if gate is not None:
            gates.append(gate)
        else:
            targets[action.target_id] = replace(targets[action.target_id], belief=belief)
        event_id = len(state.public_events) + len(public)
        if action.mode == "attack":
            public_event: PublicCompletion = PublicActionAck(
                event_id, tick, action.target_id, action.agent_id, "attack"
            )
        else:
            if observation is None:
                raise RuntimeError("sensing action did not produce an observation")
            public_event = PublicObservation(
                event_id,
                tick,
                action.target_id,
                action.agent_id,
                action.mode,
                observation,
            )
        initial_wreck = (
            action.mode == "attack"
            and state.initial_private_targets[action.target_id].true_damage == "D"
        )
        private_event = PrivateAuditEvent(
            event_id,
            tick,
            action.target_id,
            action.agent_id,
            action.mode,
            float(draw),
            truth.true_category,
            before,
            after,
            success,
            raw_reward,
            action.mode == "attack" and before == "D",
            initial_wreck,
            counter_key,
        )
        public.append(public_event)
        private.append(private_event)
        agents[action.agent_id] = replace(
            agent,
            position=action.destination,
            ammo=agent.ammo.consume(action.reserved_ammo),
            distance=agent.distance.consume(action.reserved_distance),
            active_action=None,
        )
        del locks[action.target_id]
        queued.remove(event)
    new_state = replace(
        state,
        tick=tick,
        targets=tuple(targets),
        private_targets=tuple(private_targets),
        agents=tuple(agents),
        target_locks=tuple(sorted(locks.items())),
        completion_events=tuple(sorted(queued, key=lambda event: event.heap_key)),
        public_events=state.public_events + tuple(public),
        private_audit_events=state.private_audit_events + tuple(private),
        gates=state.gates + tuple(gates),
    )
    validate_runtime_invariants(
        new_state.agents, new_state.target_locks, new_state.completion_events
    )
    return CompletionBatchResult(new_state, tuple(public), tuple(private), tuple(gates))


def evaluate_real_utility(
    scenario: DynamicScenario,
    actions: tuple[ActionCommit, ...],
    audit_events: tuple[PrivateAuditEvent, ...],
    config: DynamicConfig,
) -> UtilityEvaluation:
    """Reconstruct the frozen real-utility decomposition from private audit data."""

    service_costs = {
        "recon": config.recon_service_cost,
        "attack": config.attack_service_cost,
        "bda": config.bda_service_cost,
    }
    unmatched = list(actions)
    completed_actions: list[ActionCommit] = []
    for event in audit_events:
        matches = [
            action
            for action in unmatched
            if (
                action.finish_tick,
                action.target_id,
                action.agent_id,
                action.mode,
            )
            == (event.tick, event.target_id, event.agent_id, event.mode)
        ]
        if len(matches) != 1:
            raise ValueError("private audit event must match exactly one committed action")
        completed_actions.append(matches[0])
        unmatched.remove(matches[0])
    service_cost = math.fsum(
        math.exp(-config.discount_rate * action.start_tick * config.tick_size)
        * service_costs[action.mode]
        for action in completed_actions
    )
    destroyed_value = math.fsum(
        math.exp(-config.discount_rate * event.tick * config.tick_size)
        * event.realized_reward
        for event in audit_events
    )
    distance_consumed = math.fsum(action.reserved_distance for action in completed_actions)
    ammo_consumed = math.fsum(action.reserved_ammo for action in completed_actions)
    distance_cost = config.distance_cost_rate * distance_consumed
    ammo_cost = config.ammo_cost_rate * ammo_consumed
    realized = destroyed_value - service_cost - distance_cost - ammo_cost
    gross = math.fsum(
        config.value_high if target.true_category == "H" else config.value_low
        for target in scenario.private_targets
    )
    if gross <= 0.0:
        raise ValueError("gross scenario value must be positive")
    return UtilityEvaluation(
        destroyed_value,
        service_cost,
        distance_cost,
        ammo_cost,
        realized,
        realized / gross,
        gross,
        distance_consumed,
        ammo_consumed,
    )


def _positive_count(policy: object, snapshot: PublicSnapshot) -> int:
    counter = getattr(policy, "positive_pair_count", None)
    if counter is None:
        return 0
    value = counter(snapshot)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("positive_pair_count must return a nonnegative integer")
    return value


def _planning_clock(policy: object) -> PlanningClock:
    clock = getattr(policy, "planning_clock", None)
    if not isinstance(clock, PlanningClock):
        raise TypeError("policy planning_clock must be a PlanningClock")
    if (
        clock is PlanningClock.B1M_ONE_SHOT
        and getattr(policy, "method", None) not in {"B1m", "SCBBA"}
    ):
        raise TypeError("B1M_ONE_SHOT is reserved for registered one-shot baselines")
    return clock


def _policy_candidates(decision: object) -> tuple[object, ...]:
    """Adapt a public PolicyDecision while retaining legacy scripted tuples."""

    proposals = getattr(decision, "proposals", None)
    if proposals is None:
        return tuple(decision)  # type: ignore[arg-type]
    return tuple(proposal.commit_candidate() for proposal in proposals)


def _policy_gates(decision: object) -> tuple[GateFailure, ...]:
    gates = getattr(decision, "gates", ())
    if any(not isinstance(gate, GateFailure) for gate in gates):
        raise TypeError("policy decision gates must be GateFailure values")
    return tuple(gates)


def _has_pending_suffix(policy: object) -> bool:
    probe = getattr(policy, "has_pending_suffix", None)
    return bool(probe()) if probe is not None else False


def _brier(targets: tuple[PublicTarget, ...], truth: tuple[PrivateTarget, ...]) -> float:
    if not targets:
        return 0.0
    order = {("H", "A"): 0, ("H", "D"): 1, ("L", "A"): 2, ("L", "D"): 3}
    return math.fsum(
        math.fsum((probability - float(index == order[(actual.true_category, actual.true_damage)])) ** 2
                  for index, probability in enumerate(public.belief))
        for public, actual in zip(targets, truth, strict=True)
    ) / len(targets)


def run_episode(
    scenario: DynamicScenario,
    policy: object,
    *,
    config: DynamicConfig | None = None,
    method: str = "scripted",
    planning_grid_ticks: tuple[int, ...] = (),
    environment_model: EnvironmentModel | None = None,
) -> EpisodeResult:
    """Run one policy using public snapshots and simulator-private counter draws only."""

    cfg = DynamicConfig() if config is None else config
    environment = EnvironmentModel.from_config(cfg) if environment_model is None else environment_model
    state = initialize_state(scenario)
    binder = getattr(policy, "bind_horizon", None)
    if binder is not None:
        binder(scenario.t_max_tick)
    clock = _planning_clock(policy)
    initial_public_digest = public_digest(state.snapshot())
    initial_truth_digest = private_truth_digest(state.private_targets)
    if not planning_grid_ticks and clock is PlanningClock.PERIODIC:
        grid_provider = getattr(policy, "planning_grid_ticks", None)
        if grid_provider is not None:
            planning_grid_ticks = tuple(grid_provider())
    grids = tuple(sorted(set(tick for tick in planning_grid_ticks if 0 < tick <= scenario.t_max_tick)))
    minimum_ticks = quantize_tick(cfg.minimum_duration, cfg.tick_size)
    physical_bound = len(state.agents) * (scenario.t_max_tick // minimum_ticks + 1)
    guard = physical_bound + len(grids) + 2
    scheduler_steps = 0
    replan_count = 0
    termination = "no_positive"

    def plan_once(
        current: SimulatorState,
    ) -> tuple[SimulatorState, bool, int, tuple[GateFailure, ...]]:
        nonlocal replan_count
        snapshot = current.snapshot()
        decision = policy.decide(snapshot)
        replan_count += 1
        positive = _positive_count(policy, snapshot)
        decision_gates = _policy_gates(decision)
        if decision_gates:
            return (
                replace(current, gates=current.gates + decision_gates),
                False,
                positive,
                decision_gates,
            )
        decisions = _policy_candidates(decision)
        outcome = commit_batch(current, decisions, cfg, scenario.t_max_tick)
        if outcome.gates:
            return (
                replace(current, gates=current.gates + outcome.gates),
                False,
                positive,
                outcome.gates,
            )
        return outcome.state, bool(outcome.committed), positive, ()

    state, committed, positive, plan_gates = plan_once(state)
    fatal_termination: str | None = None
    if plan_gates and not state.completion_events:
        termination = plan_gates[-1].reason
    elif not committed and not state.completion_events:
        if positive:
            gate = GateFailure("scheduler", state.tick, "allocation_stall")
            state = replace(state, gates=state.gates + (gate,))
            termination = "allocation_stall"
        else:
            termination = "no_positive"
    else:
        while True:
            scheduler_steps += 1
            if fatal_termination is None and (
                scheduler_steps > guard or len(state.actions) > physical_bound
            ):
                gate = GateFailure("scheduler", state.tick, "event_guard")
                state = replace(state, gates=state.gates + (gate,))
                fatal_termination = gate.reason
                if not state.completion_events:
                    termination = fatal_termination
                    break
            future_grids = tuple(tick for tick in grids if tick > state.tick)
            next_grid = (
                future_grids[0]
                if fatal_termination is None
                and clock is PlanningClock.PERIODIC
                and future_grids
                else scenario.t_max_tick
            )
            next_completion = (
                state.completion_events[0].finish_tick
                if state.completion_events
                else scenario.t_max_tick
            )
            next_tick = min(next_completion, next_grid, scenario.t_max_tick)
            state = replace(state, tick=next_tick)
            due = tuple(
                event for event in state.completion_events if event.finish_tick == next_tick
            )
            if due:
                completion = complete_event_batch(state, due, cfg, environment)
                state = completion.state
                if completion.gates and fatal_termination is None:
                    fatal_termination = completion.gates[-1].reason
            if fatal_termination is not None:
                if state.completion_events:
                    continue
                termination = fatal_termination
                break
            if due and clock is PlanningClock.B1M_ONE_SHOT:
                auto_next = getattr(policy, "auto_next", None)
                if auto_next is None:
                    raise TypeError("B1m one-shot policy must provide auto_next")
                decision = auto_next(
                    state.snapshot(), tuple(event.agent_id for event in due)
                )
                decision_gates = _policy_gates(decision)
                if decision_gates:
                    state = replace(state, gates=state.gates + decision_gates)
                    fatal_termination = decision_gates[-1].reason
                    if state.completion_events:
                        continue
                    termination = fatal_termination
                    break
                candidates = _policy_candidates(decision)
                if candidates:
                    outcome = commit_batch(
                        state, candidates, cfg, scenario.t_max_tick, tick=state.tick
                    )
                    if outcome.gates:
                        suffix_gates = tuple(
                            GateFailure(
                                "b1m_suffix",
                                state.tick,
                                "infeasible",
                                (("commit_reason", gate.reason),),
                            )
                            for gate in outcome.gates
                        )
                        state = replace(state, gates=state.gates + suffix_gates)
                        fatal_termination = "infeasible"
                        if state.completion_events:
                            continue
                        termination = fatal_termination
                        break
                    state = outcome.state
                elif _has_pending_suffix(policy) and not state.completion_events:
                    gate = GateFailure("b1m_suffix", state.tick, "auto_next_stall")
                    state = replace(state, gates=state.gates + (gate,))
                    termination = gate.reason
                    break
            if next_tick == scenario.t_max_tick:
                termination = "horizon"
                break
            is_grid = clock is PlanningClock.PERIODIC and next_tick in grids
            should_plan = (
                clock is PlanningClock.EVENT_DRIVEN and bool(due)
            ) or is_grid
            if should_plan:
                state, committed, positive, plan_gates = plan_once(state)
                if plan_gates and not state.completion_events:
                    termination = plan_gates[-1].reason
                    break
                if not state.completion_events and not committed:
                    if positive:
                        gate = GateFailure("scheduler", state.tick, "allocation_stall")
                        state = replace(state, gates=state.gates + (gate,))
                        termination = "allocation_stall"
                    else:
                        termination = "normal" if state.actions else "no_positive"
                    break
            elif not state.completion_events:
                if _has_pending_suffix(policy):
                    gate = GateFailure("b1m_suffix", state.tick, "pending_without_event")
                    state = replace(state, gates=state.gates + (gate,))
                    termination = gate.reason
                    break
                positive = _positive_count(policy, state.snapshot())
                if positive == 0:
                    termination = "normal" if state.actions else "no_positive"
                    break

    if (
        state.completion_events
        or any(agent.active_action is not None for agent in state.agents)
        or any(agent.ammo.reserved or agent.distance.reserved for agent in state.agents)
        or len(state.actions) != len(state.private_audit_events)
    ):
        raise RuntimeError("episode terminated with unsettled committed actions")

    utility = evaluate_real_utility(
        scenario, state.actions, state.private_audit_events, cfg
    )
    attack_events = tuple(event for event in state.private_audit_events if event.mode == "attack")
    previous_by_target: dict[int, PrivateAuditEvent] = {}
    continuous = 0
    handoffs = 0
    for event in state.private_audit_events:
        previous = previous_by_target.get(event.target_id)
        if event.mode == "attack" and previous is not None and previous.mode == "attack":
            continuous += 1
        if previous is not None and previous.agent_id != event.agent_id:
            handoffs += 1
        previous_by_target[event.target_id] = event
    makespan_tick = max((event.tick for event in state.private_audit_events), default=0)
    record = EpisodeRecord(
        scenario.scenario_id,
        scenario.cell_id,
        method,
        initial_truth_digest,
        private_truth_digest(state.private_targets),
        initial_public_digest,
        termination,
        "gate_failure" if state.gates else "complete",
        len(state.private_audit_events),
        len(state.actions),
        replan_count,
        tuple(target.belief for target in state.targets),
        utility.destroyed_value,
        utility.service_cost,
        utility.distance_cost,
        utility.ammo_cost,
        utility.realized_utility,
        utility.normalized_utility,
        utility.gross_scenario_value,
        utility.distance_consumed,
        utility.ammo_consumed,
        makespan_tick * cfg.tick_size,
        utility.destroyed_value,
        sum(event.invalid_attack for event in attack_events),
        sum(event.initial_wreck_attack for event in attack_events),
        sum(event.mode == "recon" for event in state.private_audit_events),
        sum(event.mode == "bda" for event in state.private_audit_events),
        continuous,
        handoffs,
        0,
        sum(gate.reason == "allocation_stall" for gate in state.gates),
        0,
        _brier(state.targets, state.private_targets),
        state.gates,
        "not_checked",
    )
    return EpisodeResult(
        record,
        state.public_events,
        state.private_audit_events,
        state.gates,
        state.actions,
    )


__all__ = [
    "CommitBatchResult",
    "CompletionBatchResult",
    "SimulatorState",
    "UtilityEvaluation",
    "advance_public_belief",
    "commit_batch",
    "complete_event_batch",
    "evaluate_real_utility",
    "initialize_state",
    "run_episode",
]
