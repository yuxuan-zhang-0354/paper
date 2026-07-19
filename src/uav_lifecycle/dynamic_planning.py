"""Information-safe adapters from public dynamic state to frozen mode allocators."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, factorial, isclose

from .belief import bda_kernel, recon_kernel
from .dynamic_types import DynamicConfig, GateFailure, PublicSnapshot, quantize_tick
from .mode_allocation import (
    TOL,
    Mode,
    ModeAgent,
    ModeInstance,
    ModeKey,
    ModeTask,
    TickPathTiming,
    evaluate_mode_path,
    mode_utilities,
)
from .mode_cbba import ModeMethod, run_mode_cbba, screen_modes, validate_mode_result
from .mode_exact import (
    ExactSearchBoundExceeded,
    ModeExactResult,
    solve_all_mode_exact,
    validate_exact_audit,
)
from .rollout import RolloutParameters


_MODE_RANK = {Mode.RECON: 0, Mode.ATTACK: 1, Mode.BDA: 2}


@dataclass(frozen=True, slots=True)
class PlanningProblem:
    instance: ModeInstance
    global_agent_ids: tuple[int, ...]
    global_target_ids: tuple[int, ...]
    continuation_gates: tuple[dict[Mode, bool], ...]


@dataclass(frozen=True, slots=True)
class PlannedPath:
    agent_id: int
    tasks: tuple[ModeKey, ...]
    score: float
    start_ticks: tuple[int, ...]
    completion_ticks: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.tasks) != len(self.start_ticks) or len(self.tasks) != len(self.completion_ticks):
            raise ValueError("tasks and tick sequences must have equal lengths")


@dataclass(frozen=True, slots=True)
class PlanningResult:
    method: str
    status: str
    paths: tuple[PlannedPath, ...]
    positive_pair_count: int
    rounds: int = 0
    gates: tuple[GateFailure, ...] = ()
    search_bound: int | None = None
    search_count: int | None = None
    allocator_status: str = "converged"
    message_packets: int = 0
    message_scalars: int = 0
    allocation_objective: float = 0.0
    audit_report: tuple[tuple[str, int], ...] = ()
    eligible_pair_count: int = 0
    screened_positive_pair_count: int | None = None
    warping_activations: int = 0
    raw_prefix_increases: int = 0


@dataclass(frozen=True, slots=True)
class Tick0TargetScreening:
    region: str
    positive_single_task: bool
    resource_blocked: bool


def _params(config: DynamicConfig) -> RolloutParameters:
    return RolloutParameters(
        config.value_high,
        config.value_low,
        config.attack_success_high,
        config.attack_success_low,
        config.recon_duration,
        config.attack_duration,
        config.bda_duration,
        config.recon_service_cost,
        config.attack_service_cost,
        config.bda_service_cost,
        config.discount_rate,
    )


def _duration(config: DynamicConfig, mode: Mode) -> float:
    return {
        Mode.RECON: config.recon_duration,
        Mode.ATTACK: config.attack_duration,
        Mode.BDA: config.bda_duration,
    }[mode]


def _ammo(mode: Mode) -> int:
    return int(mode is Mode.ATTACK)


def commit_ticks(
    commit_tick: int,
    origin: tuple[float, float],
    destination: tuple[float, float],
    mode: Mode,
    config: DynamicConfig,
) -> tuple[int, int]:
    """Shared scheduler contract for one predicted/committed first leg."""

    start_tick = commit_tick + quantize_tick(dist(origin, destination) / config.speed, config.tick_size)
    finish_tick = start_tick + quantize_tick(_duration(config, mode), config.tick_size)
    return start_tick, finish_tick


def _validate_inputs(snapshot: PublicSnapshot, config: DynamicConfig, max_tick: int) -> None:
    if not isinstance(snapshot, PublicSnapshot) or not isinstance(config, DynamicConfig):
        raise TypeError("dynamic planning accepts only PublicSnapshot and DynamicConfig state")
    if isinstance(max_tick, bool) or not isinstance(max_tick, int) or max_tick < snapshot.tick:
        raise ValueError("max_tick must be an integer no earlier than the snapshot")


def target_continuation_gates(
    snapshot: PublicSnapshot, config: DynamicConfig, max_tick: int
) -> dict[int, dict[Mode, bool]]:
    """Return the frozen target/mode existence proxy from idle available resources."""

    _validate_inputs(snapshot, config, max_tick)
    locked = dict(snapshot.target_locks)
    idle = tuple(agent for agent in snapshot.agents if agent.busy_action is None)
    gates: dict[int, dict[Mode, bool]] = {}
    attack_ticks = quantize_tick(config.attack_duration, config.tick_size)
    for target in snapshot.targets:
        if target.target_id in locked:
            continue
        per_mode: dict[Mode, bool] = {}
        for mode in Mode:
            duration_ticks = quantize_tick(_duration(config, mode), config.tick_size)
            per_mode[mode] = any(
                agent.available_ammo >= _ammo(mode) + 1
                and (leg := dist(agent.position, target.position)) <= agent.available_distance + TOL
                and snapshot.tick
                + quantize_tick(leg / config.speed, config.tick_size)
                + duration_ticks
                + attack_ticks
                <= max_tick
                for agent in idle
            )
        gates[target.target_id] = per_mode
    return gates


def build_planning_problem(
    snapshot: PublicSnapshot,
    config: DynamicConfig,
    max_tick: int,
    *,
    allowed_modes: dict[int, tuple[Mode, ...]] | None = None,
) -> PlanningProblem:
    """Compact idle/unlocked public IDs and build one frozen mode instance."""

    _validate_inputs(snapshot, config, max_tick)
    global_agents = tuple(agent.agent_id for agent in snapshot.agents if agent.busy_action is None)
    locked = set(dict(snapshot.target_locks))
    global_targets = tuple(
        target.target_id
        for target in snapshot.targets
        if target.target_id not in locked
        and (allowed_modes is None or allowed_modes.get(target.target_id, tuple(Mode)))
    )
    gate_map = target_continuation_gates(snapshot, config, max_tick)
    agents = tuple(
        ModeAgent(
            local_id,
            snapshot.agents[global_id].position,
            max_tick * config.tick_size,
            snapshot.agents[global_id].available_distance,
            int(snapshot.agents[global_id].available_ammo),
        )
        for local_id, global_id in enumerate(global_agents)
    )
    zr = recon_kernel(config.recon_category_matrix, config.recon_damage_matrix)
    zb = bda_kernel(config.bda_damage_matrix)
    params = _params(config)
    groups: list[tuple[ModeTask, ...]] = []
    local_gates: list[dict[Mode, bool]] = []
    for local_target, global_target in enumerate(global_targets):
        target = snapshot.targets[global_target]
        gates = gate_map[global_target]
        utilities = mode_utilities(
            target.belief,
            zr,
            zb,
            params,
            "ammo_reachability_gate",
            gates,
        )
        permitted = tuple(Mode) if allowed_modes is None else allowed_modes.get(global_target, tuple(Mode))
        groups.append(tuple(
            ModeTask(
                local_target,
                mode,
                target.position,
                _duration(config, mode),
                _ammo(mode),
                utilities[mode],
            )
            for mode in permitted
        ))
        local_gates.append(gates)
    instance = ModeInstance(
        agents,
        tuple(groups),
        config.discount_rate,
        config.distance_cost_rate,
        config.ammo_cost_rate,
        instance_id=f"dynamic-{snapshot.tick}",
        continuation="ammo_reachability_gate",
        timing=TickPathTiming(snapshot.tick, max_tick, config.tick_size, config.speed),
    )
    return PlanningProblem(instance, global_agents, global_targets, tuple(local_gates))


def _planned_path(problem: PlanningProblem, local_agent: int, path: tuple[ModeKey, ...]) -> PlannedPath:
    evaluation = evaluate_mode_path(problem.instance, local_agent, path)
    if evaluation.start_ticks is None:
        raise RuntimeError("dynamic planner requires tick-aware evaluation")
    completion_ticks = tuple(
        evaluate_mode_path(problem.instance, local_agent, path[:index]).completion_tick
        for index in range(1, len(path) + 1)
    )
    if any(tick is None for tick in completion_ticks):
        raise RuntimeError("dynamic path completion ticks are missing")
    tasks = tuple((problem.global_target_ids[target], mode) for target, mode in path)
    return PlannedPath(
        problem.global_agent_ids[local_agent],
        tasks,
        evaluation.score,
        evaluation.start_ticks,
        tuple(int(tick) for tick in completion_ticks),
    )


def _positive_pairs(problem: PlanningProblem, screened) -> int:
    return sum(
        evaluation.feasible and evaluation.score > TOL
        for agent_id in range(len(problem.instance.agents))
        for task in screened
        if task is not None
        for evaluation in (evaluate_mode_path(problem.instance, agent_id, (task.key,)),)
    )


def positive_single_task_targets(
    snapshot: PublicSnapshot,
    config: DynamicConfig,
    max_tick: int,
    *,
    allowed_modes: dict[int, tuple[Mode, ...]],
) -> frozenset[int]:
    """Return public targets having a positive screened singleton pair."""

    problem = build_planning_problem(
        snapshot, config, max_tick, allowed_modes=allowed_modes
    )
    screened = screen_modes(problem.instance)
    positive: set[int] = set()
    for local_target, task in enumerate(screened):
        if task is None:
            continue
        if any(
            evaluation.feasible and evaluation.score > TOL
            for agent_id in range(len(problem.instance.agents))
            for evaluation in (
                evaluate_mode_path(problem.instance, agent_id, (task.key,)),
            )
        ):
            positive.add(problem.global_target_ids[local_target])
    return frozenset(positive)


def tick0_target_screening(
    snapshot: PublicSnapshot, config: DynamicConfig, max_tick: int,
) -> tuple[Tick0TargetScreening, ...]:
    """Reuse production screening to expose method-blind public t0 coverage."""

    problem = build_planning_problem(snapshot, config, max_tick)
    screened = screen_modes(problem.instance)
    positive = tuple(task for task in screened if task is not None)
    blocked_targets: set[int] = set()
    for agent_id in range(len(problem.instance.agents)):
        for first in positive:
            first_eval = evaluate_mode_path(problem.instance, agent_id, (first.key,))
            if not first_eval.feasible or first_eval.score <= TOL:
                continue
            for second in positive:
                if first.target_id == second.target_id:
                    continue
                second_eval = evaluate_mode_path(problem.instance, agent_id, (second.key,))
                pair_eval = evaluate_mode_path(
                    problem.instance, agent_id, (first.key, second.key),
                )
                if (
                    second_eval.feasible
                    and second_eval.score > TOL
                    and not pair_eval.feasible
                ):
                    blocked_targets.update((first.target_id, second.target_id))
    rows = []
    for local_target, selected in enumerate(screened):
        rows.append(Tick0TargetScreening(
            "Defer" if selected is None else {
                Mode.RECON: "R", Mode.ATTACK: "A", Mode.BDA: "B",
            }[selected.mode],
            selected is not None,
            local_target in blocked_targets,
        ))
    return tuple(rows)


def validate_johnson_audit(
    tick: int,
    result,
    report: dict[str, int],
    positive_pair_count: int,
) -> tuple[GateFailure, ...]:
    """Convert each existing Johnson structural audit into deterministic Gates."""

    gates = [
        GateFailure("johnson", tick, key, (("count", str(value)),))
        for key, value in report.items()
        if value and key != "cycle_or_timeout"
    ]
    if result.status == "cycle":
        gates.append(GateFailure("johnson", tick, "cycle", (("period", str(result.cycle_period)),)))
    elif result.status != "converged":
        gates.append(GateFailure("johnson", tick, "round_cap"))
    if positive_pair_count and not any(state.path for state in result.states):
        gates.append(GateFailure("johnson", tick, "allocation_stall"))
    return tuple(gates)


def validate_raw_audit(
    tick: int,
    result,
    report: dict[str, int],
    positive_pair_count: int,
    *,
    gate_name: str = "vanilla_raw",
) -> tuple[GateFailure, ...]:
    """Expose raw-allocator nonconvergence as an explicit failed epoch."""

    gates = [
        GateFailure(gate_name, tick, key, (("count", str(value)),))
        for key, value in report.items()
        if value and key != "cycle_or_timeout"
    ]
    if result.status == "cycle":
        gates.append(
            GateFailure(
                gate_name,
                tick,
                "cycle",
                (("period", str(result.cycle_period)),),
            )
        )
    elif result.status != "converged":
        gates.append(GateFailure(gate_name, tick, "round_cap"))
    elif positive_pair_count and not any(state.path for state in result.states):
        gates.append(GateFailure(gate_name, tick, "allocation_stall"))
    return tuple(gates)


def plan_johnson(
    snapshot: PublicSnapshot,
    config: DynamicConfig,
    max_tick: int,
    *,
    allowed_modes: dict[int, tuple[Mode, ...]] | None = None,
) -> PlanningResult:
    problem = build_planning_problem(
        snapshot, config, max_tick, allowed_modes=allowed_modes
    )
    screened = screen_modes(problem.instance)
    positive = _positive_pairs(problem, screened)
    result = run_mode_cbba(problem.instance, screened, ModeMethod.JOHNSON_WARPED)
    report = validate_mode_result(problem.instance, screened, result)
    gates = list(validate_johnson_audit(snapshot.tick, result, report, positive))
    paths = tuple(
        _planned_path(problem, state.agent_id, state.path)
        for state in result.states
        if state.path
    )
    status = "valid" if not gates else "gate_failure"
    return PlanningResult(
        "johnson", status, paths, positive, result.rounds, tuple(gates),
        allocator_status=result.status,
        message_packets=result.message_packets,
        message_scalars=result.message_scalars,
        allocation_objective=result.true_score,
        audit_report=tuple(sorted(report.items())),
        eligible_pair_count=len(problem.instance.agents) * sum(task is not None for task in screened),
        screened_positive_pair_count=positive,
        warping_activations=result.warping_activations,
        raw_prefix_increases=result.raw_prefix_increases,
    )


def plan_vanilla(
    snapshot: PublicSnapshot,
    config: DynamicConfig,
    max_tick: int,
) -> PlanningResult:
    """Classic retain-and-release CBBA with raw non-DMG marginal bids."""

    problem = build_planning_problem(snapshot, config, max_tick)
    screened = screen_modes(problem.instance)
    positive = _positive_pairs(problem, screened)
    result = run_mode_cbba(problem.instance, screened, ModeMethod.STANDARD_RAW)
    report = validate_mode_result(problem.instance, screened, result)
    gates = validate_raw_audit(snapshot.tick, result, report, positive)
    legal = not gates
    paths = tuple(
        _planned_path(problem, state.agent_id, state.path)
        for state in result.states if state.path
    ) if legal else ()
    return PlanningResult(
        "vanilla_raw", "valid" if legal else "allocator_nonconvergence",
        paths, positive, result.rounds, gates,
        allocator_status=result.status,
        message_packets=result.message_packets,
        message_scalars=result.message_scalars,
        allocation_objective=result.true_score,
        audit_report=tuple(sorted(report.items())),
        eligible_pair_count=len(problem.instance.agents) * sum(task is not None for task in screened),
        screened_positive_pair_count=positive,
        warping_activations=result.warping_activations,
        raw_prefix_increases=result.raw_prefix_increases,
    )


def plan_factorial_variant(
    snapshot: PublicSnapshot,
    config: DynamicConfig,
    max_tick: int,
    method: ModeMethod,
) -> PlanningResult:
    """Run one registered raw/warped x retain/rebuild allocator variant."""

    problem = build_planning_problem(snapshot, config, max_tick)
    screened = screen_modes(problem.instance)
    positive = _positive_pairs(problem, screened)
    result = run_mode_cbba(problem.instance, screened, method)
    report = validate_mode_result(problem.instance, screened, result)
    gates = validate_raw_audit(
        snapshot.tick, result, report, positive, gate_name=method.value
    )
    legal = not gates
    paths = tuple(
        _planned_path(problem, state.agent_id, state.path)
        for state in result.states if state.path
    ) if legal else ()
    return PlanningResult(
        method.value,
        "valid" if legal else "allocator_nonconvergence",
        paths,
        positive,
        result.rounds,
        gates,
        allocator_status=result.status,
        message_packets=result.message_packets,
        message_scalars=result.message_scalars,
        allocation_objective=result.true_score,
        audit_report=tuple(sorted(report.items())),
        eligible_pair_count=len(problem.instance.agents) * sum(task is not None for task in screened),
        screened_positive_pair_count=positive,
        warping_activations=result.warping_activations,
        raw_prefix_increases=result.raw_prefix_increases,
    )


def plan_nearest_positive(snapshot: PublicSnapshot, config: DynamicConfig, max_tick: int) -> PlanningResult:
    problem = build_planning_problem(snapshot, config, max_tick)
    screened = screen_modes(problem.instance)
    pairs: list[tuple[float, float, int, int, tuple[ModeKey, ...]]] = []
    for agent_id in range(len(problem.instance.agents)):
        for task in screened:
            if task is None:
                continue
            path = (task.key,)
            evaluation = evaluate_mode_path(problem.instance, agent_id, path)
            if evaluation.feasible and evaluation.score > TOL:
                pairs.append((evaluation.distance, evaluation.score, task.target_id, agent_id, path))
    frozen = tuple(sorted(
        pairs,
        key=lambda item: nearest_positive_order_key(item[0], item[1], item[2], item[3]),
    ))
    used_agents: set[int] = set()
    used_targets: set[int] = set()
    selected: list[PlannedPath] = []
    for _, _, target_id, agent_id, path in frozen:
        if agent_id in used_agents or target_id in used_targets:
            continue
        selected.append(_planned_path(problem, agent_id, path))
        used_agents.add(agent_id)
        used_targets.add(target_id)
    selected_paths = tuple(selected)
    return PlanningResult(
        "nearest_positive", "valid", selected_paths, len(frozen),
        allocation_objective=sum(path.score for path in selected_paths),
        eligible_pair_count=len(problem.instance.agents) * sum(task is not None for task in screened),
        screened_positive_pair_count=len(frozen),
    )


def nearest_positive_order_key(
    incremental_distance: float,
    unwarped_public_marginal: float,
    target_id: int,
    agent_id: int,
) -> tuple[float, float, int, int]:
    return incremental_distance, -unwarped_public_marginal, target_id, agent_id


def exact_audit_gates(
    tick: int, result: ModeExactResult, search_bound: int
) -> tuple[GateFailure, ...]:
    return tuple(
        GateFailure("all_mode_exact", tick, reason)
        for reason in validate_exact_audit(result, search_bound)
    )


def run_exact_search(
    instance: ModeInstance,
    tick: int,
    search_bound: int,
) -> tuple[ModeExactResult | None, tuple[GateFailure, ...], int]:
    """Run bounded exact search; structural failure never exposes an incumbent."""

    try:
        result = solve_all_mode_exact(instance, operation_cap=search_bound)
    except ExactSearchBoundExceeded as error:
        gate = GateFailure(
            "all_mode_exact",
            tick,
            "search_bound_exceeded",
            (("actual_count", str(error.actual_count)), ("search_bound", str(search_bound))),
        )
        return None, (gate,), error.actual_count
    gates = exact_audit_gates(tick, result, search_bound)
    if gates:
        return None, gates, result.audit.profile_equivalent_count
    return result, (), result.audit.profile_equivalent_count


def plan_all_mode_exact(snapshot: PublicSnapshot, config: DynamicConfig, max_tick: int) -> PlanningResult:
    problem = build_planning_problem(snapshot, config, max_tick)
    target_count = len(problem.global_target_ids)
    if target_count > 5:
        raise ValueError("all-mode exact supports at most 5 unlocked targets")
    search_bound = factorial(target_count) * (1 + 3 * len(problem.global_agent_ids)) ** target_count
    result, audit_gates, search_count = run_exact_search(
        problem.instance, snapshot.tick, search_bound
    )
    if result is None:
        return PlanningResult(
            "all_mode_exact",
            "gate_failure",
            (),
            0,
            gates=audit_gates,
            search_bound=search_bound,
            search_count=search_count,
        )
    assigned_targets = [target for path in result.paths for target, _ in path]
    if len(assigned_targets) != len(set(assigned_targets)):
        gate = GateFailure("all_mode_exact", snapshot.tick, "target_mode_conflict")
        return PlanningResult(
            "all_mode_exact", "gate_failure", (), 0, gates=(gate,), search_bound=search_bound
        )
    paths = tuple(
        _planned_path(problem, agent_id, path)
        for agent_id, path in enumerate(result.paths)
        if path
    )
    positive = sum(
        evaluation.feasible and evaluation.score > TOL
        for agent_id in range(len(problem.instance.agents))
        for target_id, tasks in enumerate(problem.instance.tasks_by_target)
        for task in tasks
        for evaluation in (evaluate_mode_path(problem.instance, agent_id, ((target_id, task.mode),)),)
    )
    return PlanningResult(
        "all_mode_exact", "valid", paths, positive,
        search_bound=search_bound, search_count=result.audit.profile_equivalent_count,
        allocation_objective=result.score,
        eligible_pair_count=(
            len(problem.instance.agents)
            * sum(len(tasks) for tasks in problem.instance.tasks_by_target)
        ),
        screened_positive_pair_count=positive,
    )


def validate_commit_candidates(
    snapshot: PublicSnapshot,
    config: DynamicConfig,
    max_tick: int,
    candidates: tuple[PlannedPath, ...],
) -> tuple[GateFailure, ...]:
    """Pure whole-batch validation; any defect rejects the complete candidate tuple."""

    problem = build_planning_problem(snapshot, config, max_tick)
    failures: list[GateFailure] = []
    idle = set(problem.global_agent_ids)
    seen_agents: set[int] = set()
    first_targets = [candidate.tasks[0][0] for candidate in candidates if candidate.tasks]
    if len(first_targets) != len(set(first_targets)):
        return (GateFailure("commit_batch", snapshot.tick, "duplicate_target"),)
    for candidate in candidates:
        if candidate.agent_id not in idle:
            failures.append(GateFailure("commit_batch", snapshot.tick, "non_idle_winner"))
            continue
        if candidate.agent_id in seen_agents:
            failures.append(GateFailure("commit_batch", snapshot.tick, "duplicate_agent"))
        seen_agents.add(candidate.agent_id)
        if not candidate.tasks:
            failures.append(GateFailure("commit_batch", snapshot.tick, "empty_path"))
            continue
        try:
            local_agent = problem.global_agent_ids.index(candidate.agent_id)
            local_path = tuple((problem.global_target_ids.index(target), mode) for target, mode in candidate.tasks)
        except ValueError:
            failures.append(GateFailure("commit_batch", snapshot.tick, "locked_or_unknown_target"))
            continue
        expected = _planned_path(problem, local_agent, local_path)
        evaluation = evaluate_mode_path(problem.instance, local_agent, local_path)
        available = snapshot.agents[candidate.agent_id]
        if evaluation.ammo_used > available.available_ammo:
            failures.append(GateFailure("commit_batch", snapshot.tick, "ammo_unavailable"))
        elif evaluation.distance > available.available_distance + TOL:
            failures.append(GateFailure("commit_batch", snapshot.tick, "distance_unavailable"))
        elif candidate.start_ticks != expected.start_ticks or candidate.completion_ticks != expected.completion_ticks:
            failures.append(GateFailure("commit_batch", snapshot.tick, "tick_path_mismatch"))
        elif candidate.tasks != expected.tasks or not isclose(candidate.score, expected.score, rel_tol=0.0, abs_tol=TOL):
            failures.append(GateFailure("commit_batch", snapshot.tick, "path_score_mismatch"))
        elif not evaluation.feasible:
            failures.append(GateFailure("commit_batch", snapshot.tick, "path_infeasible"))
    return tuple(failures)
