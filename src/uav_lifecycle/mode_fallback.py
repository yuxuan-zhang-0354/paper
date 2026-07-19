from __future__ import annotations

from dataclasses import dataclass

from .mode_allocation import TOL, Mode, ModeInstance, evaluate_mode_path
from .mode_cbba import (
    ModeCBBAResult,
    ModeMethod,
    ScreenedTask,
    run_mode_cbba,
    screen_modes,
    validate_mode_result,
)


EPSILON_SCORE = 1e-12
MODE_RANK = {Mode.RECON: 0, Mode.ATTACK: 1, Mode.BDA: 2}


@dataclass(frozen=True)
class RankedMode:
    target_id: int
    mode: Mode
    witness_agent: int
    witness_value: float


@dataclass(frozen=True)
class FallbackIteration:
    index: int
    pointers: tuple[int, ...]
    screening: tuple[ScreenedTask | None, ...]
    result: ModeCBBAResult
    gate_report: tuple[tuple[str, int], ...]
    legal: bool
    active_orphans: tuple[int, ...]
    assigned_modes: tuple[Mode | None, ...]


@dataclass(frozen=True)
class FallbackResult:
    valid: bool
    iterations: tuple[FallbackIteration, ...]
    selected_iteration: int | None
    base_targets: tuple[int, ...]
    base_orphans: tuple[int, ...]
    fallback_unresolved: tuple[int, ...]
    resolved_targets: tuple[int, ...]
    newly_unassigned: tuple[int, ...]
    search_advances: int
    search_exhausted_targets: tuple[int, ...]
    selected_assigned_modes: tuple[Mode | None, ...]
    selected_switches: tuple[int, ...]
    selected_defers: tuple[int, ...]
    total_johnson_calls: int
    base_gate_failures: tuple[FallbackIteration, ...]
    late_gate_failures: tuple[FallbackIteration, ...]
    base_orphan_rate: float | None
    fallback_unresolved_rate: float | None


def rank_mode_candidates(instance: ModeInstance) -> tuple[tuple[RankedMode, ...], ...]:
    rows: list[tuple[RankedMode, ...]] = []
    for target_id, tasks in enumerate(instance.tasks_by_target):
        row: list[RankedMode] = []
        for task in tasks:
            feasible: list[tuple[float, int]] = []
            for agent_id in range(len(instance.agents)):
                value = evaluate_mode_path(instance, agent_id, ((target_id, task.mode),))
                if value.feasible:
                    feasible.append((value.score, agent_id))
            if feasible:
                witness_value, witness_agent = min(feasible, key=lambda item: (-item[0], item[1]))
                if witness_value > TOL:
                    row.append(RankedMode(target_id, task.mode, witness_agent, witness_value))
        rows.append(tuple(sorted(row, key=lambda item: (-item.witness_value, MODE_RANK[item.mode]))))

    result = tuple(rows)
    frozen = screen_modes(instance)
    rank_zero = tuple(
        None
        if not row
        else ScreenedTask(
            row[0].target_id,
            row[0].mode,
            row[0].witness_agent,
            row[0].witness_value,
        )
        for row in result
    )
    assert tuple(item is not None for item in rank_zero) == tuple(
        item is not None for item in frozen
    )
    assert rank_zero == frozen
    return result


def active_screening(
    rankings: tuple[tuple[RankedMode, ...], ...], pointers: tuple[int, ...]
) -> tuple[ScreenedTask | None, ...]:
    if len(pointers) != len(rankings):
        raise ValueError("pointers must contain one entry per target")

    screened: list[ScreenedTask | None] = []
    for row, pointer in zip(rankings, pointers):
        if pointer < 0 or pointer > len(row):
            raise ValueError("pointer must be between zero and the row length")
        if pointer == len(row):
            screened.append(None)
            continue
        ranked = row[pointer]
        screened.append(
            ScreenedTask(
                ranked.target_id,
                ranked.mode,
                ranked.witness_agent,
                ranked.witness_value,
            )
        )
    return tuple(screened)


def _assigned_modes(
    target_count: int, result: ModeCBBAResult
) -> tuple[Mode | None, ...]:
    assigned: list[Mode | None] = [None] * target_count
    for state in result.states:
        for target_id, mode in state.path:
            assigned[target_id] = mode
    return tuple(assigned)


def _run_one_fixed_johnson(
    instance: ModeInstance,
    rankings: tuple[tuple[RankedMode, ...], ...],
    pointers: tuple[int, ...],
    index: int,
) -> FallbackIteration:
    screening = active_screening(rankings, pointers)
    result = run_mode_cbba(instance, screening, ModeMethod.JOHNSON_WARPED)
    report = validate_mode_result(instance, screening, result)
    gate_report = tuple(sorted(report.items()))
    legal = result.status == "converged" and all(count == 0 for count in report.values())
    return FallbackIteration(
        index=index,
        pointers=pointers,
        screening=screening,
        result=result,
        gate_report=gate_report,
        legal=legal,
        active_orphans=result.orphan_targets if legal else (),
        assigned_modes=_assigned_modes(len(rankings), result),
    )


def _canonical_assigned_vector(
    iteration: FallbackIteration, base_targets: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(
        len(MODE_RANK) if iteration.assigned_modes[target] is None else MODE_RANK[iteration.assigned_modes[target]]
        for target in base_targets
    )


def select_fallback_iteration(
    iterations: tuple[FallbackIteration, ...],
    base_targets: tuple[int, ...],
    epsilon_score: float = EPSILON_SCORE,
) -> FallbackIteration:
    if not iterations:
        raise ValueError("at least one legal iteration is required")
    if epsilon_score < 0:
        raise ValueError("epsilon_score must be nonnegative")
    jmax = max(item.result.true_score for item in iterations)
    near = [
        item
        for item in iterations
        if item.result.true_score >= jmax - epsilon_score
    ]
    return min(
        near,
        key=lambda item: (
            sum(item.assigned_modes[target] is None for target in base_targets),
            _canonical_assigned_vector(item, base_targets),
            item.index,
        ),
    )


def _invalid_fallback_result(
    target_count: int,
    attempted: FallbackIteration,
) -> FallbackResult:
    return FallbackResult(
        valid=False,
        iterations=(),
        selected_iteration=None,
        base_targets=(),
        base_orphans=(),
        fallback_unresolved=(),
        resolved_targets=(),
        newly_unassigned=(),
        search_advances=0,
        search_exhausted_targets=(),
        selected_assigned_modes=(None,) * target_count,
        selected_switches=(),
        selected_defers=(),
        total_johnson_calls=1,
        base_gate_failures=(attempted,),
        late_gate_failures=(),
        base_orphan_rate=None,
        fallback_unresolved_rate=None,
    )


def finalize_fallback_attempts(
    attempts: tuple[FallbackIteration, ...],
    rankings: tuple[tuple[RankedMode, ...], ...],
) -> FallbackResult:
    if not attempts:
        raise ValueError("at least one fallback attempt is required")
    if any(len(item.pointers) != len(rankings) for item in attempts):
        raise ValueError("every attempt must contain one pointer per target")

    legal_iterations: list[FallbackIteration] = []
    late_gate_failures: list[FallbackIteration] = []
    consumed: list[FallbackIteration] = []
    for attempt in attempts:
        consumed.append(attempt)
        if not attempt.legal:
            if not legal_iterations:
                return _invalid_fallback_result(len(rankings), attempt)
            late_gate_failures.append(attempt)
            break
        legal_iterations.append(attempt)

    search_advances = sum(
        sum(right.pointers) - sum(left.pointers)
        for left, right in zip(consumed, consumed[1:])
    )
    exhausted: set[int] = set()
    for left, right in zip(consumed, consumed[1:]):
        exhausted.update(
            target
            for target, pointer in enumerate(right.pointers)
            if pointer == len(rankings[target]) and pointer > left.pointers[target]
        )

    base_targets = tuple(
        target
        for target, screened in enumerate(legal_iterations[0].screening)
        if screened is not None
    )
    base_orphans = legal_iterations[0].active_orphans
    selected = select_fallback_iteration(tuple(legal_iterations), base_targets)
    fallback_unresolved = tuple(
        target for target in base_targets if selected.assigned_modes[target] is None
    )
    resolved_targets = tuple(
        target for target in base_orphans if target not in fallback_unresolved
    )
    newly_unassigned = tuple(
        target for target in fallback_unresolved if target not in base_orphans
    )
    selected_switches = tuple(
        target
        for target in base_targets
        if selected.assigned_modes[target] is not None
        and rankings[target]
        and selected.assigned_modes[target] is not rankings[target][0].mode
    )
    base_iteration = legal_iterations[0]
    selected_defers = tuple(
        target
        for target in base_targets
        if base_iteration.assigned_modes[target] is not None
        and selected.assigned_modes[target] is None
    )
    denominator = len(base_targets)
    unresolved_rate = (
        len(fallback_unresolved) / denominator if denominator else 0.0
    )

    return FallbackResult(
        valid=True,
        iterations=tuple(legal_iterations),
        selected_iteration=selected.index,
        base_targets=base_targets,
        base_orphans=base_orphans,
        fallback_unresolved=fallback_unresolved,
        resolved_targets=resolved_targets,
        newly_unassigned=newly_unassigned,
        search_advances=search_advances,
        search_exhausted_targets=tuple(sorted(exhausted)),
        selected_assigned_modes=selected.assigned_modes,
        selected_switches=selected_switches,
        selected_defers=selected_defers,
        total_johnson_calls=len(consumed),
        base_gate_failures=(),
        late_gate_failures=tuple(late_gate_failures),
        base_orphan_rate=(len(base_orphans) / denominator if denominator else 0.0),
        fallback_unresolved_rate=unresolved_rate,
    )


def run_ranked_fallback(instance: ModeInstance) -> FallbackResult:
    rankings = rank_mode_candidates(instance)
    pointers = tuple(0 for _ in rankings)
    attempts: list[FallbackIteration] = []
    call_bound = 1 + sum(len(row) for row in rankings)

    while True:
        iteration = _run_one_fixed_johnson(
            instance, rankings, pointers, len(attempts)
        )
        attempts.append(iteration)
        assert len(attempts) <= call_bound
        if not iteration.legal or not iteration.active_orphans:
            break

        next_pointers = tuple(
            pointer + 1 if target in iteration.active_orphans else pointer
            for target, pointer in enumerate(pointers)
        )
        assert sum(next_pointers) > sum(pointers)
        pointers = next_pointers

    return finalize_fallback_attempts(tuple(attempts), rankings)
