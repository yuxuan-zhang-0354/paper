from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import factorial

from .mode_allocation import TOL, Mode, ModeInstance, ModeKey, evaluate_mode_path


_MODE_RANK = {Mode.RECON: 0, Mode.ATTACK: 1, Mode.BDA: 2}


@dataclass(frozen=True)
class ExactSearchAudit:
    """Observable exact-search work under a deterministic counting contract.

    Candidate evaluations count every non-empty ordered mode path passed to
    ``evaluate_mode_path``, including infeasible paths. DP transitions count
    every disjoint `(assigned_mask, agent_mask)` combination scored.
    """

    candidate_path_evaluations: tuple[int, ...]
    dp_transitions: int
    profile_equivalent_count: int
    solution_key: tuple[tuple[tuple[int, int], ...], ...]

    @property
    def total_count(self) -> int:
        return sum(self.candidate_path_evaluations) + self.dp_transitions


class ExactSearchBoundExceeded(RuntimeError):
    def __init__(self, actual_count: int, operation_cap: int) -> None:
        super().__init__(f"exact search operation count {actual_count} exceeds cap {operation_cap}")
        self.actual_count = actual_count
        self.operation_cap = operation_cap


@dataclass
class _SearchCounter:
    per_agent: list[int]
    dp_transitions: int = 0

    @property
    def total(self) -> int:
        return sum(self.per_agent) + self.dp_transitions

@dataclass(frozen=True)
class ModeExactResult:
    score: float
    paths: tuple[tuple[ModeKey, ...], ...]
    target_modes: tuple[Mode | None, ...]
    assigned_mask: int
    audit: ExactSearchAudit


def _path_key(path: tuple[ModeKey, ...]) -> tuple[tuple[int, int], ...]:
    return tuple((target_id, _MODE_RANK[mode]) for target_id, mode in path)


def profile_equivalent_count(agent_count: int, target_count: int) -> int:
    """Count the exact deterministic profile space registered by CEX.

    Each target independently chooses Defer or one of three modes for one of
    ``agent_count`` agents. For each such assignment, agent ``i`` can order its
    ``k_i`` assigned targets in ``k_i!`` ways, so that assignment contributes
    ``product(k_i!)`` path profiles. The count is independent of feasibility
    pruning and solver implementation stages. It is at most
    ``target_count! * (1 + 3 * agent_count) ** target_count`` because there are
    ``(1 + 3N)^M`` assignments and every product of disjoint group factorials
    is at most ``M!``. The empty-target profile is counted once.
    """

    if isinstance(agent_count, bool) or not isinstance(agent_count, int) or agent_count < 0:
        raise ValueError("agent_count must be a nonnegative integer")
    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 0:
        raise ValueError("target_count must be a nonnegative integer")
    choice_count = 1 + 3 * agent_count
    total = 0
    for assignment in product(range(choice_count), repeat=target_count):
        counts = [0] * agent_count
        for choice in assignment:
            if choice:
                counts[(choice - 1) // 3] += 1
        contribution = 1
        for count in counts:
            contribution *= factorial(count)
        total += contribution
    return total


def best_mode_paths_by_target_mask(
    instance: ModeInstance,
    agent_id: int,
    allowed_modes: dict[int, Mode] | None = None,
    *,
    _counter: _SearchCounter | None = None,
) -> dict[int, tuple[float, tuple[ModeKey, ...]]]:
    """Return the best feasible ordered path for every target subset."""
    target_count = len(instance.tasks_by_target)
    best: dict[int, tuple[float, tuple[ModeKey, ...]]] = {0: (0.0, ())}
    targets = range(target_count)
    for length in range(1, target_count + 1):
        for ordered_targets in permutations(targets, length):
            choices: list[tuple[Mode, ...]] = []
            for target_id in ordered_targets:
                if allowed_modes is None:
                    choices.append(tuple(task.mode for task in instance.tasks_by_target[target_id]))
                elif target_id in allowed_modes:
                    choices.append((allowed_modes[target_id],))
                else:
                    choices = []
                    break
            if not choices:
                continue
            for modes in product(*choices):
                path = tuple(zip(ordered_targets, modes, strict=True))
                if _counter is not None:
                    _counter.per_agent[agent_id] += 1
                evaluation = evaluate_mode_path(instance, agent_id, path)
                if not evaluation.feasible:
                    continue
                mask = sum(1 << target_id for target_id in ordered_targets)
                current = best.get(mask)
                if current is None or evaluation.score > current[0] + TOL or (
                    abs(evaluation.score - current[0]) <= TOL
                    and _path_key(path) < _path_key(current[1])
                ):
                    best[mask] = (evaluation.score, path)
    return best


def _solution_key(paths: tuple[tuple[ModeKey, ...], ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(_path_key(path) for path in paths)


def _solve(
    instance: ModeInstance,
    allowed_modes: dict[int, Mode] | None,
    operation_cap: int | None,
) -> ModeExactResult:
    if operation_cap is not None and operation_cap < 0:
        raise ValueError("operation_cap must be nonnegative")
    profile_count = profile_equivalent_count(
        len(instance.agents), len(instance.tasks_by_target)
    )
    if operation_cap is not None and profile_count > operation_cap:
        raise ExactSearchBoundExceeded(profile_count, operation_cap)
    counter = _SearchCounter([0] * len(instance.agents))
    # DP state: assigned target mask -> (score, paths for agents processed so far).
    dp: dict[int, tuple[float, tuple[tuple[ModeKey, ...], ...]]] = {0: (0.0, ())}
    for agent_id in range(len(instance.agents)):
        per_agent = best_mode_paths_by_target_mask(
            instance,
            agent_id,
            allowed_modes,
            _counter=counter,
        )
        updated: dict[int, tuple[float, tuple[tuple[ModeKey, ...], ...]]] = {}
        for assigned, (base_score, base_paths) in dp.items():
            for agent_mask, (agent_score, agent_path) in per_agent.items():
                if assigned & agent_mask:
                    continue
                counter.dp_transitions += 1
                mask = assigned | agent_mask
                candidate = (base_score + agent_score, base_paths + (agent_path,))
                current = updated.get(mask)
                if current is None or candidate[0] > current[0] + TOL or (
                    abs(candidate[0] - current[0]) <= TOL
                    and _solution_key(candidate[1]) < _solution_key(current[1])
                ):
                    updated[mask] = candidate
        dp = updated

    assigned_mask, (score, paths) = min(
        dp.items(),
        key=lambda item: (-item[1][0], _solution_key(item[1][1])),
    )
    target_modes: list[Mode | None] = [None] * len(instance.tasks_by_target)
    for path in paths:
        for target_id, mode in path:
            target_modes[target_id] = mode
    audit = ExactSearchAudit(
        tuple(counter.per_agent), counter.dp_transitions, profile_count, _solution_key(paths)
    )
    return ModeExactResult(score, paths, tuple(target_modes), assigned_mask, audit)


def validate_exact_audit(result: ModeExactResult, operation_cap: int) -> tuple[str, ...]:
    """Pure structural audit for injected faults and adapter validation."""

    failures: list[str] = []
    if result.audit.profile_equivalent_count > operation_cap:
        failures.append("search_bound_exceeded")
    if result.audit.solution_key != _solution_key(result.paths):
        failures.append("tie_key_mismatch")
    return tuple(failures)


def solve_all_mode_exact(
    instance: ModeInstance, operation_cap: int | None = None
) -> ModeExactResult:
    return _solve(instance, None, operation_cap)


def solve_fixed_mode_exact(
    instance: ModeInstance,
    screened: dict[int, Mode],
    operation_cap: int | None = None,
) -> ModeExactResult:
    return _solve(instance, screened, operation_cap)
