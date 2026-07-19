from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

from .allocation_model import TOL, StaticInstance, evaluate_agent_path


@dataclass(frozen=True)
class ExactResult:
    score: float
    paths: tuple[tuple[int, ...], ...]
    assigned_mask: int


def best_path_by_subset(instance: StaticInstance, agent_id: int) -> dict[int, tuple[float, tuple[int, ...]]]:
    capacity = instance.agents[agent_id].capacity
    result: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    task_ids = range(len(instance.tasks))
    for size in range(1, min(capacity, len(instance.tasks)) + 1):
        for subset in combinations(task_ids, size):
            mask = sum(1 << j for j in subset)
            best: tuple[float, tuple[int, ...]] | None = None
            for path in permutations(subset):
                evaluation = evaluate_agent_path(instance, agent_id, path)
                if not evaluation.feasible:
                    continue
                candidate = (evaluation.score, path)
                if best is None or candidate[0] > best[0] + TOL or (
                    abs(candidate[0] - best[0]) <= TOL and candidate[1] < best[1]
                ):
                    best = candidate
            if best is not None:
                result[mask] = best
    return result


def solve_exact(instance: StaticInstance) -> ExactResult:
    dp: dict[int, tuple[float, tuple[tuple[int, ...], ...]]] = {0: (0.0, ())}
    for agent_id in range(len(instance.agents)):
        options = best_path_by_subset(instance, agent_id)
        next_dp: dict[int, tuple[float, tuple[tuple[int, ...], ...]]] = {}
        for used_mask, (score, paths) in dp.items():
            for subset_mask, (agent_score, path) in options.items():
                if used_mask & subset_mask:
                    continue
                mask = used_mask | subset_mask
                candidate = (score + agent_score, paths + (path,))
                old = next_dp.get(mask)
                if old is None or candidate[0] > old[0] + TOL or (
                    abs(candidate[0] - old[0]) <= TOL and candidate[1] < old[1]
                ):
                    next_dp[mask] = candidate
        dp = next_dp
    mask, (score, paths) = min(
        dp.items(), key=lambda item: (-item[1][0], item[1][1], item[0])
    )
    return ExactResult(score, paths, mask)
