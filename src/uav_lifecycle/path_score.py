"""Discounted route scoring and the audited constrained-DMG counterexample."""

from dataclasses import dataclass
from math import dist, isfinite
from typing import Sequence


Point = tuple[float, float]


def _point(value: Sequence[float], name: str) -> Point:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    point = (float(value[0]), float(value[1]))
    if not all(isfinite(coordinate) for coordinate in point):
        raise ValueError(f"{name} coordinates must be finite")
    return point


def _nonnegative(value: float, name: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    location: Point
    service_time: float
    value: float

    def __post_init__(self) -> None:
        if not str(self.task_id):
            raise ValueError("task_id must be nonempty")
        object.__setattr__(self, "task_id", str(self.task_id))
        object.__setattr__(self, "location", _point(self.location, "location"))
        object.__setattr__(
            self,
            "service_time",
            _nonnegative(self.service_time, "service_time"),
        )
        object.__setattr__(self, "value", _nonnegative(self.value, "value"))


@dataclass(frozen=True, slots=True)
class PathEvaluation:
    score: float
    completion_time: float
    start_times: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class InsertionResult:
    insertion_index: int
    path: tuple[Task, ...]
    evaluation: PathEvaluation
    marginal_gain: float


def evaluate_path(
    path: Sequence[Task],
    discount_base: float,
    origin: Sequence[float] = (0.0, 0.0),
) -> PathEvaluation:
    """Score a route at service start times and return final completion time."""

    base = float(discount_base)
    if not isfinite(base) or not 0.0 < base <= 1.0:
        raise ValueError("discount_base must lie in (0, 1]")
    position = _point(origin, "origin")
    time = 0.0
    score = 0.0
    start_times: list[float] = []
    for task in path:
        time += dist(position, task.location)
        position = task.location
        start_times.append(time)
        score += base**time * task.value
        time += task.service_time
    return PathEvaluation(float(score), float(time), tuple(start_times))


def all_insertions(
    path: Sequence[Task], task: Task
) -> tuple[tuple[Task, ...], ...]:
    """Insert *task* at every position while preserving existing order."""

    frozen_path = tuple(path)
    return tuple(
        frozen_path[:index] + (task,) + frozen_path[index:]
        for index in range(len(frozen_path) + 1)
    )


def best_insertion(
    path: Sequence[Task],
    task: Task,
    discount_base: float,
    deadline: float | None = None,
    origin: Sequence[float] = (0.0, 0.0),
) -> InsertionResult | None:
    """Return the maximum marginal insertion, optionally filtering a deadline."""

    deadline_value = (
        None if deadline is None else _nonnegative(deadline, "deadline")
    )
    frozen_path = tuple(path)
    baseline = evaluate_path(frozen_path, discount_base, origin)
    best: InsertionResult | None = None
    for index, candidate in enumerate(all_insertions(frozen_path, task)):
        evaluation = evaluate_path(candidate, discount_base, origin)
        if (
            deadline_value is not None
            and evaluation.completion_time > deadline_value
        ):
            continue
        result = InsertionResult(
            insertion_index=index,
            path=candidate,
            evaluation=evaluation,
            marginal_gain=float(evaluation.score - baseline.score),
        )
        if best is None or result.marginal_gain > best.marginal_gain:
            best = result
    return best


def reproduce_deadline_counterexample() -> dict[str, object]:
    """Reproduce the fixed raw-DMG/constrained-DMG numerical witness."""

    task_a = Task("A", (-7.0, -7.0), service_time=1.0, value=100.0)
    task_k = Task("K", (5.0, -2.0), service_time=0.0, value=50.0)
    task_j = Task("J", (8.0, -2.0), service_time=3.0, value=40.0)
    small_path = (task_a,)
    large_path = (task_k, task_a)
    discount_base = 0.98
    deadline = 29.0

    raw_small = best_insertion(small_path, task_j, discount_base)
    raw_large = best_insertion(large_path, task_j, discount_base)
    feasible_small = best_insertion(
        small_path, task_j, discount_base, deadline=deadline
    )
    feasible_large = best_insertion(
        large_path, task_j, discount_base, deadline=deadline
    )
    if any(
        result is None
        for result in (raw_small, raw_large, feasible_small, feasible_large)
    ):
        raise RuntimeError("audited counterexample unexpectedly has no insertion")
    assert raw_small is not None
    assert raw_large is not None
    assert feasible_small is not None
    assert feasible_large is not None

    raw_small_gain = raw_small.marginal_gain
    raw_large_gain = raw_large.marginal_gain
    feasible_small_gain = feasible_small.marginal_gain
    feasible_large_gain = feasible_large.marginal_gain
    return {
        "raw_small": raw_small_gain,
        "raw_large": raw_large_gain,
        "feasible_small": feasible_small_gain,
        "feasible_large": feasible_large_gain,
        "raw_dmg_holds": bool(raw_small_gain >= raw_large_gain),
        "constrained_dmg_violated": bool(
            feasible_small_gain < feasible_large_gain
        ),
        "details": {
            "raw_small_insertion_index": raw_small.insertion_index,
            "raw_large_insertion_index": raw_large.insertion_index,
            "feasible_small_insertion_index": feasible_small.insertion_index,
            "feasible_large_insertion_index": feasible_large.insertion_index,
            "raw_small_completion_time": raw_small.evaluation.completion_time,
            "raw_large_completion_time": raw_large.evaluation.completion_time,
            "feasible_small_completion_time": (
                feasible_small.evaluation.completion_time
            ),
            "feasible_large_completion_time": (
                feasible_large.evaluation.completion_time
            ),
        },
    }
