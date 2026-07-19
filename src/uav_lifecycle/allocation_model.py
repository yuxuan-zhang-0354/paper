from __future__ import annotations

from dataclasses import dataclass
from math import dist, isfinite


TOL = 1e-12


@dataclass(frozen=True)
class StaticTask:
    task_id: int
    position: tuple[float, float]
    service: float
    value: float

    def __post_init__(self) -> None:
        if self.task_id < 0 or self.service < 0 or self.value < 0:
            raise ValueError("task id, service, and value must be nonnegative")
        if not all(isfinite(x) for x in (*self.position, self.service, self.value)):
            raise ValueError("task fields must be finite")


@dataclass(frozen=True)
class StaticAgent:
    agent_id: int
    origin: tuple[float, float]
    deadline: float
    capacity: int

    def __post_init__(self) -> None:
        if self.agent_id < 0 or self.deadline < 0 or self.capacity < 0:
            raise ValueError("agent id, deadline, and capacity must be nonnegative")
        if not all(isfinite(x) for x in (*self.origin, self.deadline)):
            raise ValueError("agent fields must be finite")


@dataclass(frozen=True)
class StaticInstance:
    agents: tuple[StaticAgent, ...]
    tasks: tuple[StaticTask, ...]
    discount: float
    instance_id: str = "instance"

    def __post_init__(self) -> None:
        if not 0 < self.discount <= 1 or not isfinite(self.discount):
            raise ValueError("discount must lie in (0, 1]")
        if tuple(a.agent_id for a in self.agents) != tuple(range(len(self.agents))):
            raise ValueError("agent ids must be contiguous from zero")
        if tuple(t.task_id for t in self.tasks) != tuple(range(len(self.tasks))):
            raise ValueError("task ids must be contiguous from zero")


@dataclass(frozen=True)
class PathEvaluation:
    score: float
    completion_time: float
    start_times: tuple[float, ...]
    feasible: bool


@dataclass(frozen=True)
class Insertion:
    position: int
    path: tuple[int, ...]
    marginal: float
    evaluation: PathEvaluation


def evaluate_agent_path(instance: StaticInstance, agent_id: int, path: tuple[int, ...]) -> PathEvaluation:
    if len(set(path)) != len(path):
        raise ValueError("duplicate task in path")
    agent = instance.agents[agent_id]
    current = agent.origin
    elapsed = 0.0
    score = 0.0
    starts: list[float] = []
    for task_id in path:
        if task_id < 0 or task_id >= len(instance.tasks):
            raise IndexError("task id out of range")
        task = instance.tasks[task_id]
        elapsed += dist(current, task.position)
        starts.append(elapsed)
        score += task.value * instance.discount**elapsed
        elapsed += task.service
        current = task.position
    feasible = len(path) <= agent.capacity and elapsed <= agent.deadline + TOL
    return PathEvaluation(score, elapsed, tuple(starts), feasible)


def best_true_insertion(
    instance: StaticInstance, agent_id: int, path: tuple[int, ...], task_id: int
) -> Insertion | None:
    if task_id in path:
        return None
    base = evaluate_agent_path(instance, agent_id, path)
    best: Insertion | None = None
    for position in range(len(path) + 1):
        candidate = path[:position] + (task_id,) + path[position:]
        evaluation = evaluate_agent_path(instance, agent_id, candidate)
        if not evaluation.feasible:
            continue
        insertion = Insertion(position, candidate, evaluation.score - base.score, evaluation)
        if best is None or insertion.marginal > best.marginal + TOL:
            best = insertion
    return best
