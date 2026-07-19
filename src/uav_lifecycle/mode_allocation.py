from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import dist, exp, isfinite

from numpy.typing import ArrayLike

from .dynamic_types import quantize_tick
from .attack import expected_attack_reward
from .rollout import RolloutParameters, action_values, discount


TOL = 1e-12


class Mode(str, Enum):
    RECON = "recon"
    ATTACK = "attack"
    BDA = "bda"


ModeKey = tuple[int, Mode]


@dataclass(frozen=True)
class ModeTask:
    target_id: int
    mode: Mode
    position: tuple[float, float]
    duration: float
    ammo: int
    utility: float

    def __post_init__(self) -> None:
        if self.target_id < 0 or self.duration < 0 or self.ammo < 0:
            raise ValueError("target id, duration, and ammo must be nonnegative")
        if not all(isfinite(value) for value in (*self.position, self.duration, self.utility)):
            raise ValueError("task values must be finite")


@dataclass(frozen=True)
class ModeAgent:
    agent_id: int
    origin: tuple[float, float]
    horizon: float
    distance_budget: float
    ammo: int

    def __post_init__(self) -> None:
        if self.agent_id < 0 or self.horizon < 0 or self.distance_budget < 0 or self.ammo < 0:
            raise ValueError("agent resources must be nonnegative")


@dataclass(frozen=True)
class TickPathTiming:
    """Optional absolute-tick contract for dynamic planning epochs."""

    epoch_tick: int
    max_tick: int
    tick_size: float
    speed: float

    def __post_init__(self) -> None:
        if self.epoch_tick < 0 or self.max_tick < 0 or self.epoch_tick > self.max_tick:
            raise ValueError("tick horizon must be ordered and nonnegative")
        if not isfinite(self.tick_size) or self.tick_size <= 0:
            raise ValueError("tick_size must be finite and positive")
        if not isfinite(self.speed) or self.speed <= 0:
            raise ValueError("speed must be finite and positive")


@dataclass(frozen=True)
class ModeInstance:
    agents: tuple[ModeAgent, ...]
    tasks_by_target: tuple[tuple[ModeTask, ...], ...]
    beta: float
    distance_cost: float
    ammo_cost: float
    instance_id: str = "mode-instance"
    continuation: str = "optimistic"
    timing: TickPathTiming | None = None

    def __post_init__(self) -> None:
        if tuple(agent.agent_id for agent in self.agents) != tuple(range(len(self.agents))):
            raise ValueError("agent ids must be contiguous")
        if self.beta < 0 or self.distance_cost < 0 or self.ammo_cost < 0:
            raise ValueError("score parameters must be nonnegative")
        for target_id, tasks in enumerate(self.tasks_by_target):
            if not tasks or any(task.target_id != target_id for task in tasks):
                raise ValueError("each target task group must be nonempty and indexed")
            if len({task.mode for task in tasks}) != len(tasks):
                raise ValueError("duplicate mode for target")

    def task(self, key: ModeKey) -> ModeTask:
        target_id, mode = key
        for task in self.tasks_by_target[target_id]:
            if task.mode is mode:
                return task
        raise KeyError(key)


@dataclass(frozen=True)
class ModePathEvaluation:
    score: float
    completion_time: float
    distance: float
    ammo_used: int
    start_times: tuple[float, ...]
    feasible: bool
    start_ticks: tuple[int, ...] | None = None
    completion_tick: int | None = None


@dataclass(frozen=True)
class ModeInsertion:
    position: int
    path: tuple[ModeKey, ...]
    marginal: float
    evaluation: ModePathEvaluation


def evaluate_mode_path(instance: ModeInstance, agent_id: int, path: tuple[ModeKey, ...]) -> ModePathEvaluation:
    agent = instance.agents[agent_id]
    current = agent.origin
    elapsed = 0.0
    travelled = 0.0
    ammo = 0
    reward = 0.0
    starts: list[float] = []
    target_ids: list[int] = []
    timing = instance.timing
    current_tick = timing.epoch_tick if timing is not None else None
    start_ticks: list[int] | None = [] if timing is not None else None
    for key in path:
        task = instance.task(key)
        leg = dist(current, task.position)
        travelled += leg
        if timing is None:
            elapsed += leg
            starts.append(elapsed)
            reward += exp(-instance.beta * elapsed) * task.utility
        else:
            if current_tick is None or start_ticks is None:  # pragma: no cover - dataclass invariant
                raise RuntimeError("dynamic tick state was not initialized")
            current_tick += quantize_tick(leg / timing.speed, timing.tick_size)
            start_ticks.append(current_tick)
            absolute_start = current_tick * timing.tick_size
            starts.append(absolute_start)
            reward += exp(-instance.beta * absolute_start) * task.utility
        ammo += task.ammo
        if timing is None:
            elapsed += task.duration
        else:
            current_tick += quantize_tick(task.duration, timing.tick_size)
            elapsed = current_tick * timing.tick_size
        current = task.position
        target_ids.append(task.target_id)
    score = reward - instance.distance_cost * travelled - instance.ammo_cost * ammo
    feasible = (
        len(target_ids) == len(set(target_ids))
        and (
            elapsed <= agent.horizon + TOL
            if timing is None
            else current_tick is not None and current_tick <= timing.max_tick
        )
        and travelled <= agent.distance_budget + TOL
        and ammo <= agent.ammo
    )
    return ModePathEvaluation(
        score,
        elapsed,
        travelled,
        ammo,
        tuple(starts),
        feasible,
        None if start_ticks is None else tuple(start_ticks),
        current_tick,
    )


def best_mode_insertion(instance: ModeInstance, agent_id: int, path: tuple[ModeKey, ...], key: ModeKey) -> ModeInsertion | None:
    if key[0] in {existing[0] for existing in path}:
        return None
    base = evaluate_mode_path(instance, agent_id, path)
    best = None
    for position in range(len(path) + 1):
        candidate_path = path[:position] + (key,) + path[position:]
        evaluation = evaluate_mode_path(instance, agent_id, candidate_path)
        if not evaluation.feasible:
            continue
        candidate = ModeInsertion(position, candidate_path, evaluation.score - base.score, evaluation)
        if best is None or candidate.marginal > best.marginal + TOL:
            best = candidate
    return best


def mode_utilities(
    belief: ArrayLike,
    recon_observation_kernel: ArrayLike,
    bda_observation_kernel: ArrayLike,
    params: RolloutParameters,
    variant: str,
    gates: dict[Mode, bool] | None = None,
) -> dict[Mode, float]:
    optimistic_raw = action_values(belief, recon_observation_kernel, bda_observation_kernel, params)
    optimistic = {mode: optimistic_raw[mode.value] for mode in Mode}
    immediate_attack = -params.cost_a + discount(params.duration_a, params.beta) * expected_attack_reward(
        belief, params.pi_h, params.pi_l, params.value_h, params.value_l
    )
    no_cont = {Mode.RECON: -params.cost_r, Mode.ATTACK: immediate_attack, Mode.BDA: -params.cost_b}
    if variant == "optimistic":
        return optimistic
    if variant == "no_continuation":
        return no_cont
    if variant == "ammo_reachability_gate":
        if gates is None:
            raise ValueError("ammo_reachability_gate requires per-mode gates")
        return {
            mode: no_cont[mode] + (optimistic[mode] - no_cont[mode]) * bool(gates.get(mode, False))
            for mode in Mode
        }
    raise ValueError(f"unknown continuation variant: {variant}")
