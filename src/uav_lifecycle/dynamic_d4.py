"""Frozen D4 battlefield-structure and reachability scenario generators."""

from __future__ import annotations

from fractions import Fraction
from math import ceil, cos, log, pi, sqrt

from .dynamic_d3 import D3ScaleCell
from .dynamic_rng import DrawKey, uniform01
from .dynamic_types import (
    DynamicConfig,
    DynamicScenario,
    InternalAgentState,
    PrivateTarget,
    PublicTarget,
    ResourceLedger,
    quantize_tick,
)


_EXPERIMENT = "dynamic-lifecycle-mainline-v2"
_RNG = "sha256-u64-v1"
_CELL = D3ScaleCell(6, 20)
BATTLEFIELD_STRUCTURES = ("uniform", "clustered", "mixed", "value_correlated")
WRECK_RATES = (0.0, 0.2, 0.4, 0.6)
REACHABILITY_SCALES = (0.75, 1.0, 1.25)


def _key(version: str, cell_id: str, seed: int, namespace: str, entity: int, event: str, subdraw: int) -> DrawKey:
    return DrawKey(_RNG, _EXPERIMENT, version, cell_id, seed, namespace, entity, event, 0, subdraw)


def _u(version: str, cell_id: str, seed: int, namespace: str, entity: int, event: str, subdraw: int = 0) -> float:
    return float(uniform01(_key(version, cell_id, seed, namespace, entity, event, subdraw)))


def _uniform_position(version: str, cell_id: str, seed: int, namespace: str, entity: int, half: float) -> tuple[float, float]:
    return tuple(
        float(Fraction.from_float(2.0 * half) * uniform01(
            _key(version, cell_id, seed, namespace, entity, "uniform_position", axis)
        ) - Fraction.from_float(half))
        for axis in (0, 1)
    )  # type: ignore[return-value]


def _reflect(value: float, half: float) -> float:
    while value < -half or value > half:
        value = -2.0 * half - value if value < -half else 2.0 * half - value
    return value


def _cluster_position(version: str, cell_id: str, seed: int, entity: int, half: float) -> tuple[float, float]:
    cluster = int(2.0 * _u(version, cell_id, seed, "target", entity, "cluster_index"))
    centers = ((-0.35 * half, -0.10 * half), (0.35 * half, 0.10 * half))
    u1 = max(_u(version, cell_id, seed, "target", entity, "cluster_normal", 0), 1e-15)
    u2 = _u(version, cell_id, seed, "target", entity, "cluster_normal", 1)
    radius = sqrt(-2.0 * log(u1))
    z = (radius * cos(2.0 * pi * u2), radius * cos(2.0 * pi * (u2 + 0.25)))
    sigma = 0.16 * half
    return tuple(_reflect(centers[cluster][axis] + sigma * z[axis], half) for axis in (0, 1))  # type: ignore[return-value]


def _joint_belief(p_high: float, wreck_rate: float) -> tuple[float, float, float, float]:
    alive = 1.0 - wreck_rate
    return p_high * alive, p_high * wreck_rate, (1.0 - p_high) * alive, (1.0 - p_high) * wreck_rate


def _target(
    version: str,
    cell_id: str,
    seed: int,
    target_id: int,
    half: float,
    structure: str,
    wreck_rate: float,
) -> tuple[PublicTarget, PrivateTarget]:
    clustered = structure == "clustered"
    if structure == "mixed":
        clustered = _u(version, cell_id, seed, "target", target_id, "spatial_component") < 0.60
    elif structure == "value_correlated":
        clustered = _u(version, cell_id, seed, "target", target_id, "spatial_component") < 0.50
    position = (
        _cluster_position(version, cell_id, seed, target_id, half)
        if clustered else
        _uniform_position(version, cell_id, seed, "target", target_id, half)
    )
    p_high = 0.65 if structure == "value_correlated" and clustered else 0.15 if structure == "value_correlated" else 0.40
    category = "H" if _u(version, cell_id, seed, "target", target_id, "category_truth") < p_high else "L"
    damage = "D" if _u(version, cell_id, seed, "target", target_id, "damage_truth") < wreck_rate else "A"
    return (
        PublicTarget(target_id, position, _joint_belief(p_high, wreck_rate)),
        PrivateTarget(target_id, category, damage, False),
    )


def _agents(version: str, cell_id: str, seed: int, half: float, ammo: int, max_range: float) -> tuple[InternalAgentState, ...]:
    return tuple(
        InternalAgentState(
            agent_id,
            _uniform_position(version, cell_id, seed, "agent", agent_id, half),
            ResourceLedger(ammo, 0.0, 0.0),
            ResourceLedger(max_range, 0.0, 0.0),
            ammo,
            max_range,
            None,
        )
        for agent_id in range(_CELL.agents)
    )


def generate_battlefield_structure(structure: str, wreck_rate: float, seed: int, config: DynamicConfig) -> DynamicScenario:
    if structure not in BATTLEFIELD_STRUCTURES or wreck_rate not in WRECK_RATES:
        raise ValueError("unregistered D4 battlefield condition")
    if not 8000 <= seed < 8064:
        raise ValueError("formal D4 battlefield seed must lie in [8000, 8064)")
    version, cell_id = "d4-battlefield-generator-v1", "N6-M20-D4-battlefield"
    pairs = tuple(
        _target(version, cell_id, seed, target_id, _CELL.arena_half_width, structure, wreck_rate)
        for target_id in range(_CELL.targets)
    )
    rate = int(round(100.0 * wreck_rate))
    return DynamicScenario(
        f"D4A-{structure}-D{rate:02d}-{cell_id}-S{seed:04d}", cell_id, seed,
        f"{_EXPERIMENT}/{version}", tuple(item[0] for item in pairs),
        tuple(item[1] for item in pairs),
        _agents(version, cell_id, seed, _CELL.arena_half_width, _CELL.ammo, _CELL.max_range),
        quantize_tick(_CELL.horizon, config.tick_size),
    )


def generate_reachability(map_scale: float, time_scale: float, seed: int, config: DynamicConfig) -> DynamicScenario:
    if map_scale not in REACHABILITY_SCALES or time_scale not in REACHABILITY_SCALES:
        raise ValueError("unregistered D4 reachability condition")
    if not 9000 <= seed < 9064:
        raise ValueError("formal D4 reachability seed must lie in [9000, 9064)")
    version, cell_id = "d4-reachability-generator-v1", "N6-M20-D4-reachability"
    half = _CELL.arena_half_width * map_scale
    horizon = ceil(_CELL.horizon * time_scale)
    max_range = 1.25 * horizon
    pairs = tuple(
        _target(version, cell_id, seed, target_id, half, "uniform", 0.2)
        for target_id in range(_CELL.targets)
    )
    m, t = int(round(100.0 * map_scale)), int(round(100.0 * time_scale))
    return DynamicScenario(
        f"D4B-L{m:03d}-T{t:03d}-{cell_id}-S{seed:04d}", cell_id, seed,
        f"{_EXPERIMENT}/{version}", tuple(item[0] for item in pairs),
        tuple(item[1] for item in pairs), _agents(version, cell_id, seed, half, _CELL.ammo, max_range),
        quantize_tick(horizon, config.tick_size),
    )


__all__ = [
    "BATTLEFIELD_STRUCTURES", "REACHABILITY_SCALES", "WRECK_RATES",
    "generate_battlefield_structure", "generate_reachability",
]
