"""Minimal D3 scenario, preference, and environment-model factories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from math import ceil, log, sqrt

from .dynamic_rng import DrawKey, categorical, uniform01
from .dynamic_scenarios import D1Cell, D1_CELLS
from .dynamic_types import (
    DynamicConfig,
    DynamicScenario,
    EnvironmentModel,
    ExperimentalDynamicConfig,
    InternalAgentState,
    PrivateTarget,
    PublicTarget,
    ResourceLedger,
    quantize_tick,
)


_EXPERIMENT = "dynamic-lifecycle-mainline-v2"
_RNG = "sha256-u64-v1"
_ARCHETYPES = (
    (0.00, 0.42, 0.24, 0.34),
    (1.00, 0.00, 0.00, 0.00),
    (0.26, 0.66, 0.00, 0.08),
    (0.00, 0.08, 0.06, 0.86),
)
_TRUTH = (("H", "A"), ("H", "D"), ("L", "A"), ("L", "D"))


@dataclass(frozen=True, slots=True)
class D3ScaleCell:
    agents: int
    targets: int

    @property
    def cell_id(self) -> str:
        return f"N{self.agents}-M{self.targets}-Rscaled"

    @property
    def arena_half_width(self) -> float:
        return 6.0 * sqrt(self.targets / 5.0)

    @property
    def ammo(self) -> int:
        return ceil(1.2 * self.targets / self.agents)

    @property
    def horizon(self) -> int:
        return ceil(12.0 * sqrt(self.targets / 5.0) + 8.0 * self.targets / self.agents)

    @property
    def max_range(self) -> float:
        return 1.25 * self.horizon


D3_SCALE_CELLS = tuple(
    D3ScaleCell(n, m)
    for n, m in ((4, 10), (6, 15), (8, 20), (6, 10), (6, 20), (6, 30), (8, 30))
)


def _key(
    version: str, cell_id: str, index: int, namespace: str,
    entity_id: int, event: str, subdraw: int,
) -> DrawKey:
    return DrawKey(
        _RNG, _EXPERIMENT, version, cell_id, index,
        namespace, entity_id, event, 0, subdraw,
    )


def _position(version: str, cell_id: str, index: int, namespace: str, entity_id: int, half: float) -> tuple[float, float]:
    return tuple(
        float(Fraction.from_float(2.0 * half) * uniform01(
            _key(version, cell_id, index, namespace, entity_id, "initial_position", axis)
        ) - Fraction.from_float(half))
        for axis in (0, 1)
    )  # type: ignore[return-value]


def _continuous_belief(version: str, cell_id: str, index: int, target_id: int) -> tuple[float, float, float, float]:
    values = [
        -log(float(uniform01(_key(
            version, cell_id, index, "target", target_id, "initial_belief", component,
        ))))
        for component in range(4)
    ]
    total = sum(values)
    return tuple(value / total for value in values)  # type: ignore[return-value]


def _scenario(
    *, scenario_id: str, cell_id: str, index: int, version: str,
    agent_count: int, target_count: int, ammo: float, horizon: float,
    max_range: float, half_width: float, continuous: bool,
    config: DynamicConfig,
) -> DynamicScenario:
    targets = []
    private = []
    for target_id in range(target_count):
        belief = (
            _continuous_belief(version, cell_id, index, target_id)
            if continuous else _ARCHETYPES[(target_id + index) % 4]
        )
        targets.append(PublicTarget(
            target_id,
            _position(version, cell_id, index, "target", target_id, half_width),
            belief,
        ))
        truth = _TRUTH[categorical(
            _key(version, cell_id, index, "target", target_id, "initial_truth", 0), belief,
        )]
        private.append(PrivateTarget(target_id, truth[0], truth[1], False))
    agents = tuple(
        InternalAgentState(
            agent_id,
            _position(version, cell_id, index, "agent", agent_id, half_width),
            ResourceLedger(ammo, 0.0, 0.0),
            ResourceLedger(max_range, 0.0, 0.0),
            ammo,
            max_range,
            None,
        )
        for agent_id in range(agent_count)
    )
    return DynamicScenario(
        scenario_id, cell_id, index, f"{_EXPERIMENT}/{version}",
        tuple(targets), tuple(private), agents,
        quantize_tick(horizon, config.tick_size),
    )


def generate_scale(cell: D3ScaleCell, index: int, config: DynamicConfig, *, smoke: bool = False) -> DynamicScenario:
    if cell not in D3_SCALE_CELLS:
        raise ValueError("unregistered D3 scale cell")
    if not smoke and not 2000 <= index < 2096:
        raise ValueError("formal scale index must lie in [2000, 2096)")
    version = "d3-structural-smoke-v1" if smoke else "d3-scale-generator-v1"
    stage = "D3X-scale" if smoke else "D3S"
    return _scenario(
        scenario_id=f"{stage}-{cell.cell_id}-S{index:04d}", cell_id=cell.cell_id,
        index=index, version=version, agent_count=cell.agents, target_count=cell.targets,
        ammo=cell.ammo, horizon=cell.horizon, max_range=cell.max_range,
        half_width=cell.arena_half_width, continuous=False, config=config,
    )


def generate_d5_scale(cell: D3ScaleCell, index: int, config: DynamicConfig) -> DynamicScenario:
    """Fresh registered scale scenarios for the D5 allocator factorial."""

    if cell not in D3_SCALE_CELLS or not 8000 <= index < 8064:
        raise ValueError("formal D5 scale index must lie in [8000, 8064)")
    version = "d5-factorial-scale-v1"
    return _scenario(
        scenario_id=f"D5S-{cell.cell_id}-S{index:04d}", cell_id=cell.cell_id,
        index=index, version=version, agent_count=cell.agents, target_count=cell.targets,
        ammo=cell.ammo, horizon=cell.horizon, max_range=cell.max_range,
        half_width=cell.arena_half_width, continuous=False, config=config,
    )


def _base_cell(cell: D1Cell) -> None:
    if cell not in D1_CELLS:
        raise ValueError("unregistered D3 base cell")


def generate_continuous(cell: D1Cell, index: int, config: DynamicConfig, *, smoke: bool = False) -> DynamicScenario:
    _base_cell(cell)
    if not smoke and not 3000 <= index < 3128:
        raise ValueError("formal continuous index must lie in [3000, 3128)")
    version = "d3-structural-smoke-v1" if smoke else "d3-continuous-generator-v1"
    stage = "D3X-cont" if smoke else "D3C"
    return _scenario(
        scenario_id=f"{stage}-{cell.cell_id}-S{index:04d}", cell_id=cell.cell_id,
        index=index, version=version, agent_count=cell.agent_count,
        target_count=cell.target_count, ammo=cell.ammo_per_agent,
        horizon=cell.t_max, max_range=cell.range_per_agent, half_width=6.0,
        continuous=True, config=config,
    )


def generate_mismatch(
    cell: D1Cell, condition: str, index: int, config: DynamicConfig, *, smoke: bool = False,
) -> DynamicScenario:
    _base_cell(cell)
    if condition not in MISMATCH_CONDITIONS:
        raise ValueError("unknown mismatch condition")
    if not smoke and not 4000 <= index < 4064:
        raise ValueError("formal mismatch index must lie in [4000, 4064)")
    version = "d3-structural-smoke-v1" if smoke else "d3-mismatch-generator-v1"
    stage = "D3X-mismatch" if smoke else "D3M"
    return _scenario(
        scenario_id=f"{stage}-{condition}-{cell.cell_id}-S{index:04d}",
        cell_id=cell.cell_id, index=index, version=version,
        agent_count=cell.agent_count, target_count=cell.target_count,
        ammo=cell.ammo_per_agent, horizon=cell.t_max,
        max_range=cell.range_per_agent, half_width=6.0, continuous=False, config=config,
    )


def generate_weight(
    cell: D1Cell, profile: str, index: int, config: DynamicConfig, *, smoke: bool = False,
) -> DynamicScenario:
    _base_cell(cell)
    if profile not in UTILITY_PROFILES:
        raise ValueError("unknown utility profile")
    if not smoke and not 5000 <= index < 5064:
        raise ValueError("formal weight index must lie in [5000, 5064)")
    version = "d3-structural-smoke-v1" if smoke else "d3-weight-generator-v1"
    stage = "D3X-weight" if smoke else "D3W"
    return _scenario(
        scenario_id=f"{stage}-{profile}-{cell.cell_id}-S{index:04d}",
        cell_id=cell.cell_id, index=index, version=version,
        agent_count=cell.agent_count, target_count=cell.target_count,
        ammo=cell.ammo_per_agent, horizon=cell.t_max,
        max_range=cell.range_per_agent, half_width=6.0, continuous=False, config=config,
    )


MISMATCH_CONDITIONS = (
    "nominal", "sensor_m20", "sensor_m10", "sensor_p10", "sensor_p20",
    "attack_m20", "attack_m10", "attack_p10", "attack_p20",
)
UTILITY_PROFILES = ("value_priority", "balanced", "resource_saving")
ALLOCATION_PRESSURE_CONDITIONS = (
    "reference", "shared_high_value", "target_clustered",
    "tight_resources", "long_routes", "combined_stress",
)


def generate_cbba_isolation(
    cell: D1Cell, index: int, config: DynamicConfig, *, smoke: bool = False,
) -> DynamicScenario:
    _base_cell(cell)
    if not smoke and not 6000 <= index < 6064:
        raise ValueError("formal CBBA-isolation index must lie in [6000, 6064)")
    version = "d3-structural-smoke-v1" if smoke else "d3-cbba-isolation-v1"
    stage = "D3X-isolation" if smoke else "D3A"
    return _scenario(
        scenario_id=f"{stage}-{cell.cell_id}-S{index:04d}", cell_id=cell.cell_id,
        index=index, version=version, agent_count=cell.agent_count,
        target_count=cell.target_count, ammo=cell.ammo_per_agent,
        horizon=cell.t_max, max_range=cell.range_per_agent, half_width=6.0,
        continuous=False, config=config,
    )


def _pressure_position(
    version: str, index: int, namespace: str, entity_id: int,
    half_width: float, condition: str,
) -> tuple[float, float]:
    cell_id = "N6-M15-allocation-pressure"
    base = _position(version, cell_id, index, namespace, entity_id, half_width)
    if condition == "shared_high_value" and namespace == "agent":
        return base[0] / half_width, base[1] / half_width
    if condition in {"target_clustered", "combined_stress"} and namespace == "target":
        center = -0.45 * half_width if entity_id % 2 == 0 else 0.45 * half_width
        return center + 0.12 * base[0], 0.12 * base[1]
    if condition == "combined_stress" and namespace == "agent":
        return 0.15 * base[0], 0.15 * base[1]
    return base


def generate_allocation_pressure(
    condition: str, index: int, config: DynamicConfig, *, smoke: bool = False,
    pilot: bool = False,
) -> DynamicScenario:
    if condition not in ALLOCATION_PRESSURE_CONDITIONS:
        raise ValueError("unknown allocation-pressure condition")
    if smoke and pilot:
        raise ValueError("smoke and pilot are mutually exclusive")
    if pilot and not 6500 <= index < 6564:
        raise ValueError("pilot allocation-pressure index must lie in [6500, 6564)")
    if not smoke and not pilot and not 7000 <= index < 7064:
        raise ValueError("formal allocation-pressure index must lie in [7000, 7064)")
    version = (
        "d3-structural-smoke-v1" if smoke else
        "d3-allocation-pressure-pilot-v1" if pilot else
        "d3-allocation-pressure-v1"
    )
    stage = "D3X-pressure" if smoke else "D3Q" if pilot else "D3P"
    cell_id = "N6-M15-allocation-pressure"
    base = D3ScaleCell(6, 15)
    half_width = base.arena_half_width * (1.5 if condition == "long_routes" else 1.0)
    tight = condition in {"tight_resources", "combined_stress"}
    ammo = 2 if tight else base.ammo
    horizon = 31 if tight else base.horizon
    max_range = 35.0 if tight else (1.1 * base.horizon if condition == "long_routes" else base.max_range)
    high_belief = (0.72, 0.08, 0.18, 0.02)

    targets: list[PublicTarget] = []
    private: list[PrivateTarget] = []
    for target_id in range(15):
        belief = high_belief if condition in {"shared_high_value", "combined_stress"} else _ARCHETYPES[(target_id + index) % 4]
        targets.append(PublicTarget(
            target_id,
            _pressure_position(version, index, "target", target_id, half_width, condition),
            belief,
        ))
        truth = _TRUTH[categorical(
            _key(version, cell_id, index, "target", target_id, "initial_truth", 0), belief,
        )]
        private.append(PrivateTarget(target_id, truth[0], truth[1], False))
    agents = tuple(
        InternalAgentState(
            agent_id,
            _pressure_position(version, index, "agent", agent_id, half_width, condition),
            ResourceLedger(ammo, 0.0, 0.0), ResourceLedger(max_range, 0.0, 0.0),
            ammo, max_range, None,
        )
        for agent_id in range(6)
    )
    return DynamicScenario(
        f"{stage}-{condition}-{cell_id}-S{index:04d}", cell_id, index,
        f"{_EXPERIMENT}/{version}", tuple(targets), tuple(private), agents,
        quantize_tick(horizon, config.tick_size),
    )


def generate_d5_allocation_pressure(
    condition: str, index: int, config: DynamicConfig,
) -> DynamicScenario:
    """Fresh registered pressure scenarios for the D5 allocator factorial."""

    if condition not in ALLOCATION_PRESSURE_CONDITIONS or not 9000 <= index < 9128:
        raise ValueError("formal D5 pressure index must lie in [9000, 9128)")
    version = "d5-factorial-pressure-v1"
    cell_id = "N6-M15-allocation-pressure"
    base = D3ScaleCell(6, 15)
    half_width = base.arena_half_width * (1.5 if condition == "long_routes" else 1.0)
    tight = condition in {"tight_resources", "combined_stress"}
    ammo = 2 if tight else base.ammo
    horizon = 31 if tight else base.horizon
    max_range = 35.0 if tight else (1.1 * base.horizon if condition == "long_routes" else base.max_range)
    high_belief = (0.72, 0.08, 0.18, 0.02)
    targets: list[PublicTarget] = []
    private: list[PrivateTarget] = []
    for target_id in range(15):
        belief = high_belief if condition in {"shared_high_value", "combined_stress"} else _ARCHETYPES[(target_id + index) % 4]
        targets.append(PublicTarget(
            target_id,
            _pressure_position(version, index, "target", target_id, half_width, condition),
            belief,
        ))
        truth = _TRUTH[categorical(
            _key(version, cell_id, index, "target", target_id, "initial_truth", 0), belief,
        )]
        private.append(PrivateTarget(target_id, truth[0], truth[1], False))
    agents = tuple(
        InternalAgentState(
            agent_id,
            _pressure_position(version, index, "agent", agent_id, half_width, condition),
            ResourceLedger(ammo, 0.0, 0.0), ResourceLedger(max_range, 0.0, 0.0),
            ammo, max_range, None,
        )
        for agent_id in range(6)
    )
    return DynamicScenario(
        f"D5P-{condition}-{cell_id}-S{index:04d}", cell_id, index,
        f"{_EXPERIMENT}/{version}", tuple(targets), tuple(private), agents,
        quantize_tick(horizon, config.tick_size),
    )


def utility_config(profile: str) -> ExperimentalDynamicConfig:
    multipliers = {
        "value_priority": (0.5, 0.5, 0.5),
        "balanced": (1.0, 1.0, 1.0),
        "resource_saving": (1.5, 3.0, 4.0),
    }
    if profile not in multipliers:
        raise ValueError("unknown utility profile")
    service, distance, ammo = multipliers[profile]
    values = asdict(DynamicConfig())
    values.update({
        "config_id": f"d3-{profile}",
        "recon_service_cost": values["recon_service_cost"] * service,
        "attack_service_cost": values["attack_service_cost"] * service,
        "bda_service_cost": values["bda_service_cost"] * service,
        "distance_cost_rate": values["distance_cost_rate"] * distance,
        "ammo_cost_rate": values["ammo_cost_rate"] * ammo,
        "experiment_profile": profile,
    })
    return ExperimentalDynamicConfig(**values)


def _sensor_transform(
    matrix: tuple[tuple[float, float], tuple[float, float]], q: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    target = ((1.0, 0.0), (0.0, 1.0)) if q > 0 else ((0.5, 0.5), (0.5, 0.5))
    amount = abs(q)
    return tuple(tuple(
        (1.0 - amount) * matrix[row][column] + amount * target[row][column]
        for column in range(2)
    ) for row in range(2))  # type: ignore[return-value]


def environment_model(condition: str, config: DynamicConfig | None = None) -> EnvironmentModel:
    nominal = DynamicConfig() if config is None else config
    if condition not in MISMATCH_CONDITIONS:
        raise ValueError("unknown mismatch condition")
    p_h, p_l = nominal.attack_success_high, nominal.attack_success_low
    matrices = (
        nominal.recon_category_matrix,
        nominal.recon_damage_matrix,
        nominal.bda_damage_matrix,
    )
    if condition.startswith("sensor_"):
        q = {"m20": -0.2, "m10": -0.1, "p10": 0.1, "p20": 0.2}[condition[-3:]]
        matrices = tuple(_sensor_transform(matrix, q) for matrix in matrices)
    elif condition.startswith("attack_"):
        delta = {"m20": -0.2, "m10": -0.1, "p10": 0.1, "p20": 0.2}[condition[-3:]]
        p_h = min(1.0, max(0.0, p_h * (1.0 + delta)))
        p_l = min(1.0, max(0.0, p_l * (1.0 + delta)))
    return EnvironmentModel(p_h, p_l, matrices[0], matrices[1], matrices[2], condition)


__all__ = [
    "ALLOCATION_PRESSURE_CONDITIONS", "D3ScaleCell", "D3_SCALE_CELLS",
    "MISMATCH_CONDITIONS", "UTILITY_PROFILES", "environment_model",
    "generate_allocation_pressure", "generate_d5_allocation_pressure",
    "generate_d5_scale", "generate_cbba_isolation", "generate_continuous",
    "generate_mismatch", "generate_scale", "generate_weight", "utility_config",
]
