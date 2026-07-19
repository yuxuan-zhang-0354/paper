from dataclasses import asdict

import pytest

from uav_lifecycle.dynamic_d3 import (
    ALLOCATION_PRESSURE_CONDITIONS,
    D3_SCALE_CELLS,
    MISMATCH_CONDITIONS,
    UTILITY_PROFILES,
    environment_model,
    generate_allocation_pressure,
    generate_cbba_isolation,
    generate_continuous,
    generate_mismatch,
    generate_scale,
    generate_weight,
    utility_config,
)
from uav_lifecycle.dynamic_scenarios import D1_CELLS
from uav_lifecycle.dynamic_types import DynamicConfig, EnvironmentModel, ExperimentalDynamicConfig


def _initial_payload(scenario):
    return scenario.cell_id, scenario.seed, scenario.targets, scenario.private_targets, scenario.agents


def test_d3_scale_and_continuous_generators_are_valid_and_new() -> None:
    nominal = DynamicConfig()
    scale = generate_scale(D3_SCALE_CELLS[-1], 2000, nominal)
    continuous = generate_continuous(D1_CELLS[0], 3000, nominal)
    assert scale.scenario_id.startswith("D3S-") and len(scale.targets) == 30 and len(scale.agents) == 8
    assert continuous.scenario_id.startswith("D3C-")
    assert all(0 < value < 1 for target in continuous.targets for value in target.belief)
    assert all(sum(target.belief) == pytest.approx(1.0) for target in continuous.targets)


def test_mismatch_conditions_share_initial_crn_and_only_environment_changes() -> None:
    nominal = DynamicConfig()
    scenarios = [generate_mismatch(D1_CELLS[0], condition, 4000, nominal) for condition in MISMATCH_CONDITIONS]
    assert len({scenario.scenario_id for scenario in scenarios}) == len(MISMATCH_CONDITIONS)
    assert all(_initial_payload(scenario) == _initial_payload(scenarios[0]) for scenario in scenarios[1:])
    assert all(scenario.crn_namespace == scenarios[0].crn_namespace for scenario in scenarios)
    assert environment_model("sensor_m20").recon_category_matrix != environment_model("nominal").recon_category_matrix
    assert environment_model("attack_p20").attack_success_high == pytest.approx(0.48)


def test_weight_profiles_share_physics_and_initial_crn() -> None:
    scenarios = [
        generate_weight(D1_CELLS[0], profile, 5000, utility_config(profile))
        for profile in UTILITY_PROFILES
    ]
    assert all(_initial_payload(scenario) == _initial_payload(scenarios[0]) for scenario in scenarios[1:])
    configs = [utility_config(profile) for profile in UTILITY_PROFILES]
    assert all(isinstance(config, ExperimentalDynamicConfig) for config in configs)
    assert [config.ammo_cost_rate for config in configs] == [0.25, 0.5, 2.0]


def test_environment_model_validation_and_nominal_projection() -> None:
    config = DynamicConfig()
    assert asdict(EnvironmentModel.from_config(config)) == asdict(environment_model("nominal"))
    with pytest.raises(ValueError):
        EnvironmentModel(1.2, 0.5, config.recon_category_matrix, config.recon_damage_matrix, config.bda_damage_matrix)


def test_cbba_isolation_and_allocation_pressure_generators() -> None:
    config = DynamicConfig()
    isolation = generate_cbba_isolation(D1_CELLS[0], 6000, config)
    assert isolation.scenario_id.startswith("D3A-")
    scenarios = [
        generate_allocation_pressure(condition, 7000, config)
        for condition in ALLOCATION_PRESSURE_CONDITIONS
    ]
    assert all(len(item.agents) == 6 and len(item.targets) == 15 for item in scenarios)
    assert len({item.scenario_id for item in scenarios}) == len(scenarios)
    tight = scenarios[ALLOCATION_PRESSURE_CONDITIONS.index("tight_resources")]
    reference = scenarios[0]
    assert tight.agents[0].initial_ammo_total < reference.agents[0].initial_ammo_total
    shared = scenarios[ALLOCATION_PRESSURE_CONDITIONS.index("shared_high_value")]
    assert all(target.belief == (0.72, 0.08, 0.18, 0.02) for target in shared.targets)
