from experiments.build_d3_manifests import canonical_digest
from experiments.build_d4_manifest import build_d4_manifest
from uav_lifecycle.dynamic_d4 import generate_battlefield_structure, generate_reachability
from uav_lifecycle.dynamic_types import DynamicConfig


def test_d4_manifest_rectangle_is_frozen() -> None:
    manifest = build_d4_manifest()
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    assert manifest["manifest_digest"] == canonical_digest(unsigned)
    assert manifest["scenario_count"] == 1600
    assert manifest["expected_record_count"] == 8448
    assert len(manifest["scenario_ids"]) == len(set(manifest["scenario_ids"])) == 1600
    assert len({tuple(item) for item in manifest["expected_rectangle"]}) == 8448


def test_wreck_conditions_are_nested_and_truth_hidden() -> None:
    config = DynamicConfig()
    scenarios = [generate_battlefield_structure("uniform", rate, 8000, config) for rate in (0.0, 0.2, 0.4, 0.6)]
    wreck_sets = [
        {target.target_id for target in scenario.private_targets if target.true_damage == "D"}
        for scenario in scenarios
    ]
    assert wreck_sets[0] <= wreck_sets[1] <= wreck_sets[2] <= wreck_sets[3]
    assert all(target.belief[1] + target.belief[3] == rate for scenario, rate in zip(scenarios, (0.0, 0.2, 0.4, 0.6), strict=True) for target in scenario.targets)
    assert [target.true_category for target in scenarios[0].private_targets] == [target.true_category for target in scenarios[-1].private_targets]


def test_value_correlated_prior_is_probabilistic_and_reachability_scales() -> None:
    config = DynamicConfig()
    correlated = generate_battlefield_structure("value_correlated", 0.2, 8000, config)
    p_high = {round(target.belief[0] + target.belief[1], 2) for target in correlated.targets}
    assert p_high == {0.15, 0.65}
    low = generate_reachability(1.25, 0.75, 9000, config)
    high = generate_reachability(0.75, 1.25, 9000, config)
    assert low.t_max_tick < high.t_max_tick
    assert low.agents[0].initial_distance_total < high.agents[0].initial_distance_total
