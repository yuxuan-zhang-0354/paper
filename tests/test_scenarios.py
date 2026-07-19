import numpy as np

from uav_lifecycle.artifacts import sha256_file
from uav_lifecycle.scenarios import (
    PREREGISTRATION_SHA256,
    preregistration_path,
    validation_parameter_sets,
)


def test_preregistration_snapshot_matches_pinned_digest():
    assert sha256_file(preregistration_path()) == PREREGISTRATION_SHA256


def test_pre_registered_parameter_count_and_story_direction():
    configs = validation_parameter_sets()
    assert len(configs) == 108
    assert all(config.params.pi_h < config.params.pi_l for config in configs)
    assert len({config.config_id for config in configs}) == 108


def test_sensor_variants_apply_only_the_registered_single_factor():
    configs = {config.sensor_variant: config for config in validation_parameter_sets()}
    baseline = configs["baseline"]
    class_minus = configs["recon_class_minus_010"]
    recon_damage_plus = configs["recon_damage_plus_010"]
    bda_damage_minus = configs["bda_damage_minus_010"]

    np.testing.assert_allclose(
        class_minus.recon_class_matrix, [[0.55, 0.25], [0.45, 0.75]]
    )
    assert class_minus.recon_damage_matrix == baseline.recon_damage_matrix
    assert class_minus.bda_damage_matrix == baseline.bda_damage_matrix

    np.testing.assert_allclose(
        recon_damage_plus.recon_damage_matrix,
        [[0.85, 0.15], [0.15, 0.85]],
    )
    assert recon_damage_plus.recon_class_matrix == baseline.recon_class_matrix
    assert recon_damage_plus.bda_damage_matrix == baseline.bda_damage_matrix

    np.testing.assert_allclose(
        bda_damage_minus.bda_damage_matrix, [[0.82, 0.16], [0.18, 0.84]]
    )
    assert bda_damage_minus.recon_class_matrix == baseline.recon_class_matrix
    assert bda_damage_minus.recon_damage_matrix == baseline.recon_damage_matrix


def test_all_registered_sensor_matrices_are_column_stochastic():
    for config in validation_parameter_sets():
        for matrix in (
            config.recon_class_matrix,
            config.recon_damage_matrix,
            config.bda_damage_matrix,
        ):
            np.testing.assert_allclose(np.asarray(matrix).sum(axis=0), [1.0, 1.0])
