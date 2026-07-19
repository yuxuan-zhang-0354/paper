import numpy as np
import pytest

from uav_lifecycle.belief import (
    bayes_update,
    bda_kernel,
    expected_posterior,
    observation_probabilities,
    recon_kernel,
)


CLASS = np.array([[0.65, 0.15], [0.35, 0.85]])
DAMAGE_R = np.array([[0.75, 0.25], [0.25, 0.75]])
DAMAGE_B = np.array([[0.92, 0.06], [0.08, 0.94]])
B = np.array([0.4, 0.1, 0.3, 0.2])


def test_recon_kernel_shape_and_columns():
    kernel = recon_kernel(CLASS, DAMAGE_R)
    assert kernel.shape == (4, 4)
    np.testing.assert_allclose(kernel.sum(axis=0), np.ones(4))


def test_bda_has_no_direct_class_channel():
    kernel = bda_kernel(DAMAGE_B)
    np.testing.assert_allclose(kernel[:, 0], kernel[:, 2])
    np.testing.assert_allclose(kernel[:, 1], kernel[:, 3])


def test_predictive_probabilities_and_posterior_are_normalized():
    kernel = recon_kernel(CLASS, DAMAGE_R)
    predictive = observation_probabilities(B, kernel)
    np.testing.assert_allclose(predictive.sum(), 1.0)
    posterior = bayes_update(B, kernel, 0)
    np.testing.assert_allclose(posterior.sum(), 1.0)
    assert np.all(posterior >= 0.0)


def test_bayes_martingale_identity():
    kernel = recon_kernel(CLASS, DAMAGE_R)
    np.testing.assert_allclose(expected_posterior(B, kernel), B, atol=1e-12)


def test_zero_probability_observation_is_rejected():
    kernel = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="zero predictive probability"):
        bayes_update(B, kernel, 1)


def test_bayes_update_rejects_kernel_without_four_hidden_states():
    wrong_kernel = np.array([[1.0], [0.0]])
    with pytest.raises(ValueError, match="four hidden-state columns"):
        bayes_update(B, wrong_kernel, 0)
