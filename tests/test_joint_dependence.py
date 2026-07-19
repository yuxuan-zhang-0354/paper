import numpy as np
import pytest

from uav_lifecycle.attack import (
    class_survival_covariance,
    expected_attack_reward,
    predict_attack,
    survival_given_class,
)
from uav_lifecycle.belief import bayes_update, bda_kernel


def test_attack_induces_dependence_from_an_independent_prior():
    belief = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.isclose(class_survival_covariance(belief), 0.0)
    predicted = predict_attack(belief, pi_h=0.2, pi_l=0.8)
    alive_h, alive_l = survival_given_class(predicted)
    assert not np.isclose(alive_h, alive_l)
    assert not np.isclose(class_survival_covariance(predicted), 0.0)


def test_bda_destroyed_observation_reduces_high_class_probability_when_high_targets_are_harder():
    prior = np.array([0.5, 0.0, 0.5, 0.0])
    attacked = predict_attack(prior, pi_h=0.2, pi_l=0.8)
    kernel = bda_kernel([[0.9, 0.1], [0.1, 0.9]])
    posterior = bayes_update(attacked, kernel, observation=1)
    assert posterior[[0, 1]].sum() < 0.5


def test_bda_alive_observation_increases_high_class_probability_when_high_targets_are_harder():
    prior = np.array([0.5, 0.0, 0.5, 0.0])
    attacked = predict_attack(prior, pi_h=0.2, pi_l=0.8)
    kernel = bda_kernel([[0.9, 0.1], [0.1, 0.9]])
    posterior = bayes_update(attacked, kernel, observation=0)
    assert posterior[[0, 1]].sum() > 0.5


def test_equal_marginals_do_not_determine_attack_value():
    independent = np.array([0.25, 0.25, 0.25, 0.25])
    correlated = np.array([0.5, 0.0, 0.0, 0.5])
    value_1 = expected_attack_reward(independent, 0.4, 0.75, 100.0, 30.0)
    value_2 = expected_attack_reward(correlated, 0.4, 0.75, 100.0, 30.0)
    assert not np.isclose(value_1, value_2)


def test_post_attack_independence_exactly_when_conditional_survivals_match():
    prior = np.array([0.125, 0.375, 0.25, 0.25])
    predicted = predict_attack(prior, pi_h=0.2, pi_l=0.6)
    np.testing.assert_allclose(survival_given_class(predicted), [0.2, 0.2])
    assert np.isclose(class_survival_covariance(predicted), 0.0, atol=1e-12)


def test_independence_condition_rejects_zero_probability_class_boundary():
    with pytest.raises(ValueError, match="both target classes"):
        survival_given_class([0.5, 0.5, 0.0, 0.0])
