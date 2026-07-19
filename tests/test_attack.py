import numpy as np

from uav_lifecycle.attack import (
    apply_physical_attack,
    attack_matrix,
    expected_attack_reward,
    marginals,
    predict_attack,
)


def test_attack_prediction_closes_simplex_and_preserves_class_marginal():
    belief = np.array([0.4, 0.1, 0.3, 0.2])
    predicted = predict_attack(belief, pi_h=0.4, pi_l=0.75)
    np.testing.assert_allclose(predicted.sum(), 1.0)
    np.testing.assert_allclose(predicted[[0, 1]].sum(), belief[[0, 1]].sum())
    np.testing.assert_allclose(predicted[[2, 3]].sum(), belief[[2, 3]].sum())
    assert predicted[[0, 2]].sum() <= belief[[0, 2]].sum()


def test_attack_matrix_is_column_stochastic():
    np.testing.assert_allclose(attack_matrix(0.4, 0.75).sum(axis=0), np.ones(4))


def test_two_attack_expected_reward_matches_at_least_one_success():
    belief = np.array([0.5, 0.0, 0.5, 0.0])
    first = expected_attack_reward(belief, 0.4, 0.75, 100.0, 30.0)
    after = predict_attack(belief, 0.4, 0.75)
    second = expected_attack_reward(after, 0.4, 0.75, 100.0, 30.0)
    expected = 0.5 * 100.0 * (1.0 - 0.6**2) + 0.5 * 30.0 * (
        1.0 - 0.25**2
    )
    np.testing.assert_allclose(first + second, expected)


def test_initial_destroyed_mass_has_zero_attack_reward():
    destroyed_only = np.array([0.0, 0.5, 0.0, 0.5])
    assert (
        expected_attack_reward(destroyed_only, 0.4, 0.75, 100.0, 30.0)
        == 0.0
    )


def test_destroyed_state_is_absorbing_and_never_repays_reward():
    next_state, reward = apply_physical_attack(
        state=1,
        pi_h=0.4,
        pi_l=0.75,
        value_h=100.0,
        value_l=30.0,
        uniform=0.0,
    )
    assert next_state == 1
    assert reward == 0.0


def test_successful_physical_attack_pays_once_and_moves_to_destroyed_state():
    next_state, reward = apply_physical_attack(
        state=2,
        pi_h=0.4,
        pi_l=0.75,
        value_h=100.0,
        value_l=30.0,
        uniform=0.5,
    )
    assert (next_state, reward) == (3, 30.0)


def test_marginals_return_high_class_and_alive_probabilities():
    p_h, p_alive = marginals([0.4, 0.1, 0.3, 0.2])
    np.testing.assert_allclose([p_h, p_alive], [0.5, 0.7])
