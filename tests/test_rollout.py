import numpy as np
import pytest

from uav_lifecycle.belief import bda_kernel, recon_kernel
from uav_lifecycle.rollout import (
    RolloutParameters,
    action_values,
    costless_information_value,
    discount,
    terminal_surrogate,
)


PARAMS = RolloutParameters(
    value_h=100.0,
    value_l=30.0,
    pi_h=0.4,
    pi_l=0.75,
    duration_r=4.0,
    duration_a=2.0,
    duration_b=1.5,
    cost_r=2.0,
    cost_a=6.0,
    cost_b=1.0,
    beta=0.02,
)
RC = np.array([[0.65, 0.15], [0.35, 0.85]])
RS = np.array([[0.75, 0.25], [0.25, 0.75]])
BS = np.array([[0.92, 0.06], [0.08, 0.94]])


def test_discount_has_semigroup_property():
    assert np.isclose(
        discount(3.0, 0.02) * discount(4.0, 0.02), discount(7.0, 0.02)
    )


def test_action_values_include_defer_zero_and_are_repeatable():
    belief = np.array([0.4, 0.1, 0.3, 0.2])
    zr = recon_kernel(RC, RS)
    zb = bda_kernel(BS)
    first = action_values(belief, zr, zb, PARAMS)
    second = action_values(belief, zr, zb, PARAMS)
    assert first == second
    assert set(first) == {"recon", "attack", "bda", "defer"}
    assert first["defer"] == 0.0


def test_costless_information_value_is_nonnegative():
    belief = np.array([0.4, 0.1, 0.3, 0.2])
    zr = recon_kernel(RC, RS)
    assert costless_information_value(belief, zr, PARAMS) >= -1e-12


def test_all_action_values_match_independent_hand_calculated_oracle():
    belief = np.array([0.4, 0.1, 0.3, 0.2])
    zr = recon_kernel(RC, RS)
    zb = bda_kernel(BS)
    values = action_values(belief, zr, zb, PARAMS)
    np.testing.assert_allclose(
        [
            terminal_surrogate(belief, PARAMS),
            values["recon"],
            values["attack"],
            values["bda"],
            values["defer"],
        ],
        [
            15.857959740715351,
            12.638741856995518,
            20.51289886564056,
            14.660392991376822,
            0.0,
        ],
        atol=1e-12,
        rtol=0.0,
    )


def test_terminal_continuation_off_branch_matches_independent_oracle():
    destroyed = np.array([0.0, 0.5, 0.0, 0.5])
    values = action_values(
        destroyed,
        recon_kernel(RC, RS),
        bda_kernel(BS),
        PARAMS,
    )
    assert terminal_surrogate(destroyed, PARAMS) == 0.0
    np.testing.assert_allclose(
        [values["recon"], values["attack"], values["bda"], values["defer"]],
        [-2.0, -6.0, -1.0, 0.0],
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"pi_h": -0.1},
        {"pi_l": 1.1},
        {"duration_r": -1.0},
        {"cost_b": -1.0},
        {"value_h": -1.0},
        {"beta": -0.1},
    ],
)
def test_rollout_parameters_reject_invalid_values(changes):
    values = {
        "value_h": 100.0,
        "value_l": 30.0,
        "pi_h": 0.4,
        "pi_l": 0.75,
        "duration_r": 4.0,
        "duration_a": 2.0,
        "duration_b": 1.5,
        "cost_r": 2.0,
        "cost_a": 6.0,
        "cost_b": 1.0,
        "beta": 0.02,
    }
    values.update(changes)
    with pytest.raises(ValueError):
        RolloutParameters(**values)
