from math import comb

import numpy as np
import pytest

from uav_lifecycle.simplex import rank_actions, simplex_grid


def test_simplex_step_half_has_ten_points():
    grid = simplex_grid(0.5)
    assert grid.shape == (10, 4)
    assert (grid.sum(axis=1) == 1.0).all()


def test_simplex_step_point_zero_two_has_registered_count():
    grid = simplex_grid(0.02)
    assert grid.shape == (comb(53, 3), 4)
    np.testing.assert_allclose(grid.sum(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("step", [0.0, -0.5, 0.3])
def test_simplex_rejects_nonreciprocal_positive_step(step):
    with pytest.raises(ValueError):
        simplex_grid(step)


def test_rank_actions_uses_deterministic_order_for_ties():
    best, second, margin = rank_actions(
        {"recon": 1.0, "attack": 1.0, "bda": 0.0, "defer": 0.0}
    )
    assert (best, second, margin) == ("recon", "attack", 0.0)
