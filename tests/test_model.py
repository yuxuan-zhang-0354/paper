import numpy as np
import pytest

from uav_lifecycle.model import StateIndex, as_belief, as_column_stochastic


def test_state_order_is_frozen():
    assert [member.name for member in StateIndex] == ["HA", "HD", "LA", "LD"]
    assert [member.value for member in StateIndex] == [0, 1, 2, 3]


def test_as_belief_accepts_simplex_vector():
    actual = as_belief([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(actual, [0.1, 0.2, 0.3, 0.4])


@pytest.mark.parametrize(
    "bad",
    [[-0.1, 0.2, 0.4, 0.5], [0.2, 0.2, 0.2, 0.2], [1.0, 0.0]],
)
def test_as_belief_rejects_invalid_vectors(bad):
    with pytest.raises(ValueError):
        as_belief(bad)


def test_column_stochastic_convention():
    matrix = as_column_stochastic([[0.65, 0.15], [0.35, 0.85]])
    np.testing.assert_allclose(matrix.sum(axis=0), [1.0, 1.0])
