"""State ordering and strict probability validators."""

from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray


class StateIndex(IntEnum):
    """Frozen hidden-state order used throughout the validation harness."""

    HA = 0
    HD = 1
    LA = 2
    LD = 3


def as_belief(values: ArrayLike, atol: float = 1e-12) -> NDArray[np.float64]:
    """Return *values* as a validated belief over ``(HA, HD, LA, LD)``."""

    belief = np.asarray(values, dtype=np.float64)
    if belief.shape != (4,):
        raise ValueError(f"belief must have shape (4,), got {belief.shape}")
    if np.any(belief < -atol):
        raise ValueError("belief contains a negative probability")
    if not np.isclose(float(belief.sum()), 1.0, atol=atol, rtol=0.0):
        raise ValueError("belief probabilities must sum to one")
    belief = np.maximum(belief, 0.0)
    return belief / belief.sum()


def as_column_stochastic(
    values: ArrayLike, atol: float = 1e-12
) -> NDArray[np.float64]:
    """Validate an observation-row/truth-column stochastic matrix."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError(
            "matrix must be two-dimensional with at least one truth column"
        )
    if np.any(matrix < -atol):
        raise ValueError("matrix contains a negative probability")
    if not np.allclose(matrix.sum(axis=0), 1.0, atol=atol, rtol=0.0):
        raise ValueError("each truth column must sum to one")
    return np.maximum(matrix, 0.0)
