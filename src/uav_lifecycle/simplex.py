"""Exact rational grids and deterministic action ranking."""

from math import comb, isfinite
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


ACTION_ORDER = ("recon", "attack", "bda", "defer")


def simplex_grid(step: float) -> NDArray[np.float64]:
    """Enumerate the four-state simplex using an exact integer denominator."""

    step_value = float(step)
    if not isfinite(step_value) or step_value <= 0.0:
        raise ValueError("step must be finite and positive")
    denominator = round(1.0 / step_value)
    if denominator < 1 or not np.isclose(
        denominator * step_value, 1.0, atol=1e-12, rtol=0.0
    ):
        raise ValueError("step must be the reciprocal of a positive integer")

    point_count = comb(denominator + 3, 3)
    integer_grid = np.empty((point_count, 4), dtype=np.int64)
    row = 0
    for ha in range(denominator + 1):
        for hd in range(denominator - ha + 1):
            for la in range(denominator - ha - hd + 1):
                ld = denominator - ha - hd - la
                integer_grid[row] = (ha, hd, la, ld)
                row += 1
    if row != point_count:
        raise RuntimeError("internal simplex enumeration count mismatch")
    return integer_grid.astype(np.float64) / denominator


def rank_actions(values: Mapping[str, float]) -> tuple[str, str, float]:
    """Return the top two actions and their margin using a fixed tie order."""

    if set(values) != set(ACTION_ORDER):
        raise ValueError(f"action keys must be exactly {ACTION_ORDER}")
    numeric = {action: float(values[action]) for action in ACTION_ORDER}
    if not all(np.isfinite(value) for value in numeric.values()):
        raise ValueError("action values must be finite")
    ranked = sorted(ACTION_ORDER, key=lambda action: -numeric[action])
    best, second = ranked[:2]
    return best, second, numeric[best] - numeric[second]
