"""Attack transitions, single-payment rewards, and joint-belief statistics."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import as_belief


def _probability(name: str, value: float) -> float:
    probability = float(value)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return probability


def attack_matrix(pi_h: float, pi_l: float) -> NDArray[np.float64]:
    """Return the absorbing Attack transition for ``(HA, HD, LA, LD)``."""

    p_h = _probability("pi_h", pi_h)
    p_l = _probability("pi_l", pi_l)
    return np.array(
        [
            [1.0 - p_h, 0.0, 0.0, 0.0],
            [p_h, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0 - p_l, 0.0],
            [0.0, 0.0, p_l, 1.0],
        ],
        dtype=np.float64,
    )


def predict_attack(
    belief: ArrayLike, pi_h: float, pi_l: float
) -> NDArray[np.float64]:
    """Predict the joint belief immediately after an unobserved Attack."""

    b = as_belief(belief)
    return attack_matrix(pi_h, pi_l) @ b


def expected_attack_reward(
    belief: ArrayLike,
    pi_h: float,
    pi_l: float,
    value_h: float,
    value_l: float,
) -> float:
    """Expected value paid only for a new alive-to-destroyed transition."""

    b = as_belief(belief)
    p_h = _probability("pi_h", pi_h)
    p_l = _probability("pi_l", pi_l)
    return float(value_h) * p_h * b[0] + float(value_l) * p_l * b[2]


def apply_physical_attack(
    state: int,
    pi_h: float,
    pi_l: float,
    value_h: float,
    value_l: float,
    uniform: float,
) -> tuple[int, float]:
    """Apply one sampled Attack to a physical hidden state."""

    if state not in (0, 1, 2, 3):
        raise ValueError("state must be one of 0, 1, 2, or 3")
    p_h = _probability("pi_h", pi_h)
    p_l = _probability("pi_l", pi_l)
    draw = _probability("uniform", uniform)
    if state in (1, 3):
        return state, 0.0
    success_probability = p_h if state == 0 else p_l
    if draw < success_probability:
        return state + 1, float(value_h if state == 0 else value_l)
    return state, 0.0


def marginals(belief: ArrayLike) -> tuple[float, float]:
    """Return ``(P(H), P(A))`` for the frozen state order."""

    b = as_belief(belief)
    return float(b[0] + b[1]), float(b[0] + b[2])


def survival_given_class(belief: ArrayLike) -> tuple[float, float]:
    """Return ``(P(A|H), P(A|L))``; reject undefined conditionals."""

    b = as_belief(belief)
    p_h = float(b[0] + b[1])
    p_l = float(b[2] + b[3])
    if p_h <= 0.0 or p_l <= 0.0:
        raise ValueError("both target classes must have positive probability")
    return float(b[0] / p_h), float(b[2] / p_l)


def class_survival_covariance(belief: ArrayLike) -> float:
    """Covariance of the high-class and alive-state indicators."""

    b = as_belief(belief)
    p_h, p_alive = marginals(b)
    return float(b[0] - p_h * p_alive)
