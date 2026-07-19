"""Observation kernels and Bayesian updates for the four-state belief."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import as_belief, as_column_stochastic


def recon_kernel(
    class_matrix: ArrayLike, damage_matrix: ArrayLike
) -> NDArray[np.float64]:
    """Build the conditionally independent joint Recon observation kernel."""

    mc = as_column_stochastic(class_matrix)
    ms = as_column_stochastic(damage_matrix)
    if mc.shape != (2, 2) or ms.shape != (2, 2):
        raise ValueError("binary Recon matrices must both have shape (2, 2)")
    kernel = np.empty((4, 4), dtype=np.float64)
    for observed_class in range(2):
        for observed_damage in range(2):
            row = 2 * observed_class + observed_damage
            for true_class in range(2):
                for true_damage in range(2):
                    column = 2 * true_class + true_damage
                    kernel[row, column] = (
                        mc[observed_class, true_class]
                        * ms[observed_damage, true_damage]
                    )
    return kernel


def bda_kernel(damage_matrix: ArrayLike) -> NDArray[np.float64]:
    """Build a BDA kernel with no direct target-class observation channel."""

    ms = as_column_stochastic(damage_matrix)
    if ms.shape != (2, 2):
        raise ValueError("binary BDA damage matrix must have shape (2, 2)")
    return np.column_stack((ms[:, 0], ms[:, 1], ms[:, 0], ms[:, 1]))


def observation_probabilities(
    belief: ArrayLike, kernel: ArrayLike
) -> NDArray[np.float64]:
    """Return predictive probabilities for all observations in *kernel*."""

    b = as_belief(belief)
    z = as_column_stochastic(kernel)
    if z.shape[1] != 4:
        raise ValueError("kernel must have four hidden-state columns")
    return z @ b


def bayes_update(
    belief: ArrayLike, kernel: ArrayLike, observation: int
) -> NDArray[np.float64]:
    """Condition *belief* on one positive-probability observation."""

    b = as_belief(belief)
    z = as_column_stochastic(kernel)
    if z.shape[1] != 4:
        raise ValueError("kernel must have four hidden-state columns")
    if observation < 0 or observation >= z.shape[0]:
        raise IndexError("observation index out of range")
    numerator = z[observation] * b
    denominator = float(numerator.sum())
    if denominator <= 0.0:
        raise ValueError("observation has zero predictive probability")
    return numerator / denominator


def expected_posterior(
    belief: ArrayLike, kernel: ArrayLike
) -> NDArray[np.float64]:
    """Return the probability-weighted posterior, equal to the prior."""

    b = as_belief(belief)
    probabilities = observation_probabilities(b, kernel)
    result = np.zeros(4, dtype=np.float64)
    for observation, probability in enumerate(probabilities):
        if probability > 0.0:
            result += probability * bayes_update(b, kernel, observation)
    return result
