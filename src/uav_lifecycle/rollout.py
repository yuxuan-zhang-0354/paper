"""Common-baseline, one-step action-value surrogates."""

from dataclasses import dataclass, fields
from math import exp, isfinite

from numpy.typing import ArrayLike

from .attack import expected_attack_reward, predict_attack
from .belief import bayes_update, observation_probabilities
from .model import as_belief


@dataclass(frozen=True, slots=True)
class RolloutParameters:
    """Immutable target-local parameters for the common terminal surrogate."""

    value_h: float
    value_l: float
    pi_h: float
    pi_l: float
    duration_r: float
    duration_a: float
    duration_b: float
    cost_r: float
    cost_a: float
    cost_b: float
    beta: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = float(getattr(self, field.name))
            if not isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            if field.name in {"pi_h", "pi_l"}:
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{field.name} must lie in [0, 1]")
            elif value < 0.0:
                raise ValueError(f"{field.name} must be nonnegative")


def discount(duration: float, beta: float) -> float:
    """Exponential discount ``exp(-beta * duration)``."""

    duration_value = float(duration)
    beta_value = float(beta)
    if not isfinite(duration_value) or duration_value < 0.0:
        raise ValueError("duration must be finite and nonnegative")
    if not isfinite(beta_value) or beta_value < 0.0:
        raise ValueError("beta must be finite and nonnegative")
    return exp(-beta_value * duration_value)


def terminal_attack_value(
    belief: ArrayLike, params: RolloutParameters
) -> float:
    """Net value of exactly one terminal Attack from *belief*."""

    reward = expected_attack_reward(
        belief,
        params.pi_h,
        params.pi_l,
        params.value_h,
        params.value_l,
    )
    return -params.cost_a + discount(params.duration_a, params.beta) * reward


def terminal_surrogate(belief: ArrayLike, params: RolloutParameters) -> float:
    """Common post-action continuation value ``max(0, A_T)``."""

    return max(0.0, terminal_attack_value(belief, params))


def _expected_terminal_value(
    belief: ArrayLike, kernel: ArrayLike, params: RolloutParameters
) -> float:
    probabilities = observation_probabilities(belief, kernel)
    return sum(
        float(probability)
        * terminal_surrogate(bayes_update(belief, kernel, observation), params)
        for observation, probability in enumerate(probabilities)
        if probability > 0.0
    )


def action_values(
    belief: ArrayLike,
    recon_observation_kernel: ArrayLike,
    bda_observation_kernel: ArrayLike,
    params: RolloutParameters,
) -> dict[str, float]:
    """Evaluate Recon, Attack, BDA, and Defer against one common baseline."""

    b = as_belief(belief)
    recon = -params.cost_r + discount(
        params.duration_r, params.beta
    ) * _expected_terminal_value(b, recon_observation_kernel, params)
    reward = expected_attack_reward(
        b, params.pi_h, params.pi_l, params.value_h, params.value_l
    )
    after_attack = predict_attack(b, params.pi_h, params.pi_l)
    attack = -params.cost_a + discount(params.duration_a, params.beta) * (
        reward + terminal_surrogate(after_attack, params)
    )
    bda = -params.cost_b + discount(
        params.duration_b, params.beta
    ) * _expected_terminal_value(b, bda_observation_kernel, params)
    return {
        "recon": float(recon),
        "attack": float(attack),
        "bda": float(bda),
        "defer": 0.0,
    }


def costless_information_value(
    belief: ArrayLike, kernel: ArrayLike, params: RolloutParameters
) -> float:
    """Gross one-step information value, excluding sensing cost and delay."""

    expected_terminal = _expected_terminal_value(belief, kernel, params)
    return expected_terminal - terminal_surrogate(belief, params)
