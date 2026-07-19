"""Run deterministic Gate A/B property checks and persist their evidence."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np

from uav_lifecycle.artifacts import write_json_atomic
from uav_lifecycle.attack import (
    apply_physical_attack,
    class_survival_covariance,
    expected_attack_reward,
    marginals,
    predict_attack,
    survival_given_class,
)
from uav_lifecycle.belief import (
    bayes_update,
    bda_kernel,
    expected_posterior,
    observation_probabilities,
    recon_kernel,
)
from uav_lifecycle.rollout import action_values, costless_information_value
from uav_lifecycle.scenarios import validation_parameter_sets


DEFAULT_SEED = 7132026
DEFAULT_SAMPLE_COUNT = 10_000
TOLERANCE = 1e-10


def _simplex_failure(values: np.ndarray, tolerance: float) -> bool:
    return bool(
        np.any(values < -tolerance)
        or not np.isclose(values.sum(), 1.0, atol=tolerance, rtol=0.0)
    )


def run_property_checks(
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Evaluate the pre-registered formula invariants on seeded beliefs."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    rng = np.random.default_rng(seed)
    beliefs = rng.dirichlet(np.ones(4, dtype=np.float64), size=sample_count)
    config = validation_parameter_sets()[0]
    params = config.params
    zr = recon_kernel(config.recon_class_matrix, config.recon_damage_matrix)
    zb = bda_kernel(config.bda_damage_matrix)

    gate_a_names = (
        "recon_predictive_normalization",
        "bda_predictive_normalization",
        "recon_posterior_simplex",
        "bda_posterior_simplex",
        "bayes_martingale",
        "attack_simplex",
        "attack_class_marginal",
        "attack_alive_monotonicity",
        "two_attack_reward_identity",
        "costless_information_nonnegative",
        "rollout_repeatability",
        "rollout_fixed_oracle",
        "rubble_zero_reward",
        "absorbing_reward_single_payment",
    )
    gate_b_names = (
        "post_attack_covariance_identity",
        "attack_induces_dependence_from_independent_prior",
        "bda_destroyed_lowers_high",
        "bda_alive_raises_high",
        "equal_marginal_reward_difference",
        "independence_condition_witness",
    )
    failures = {name: 0 for name in (*gate_a_names, *gate_b_names)}

    for belief in beliefs:
        recon_probabilities = observation_probabilities(belief, zr)
        bda_probabilities = observation_probabilities(belief, zb)
        if not np.isclose(
            recon_probabilities.sum(), 1.0, atol=TOLERANCE, rtol=0.0
        ):
            failures["recon_predictive_normalization"] += 1
        if not np.isclose(
            bda_probabilities.sum(), 1.0, atol=TOLERANCE, rtol=0.0
        ):
            failures["bda_predictive_normalization"] += 1
        if any(
            _simplex_failure(bayes_update(belief, zr, observation), TOLERANCE)
            for observation, probability in enumerate(recon_probabilities)
            if probability > 0.0
        ):
            failures["recon_posterior_simplex"] += 1
        if any(
            _simplex_failure(bayes_update(belief, zb, observation), TOLERANCE)
            for observation, probability in enumerate(bda_probabilities)
            if probability > 0.0
        ):
            failures["bda_posterior_simplex"] += 1
        if not (
            np.allclose(
                expected_posterior(belief, zr),
                belief,
                atol=TOLERANCE,
                rtol=0.0,
            )
            and np.allclose(
                expected_posterior(belief, zb),
                belief,
                atol=TOLERANCE,
                rtol=0.0,
            )
        ):
            failures["bayes_martingale"] += 1

        attacked = predict_attack(belief, params.pi_h, params.pi_l)
        if _simplex_failure(attacked, TOLERANCE):
            failures["attack_simplex"] += 1
        p_h_before, p_alive_before = marginals(belief)
        p_h_after, p_alive_after = marginals(attacked)
        if not np.isclose(
            p_h_before, p_h_after, atol=TOLERANCE, rtol=0.0
        ):
            failures["attack_class_marginal"] += 1
        if p_alive_after > p_alive_before + TOLERANCE:
            failures["attack_alive_monotonicity"] += 1

        first_reward = expected_attack_reward(
            belief,
            params.pi_h,
            params.pi_l,
            params.value_h,
            params.value_l,
        )
        second_reward = expected_attack_reward(
            attacked,
            params.pi_h,
            params.pi_l,
            params.value_h,
            params.value_l,
        )
        expected_two_attack_reward = (
            belief[0]
            * params.value_h
            * (1.0 - (1.0 - params.pi_h) ** 2)
            + belief[2]
            * params.value_l
            * (1.0 - (1.0 - params.pi_l) ** 2)
        )
        if not np.isclose(
            first_reward + second_reward,
            expected_two_attack_reward,
            atol=TOLERANCE,
            rtol=0.0,
        ):
            failures["two_attack_reward_identity"] += 1

        if (
            costless_information_value(belief, zr, params) < -TOLERANCE
            or costless_information_value(belief, zb, params) < -TOLERANCE
        ):
            failures["costless_information_nonnegative"] += 1
        first_values = action_values(belief, zr, zb, params)
        if first_values != action_values(belief, zr, zb, params):
            failures["rollout_repeatability"] += 1

        alive_h, alive_l = survival_given_class(attacked)
        p_l_after = 1.0 - p_h_after
        covariance_identity = p_h_after * p_l_after * (alive_h - alive_l)
        if not np.isclose(
            class_survival_covariance(attacked),
            covariance_identity,
            atol=TOLERANCE,
            rtol=0.0,
        ):
            failures["post_attack_covariance_identity"] += 1

    rubble_reward = expected_attack_reward(
        [0.0, 0.5, 0.0, 0.5],
        params.pi_h,
        params.pi_l,
        params.value_h,
        params.value_l,
    )
    if rubble_reward != 0.0:
        failures["rubble_zero_reward"] += 1

    rollout_oracle_belief = np.array([0.4, 0.1, 0.3, 0.2])
    rollout_oracle_values = action_values(
        rollout_oracle_belief, zr, zb, params
    )
    rollout_oracle_actual = np.array(
        [
            rollout_oracle_values["recon"],
            rollout_oracle_values["attack"],
            rollout_oracle_values["bda"],
            rollout_oracle_values["defer"],
        ]
    )
    rollout_oracle_expected = np.array(
        [
            12.638741856995518,
            20.51289886564056,
            14.660392991376822,
            0.0,
        ]
    )
    if not np.allclose(
        rollout_oracle_actual,
        rollout_oracle_expected,
        atol=1e-12,
        rtol=0.0,
    ):
        failures["rollout_fixed_oracle"] += 1
    state_after, first_physical_reward = apply_physical_attack(
        0,
        params.pi_h,
        params.pi_l,
        params.value_h,
        params.value_l,
        0.0,
    )
    _, second_physical_reward = apply_physical_attack(
        state_after,
        params.pi_h,
        params.pi_l,
        params.value_h,
        params.value_l,
        0.0,
    )
    if not (
        state_after == 1
        and first_physical_reward == params.value_h
        and second_physical_reward == 0.0
    ):
        failures["absorbing_reward_single_payment"] += 1

    prior = np.array([0.5, 0.0, 0.5, 0.0])
    attacked_prior = predict_attack(prior, 0.2, 0.8)
    evidence_bda_kernel = bda_kernel(((0.9, 0.1), (0.1, 0.9)))
    destroyed_posterior = bayes_update(
        attacked_prior, evidence_bda_kernel, observation=1
    )
    alive_posterior = bayes_update(
        attacked_prior, evidence_bda_kernel, observation=0
    )
    high_after_destroyed = marginals(destroyed_posterior)[0]
    high_after_alive = marginals(alive_posterior)[0]
    if not high_after_destroyed < 0.5:
        failures["bda_destroyed_lowers_high"] += 1
    if not high_after_alive > 0.5:
        failures["bda_alive_raises_high"] += 1

    independent = np.array([0.25, 0.25, 0.25, 0.25])
    induced = predict_attack(independent, 0.2, 0.8)
    independent_covariance_before = class_survival_covariance(independent)
    independent_covariance_after = class_survival_covariance(induced)
    if not (
        np.isclose(
            independent_covariance_before, 0.0, atol=TOLERANCE, rtol=0.0
        )
        and not np.isclose(
            independent_covariance_after, 0.0, atol=TOLERANCE, rtol=0.0
        )
    ):
        failures["attack_induces_dependence_from_independent_prior"] += 1

    correlated = np.array([0.5, 0.0, 0.0, 0.5])
    independent_reward = expected_attack_reward(
        independent, 0.4, 0.75, 100.0, 30.0
    )
    correlated_reward = expected_attack_reward(
        correlated, 0.4, 0.75, 100.0, 30.0
    )
    if np.isclose(
        independent_reward, correlated_reward, atol=TOLERANCE, rtol=0.0
    ):
        failures["equal_marginal_reward_difference"] += 1

    independence_prior = np.array([0.125, 0.375, 0.25, 0.25])
    independence_after = predict_attack(independence_prior, 0.2, 0.6)
    independence_covariance = class_survival_covariance(independence_after)
    if not np.isclose(
        independence_covariance, 0.0, atol=TOLERANCE, rtol=0.0
    ):
        failures["independence_condition_witness"] += 1

    gate_a_pass = all(failures[name] == 0 for name in gate_a_names)
    gate_b_pass = all(failures[name] == 0 for name in gate_b_names)
    return {
        "sample_count": sample_count,
        "seed": seed,
        "tolerance": TOLERANCE,
        "failure_counts": failures,
        "total_failures": sum(failures.values()),
        "gate_a_pass": gate_a_pass,
        "gate_b_pass": gate_b_pass,
        "evidence": {
            "rubble_reward": rubble_reward,
            "first_physical_reward": first_physical_reward,
            "second_physical_reward": second_physical_reward,
            "rollout_fixed_oracle_actual": rollout_oracle_actual,
            "rollout_fixed_oracle_expected": rollout_oracle_expected,
            "high_probability_after_destroyed_bda": high_after_destroyed,
            "high_probability_after_alive_bda": high_after_alive,
            "equal_marginal_independent_reward": independent_reward,
            "equal_marginal_correlated_reward": correlated_reward,
            "independent_prior_covariance_before_attack": (
                independent_covariance_before
            ),
            "independent_prior_covariance_after_attack": (
                independent_covariance_after
            ),
            "independence_witness_covariance": independence_covariance,
        },
    }


def write_property_artifacts(
    output: str | Path,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    config = validation_parameter_sets()[0]
    summary = run_property_checks(sample_count=sample_count, seed=seed)
    write_json_atomic(
        destination / "config.json",
        {
            "experiment": "gate_a_b_properties",
            "sample_count": sample_count,
            "seed": seed,
            "state_order": ["HA", "HD", "LA", "LD"],
            "matrix_convention": "observation_rows_truth_columns",
            "tolerance": TOLERANCE,
            "baseline_config": config,
        },
    )
    write_json_atomic(destination / "summary.json", summary)
    (destination / "run.log").write_text(
        (
            f"property checks: samples={sample_count}, seed={seed}, "
            f"total_failures={summary['total_failures']}\n"
        ),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/properties"))
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    summary = write_property_artifacts(args.output, args.samples, args.seed)
    return 0 if summary["total_failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
