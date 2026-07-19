"""Immutable configurations loaded from a digest-pinned pre-registration."""

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .rollout import RolloutParameters


BinaryMatrix = tuple[tuple[float, float], tuple[float, float]]
PREREGISTRATION_SHA256 = (
    "314f923560a280221149613fffd7f51eb358150658e11bebf89640d9311cb57e"
)


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """One immutable sensing/cost configuration for the simplex sweep."""

    config_id: str
    sensor_variant: str
    recon_class_matrix: BinaryMatrix
    recon_damage_matrix: BinaryMatrix
    bda_damage_matrix: BinaryMatrix
    params: RolloutParameters


def preregistration_path() -> Path:
    """Return the independent, human-auditable parameter snapshot."""

    return Path(__file__).resolve().parents[2] / "preregistration/first_batch.json"


def _load_preregistration() -> dict[str, Any]:
    path = preregistration_path()
    actual_digest = sha256_file(path)
    if actual_digest != PREREGISTRATION_SHA256:
        raise RuntimeError(
            "first-batch pre-registration digest mismatch: "
            f"expected {PREREGISTRATION_SHA256}, got {actual_digest}"
        )
    registration = json.loads(path.read_text(encoding="utf-8"))
    if registration.get("schema_version") != 1:
        raise RuntimeError("unsupported pre-registration schema")
    if registration.get("state_order") != ["HA", "HD", "LA", "LD"]:
        raise RuntimeError("pre-registration state order has changed")
    if registration.get("matrix_convention") != (
        "observation_rows_truth_columns"
    ):
        raise RuntimeError("pre-registration matrix convention has changed")
    return registration


def _matrix(values: list[list[float]]) -> BinaryMatrix:
    if len(values) != 2 or any(len(row) != 2 for row in values):
        raise RuntimeError("registered sensor matrices must have shape (2, 2)")
    return (
        (float(values[0][0]), float(values[0][1])),
        (float(values[1][0]), float(values[1][1])),
    )


def _cost_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def validation_parameter_sets() -> tuple[ValidationConfig, ...]:
    """Expand the digest-pinned four-sensor by 27-cost pre-registration."""

    registration = _load_preregistration()
    base = registration["base_parameters"]
    costs = registration["cost_grid"]
    configs: list[ValidationConfig] = []
    for variant in registration["sensor_variants"]:
        variant_id = str(variant["id"])
        recon_class = _matrix(variant["recon_class_matrix"])
        recon_damage = _matrix(variant["recon_damage_matrix"])
        bda_damage = _matrix(variant["bda_damage_matrix"])
        for cost_r, cost_a, cost_b in product(
            costs["cost_r"], costs["cost_a"], costs["cost_b"]
        ):
            cost_r = float(cost_r)
            cost_a = float(cost_a)
            cost_b = float(cost_b)
            config_id = (
                f"{variant_id}_r{_cost_label(cost_r)}"
                f"_a{_cost_label(cost_a)}_b{_cost_label(cost_b)}"
            )
            configs.append(
                ValidationConfig(
                    config_id=config_id,
                    sensor_variant=variant_id,
                    recon_class_matrix=recon_class,
                    recon_damage_matrix=recon_damage,
                    bda_damage_matrix=bda_damage,
                    params=RolloutParameters(
                        value_h=float(base["value_h"]),
                        value_l=float(base["value_l"]),
                        pi_h=float(base["pi_h"]),
                        pi_l=float(base["pi_l"]),
                        duration_r=float(base["duration_r"]),
                        duration_a=float(base["duration_a"]),
                        duration_b=float(base["duration_b"]),
                        cost_r=cost_r,
                        cost_a=cost_a,
                        cost_b=cost_b,
                        beta=float(base["beta"]),
                    ),
                )
            )
    expected_count = int(registration["expected_configuration_count"])
    if len(configs) != expected_count:
        raise RuntimeError(
            f"pre-registration expands to {len(configs)}, expected {expected_count}"
        )
    return tuple(configs)
