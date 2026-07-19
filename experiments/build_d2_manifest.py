"""Build the frozen, non-executable D2 confirmation manifest."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from experiments.run_dynamic_mainline import D1_METHODS
from uav_lifecycle.artifacts import sha256_file, write_json_atomic
from uav_lifecycle.dynamic_scenarios import D1_CELLS, dynamic_registry_digest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/dynamic_mainline/d2_design/d2_manifest.json"
SPEC = ROOT / "docs/superpowers/specs/2026-07-14-dynamic-lifecycle-mainline-design.md"
DESIGN = ROOT / "docs/superpowers/specs/2026-07-14-dynamic-lifecycle-d2-manifest-design.md"
CONFIG_ID = "recon_damage_plus_010_r2_a6_b3"
START_INDEX = 1000
SCENARIOS_PER_CELL = 512
PILOT_VARIANCES = {
    "N2-M3-Rloose": 0.02166457873611356,
    "N2-M3-Rtight": 0.012511475184398227,
    "N2-M5-Rloose": 0.022174916310528727,
    "N2-M5-Rtight": 0.007299733493060603,
    "N3-M3-Rloose": 0.040751170266725,
    "N3-M3-Rtight": 0.020488571946831985,
    "N3-M5-Rloose": 0.022846649550740762,
    "N3-M5-Rtight": 0.006298995138356681,
}


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    return sha256(payload).hexdigest()


def build_manifest() -> dict[str, object]:
    cells = tuple(sorted(cell.cell_id for cell in D1_CELLS))
    scenario_ids = tuple(
        f"D2-{cell}-S{index:04d}"
        for cell in cells
        for index in range(START_INDEX, START_INDEX + SCENARIOS_PER_CELL)
    )
    scenario_cells = {
        scenario_id: scenario_id.removeprefix("D2-").rsplit("-S", 1)[0]
        for scenario_id in scenario_ids
    }
    expected = tuple(
        (scenario_id, method) for scenario_id in scenario_ids for method in D1_METHODS
    )
    replay_ids = tuple(
        f"D2-{cell}-S{index:04d}"
        for cell in cells
        for index in (START_INDEX, START_INDEX + SCENARIOS_PER_CELL - 1)
    )
    variance_sum = sum(PILOT_VARIANCES.values())
    standard_error = (2.0 * variance_sum / (len(cells) ** 2 * SCENARIOS_PER_CELL)) ** 0.5
    manifest: dict[str, object] = {
        "manifest_version": "dynamic_lifecycle_d2_manifest_v1",
        "spec_version": "dynamic_lifecycle_mainline_v2",
        "spec_sha256": sha256_file(SPEC),
        "design_sha256": sha256_file(DESIGN),
        "spec_registry_sha256": dynamic_registry_digest(),
        "registered_config_id": CONFIG_ID,
        "generator": {
            "version": "d2-generator-v1",
            "rng_namespace": "dynamic-lifecycle-mainline-v2/d2-generator-v1",
            "same_distribution_as_d1": True,
            "scenario_index_start": START_INDEX,
            "scenario_index_stop_exclusive": START_INDEX + SCENARIOS_PER_CELL,
            "no_d1_id_overlap": True,
        },
        "cells": list(cells),
        "cell_weights": {cell: 1.0 / len(cells) for cell in cells},
        "scenarios_per_cell": SCENARIOS_PER_CELL,
        "scenario_count": len(scenario_ids),
        "scenario_ids": list(scenario_ids),
        "scenario_cells": scenario_cells,
        "methods": list(D1_METHODS),
        "expected_record_count": len(expected),
        "expected_rectangle": [list(key) for key in expected],
        "primary_contrast": "P-B1m",
        "secondary_holm_family": ["P-B2", "P-B3", "P-B4", "P-B5(4)", "P-B6"],
        "sensitivity_contrasts": ["P-B5(2)", "P-B5(8)"],
        "separate_exact_contrast": "P-CEX",
        "confirmation_rule": {
            "equal_cell_mean_at_least": 0.01,
            "bootstrap_ci_lower_above": 0.0,
            "complete_matrix": True,
            "zero_gates": True,
        },
        "bootstrap": {
            "iterations": 10000,
            "confidence_level": 0.95,
            "quantile_convention": "linear_type7",
            "namespace": "dynamic-lifecycle-mainline-v2/d2-bootstrap-v1",
            "resampling_unit": "paired_scenario_within_cell",
        },
        "sample_size_basis": {
            "d1_effect_mean_used": False,
            "d1_paired_variances_by_cell": PILOT_VARIANCES,
            "variance_inflation_factor": 2.0,
            "minimum_relevant_effect": 0.01,
            "alpha_two_sided": 0.05,
            "target_ci_detection_power": 0.90,
            "calculated_required_per_cell": 506,
            "rounded_frozen_per_cell": SCENARIOS_PER_CELL,
            "approximate_equal_cell_standard_error": standard_error,
            "boundary_note": "Joint success probability is about 0.5 when the true effect equals the point-estimate threshold 0.01; no sample size removes that boundary property.",
        },
        "canonical_workers": 22,
        "replay_audit": {
            "workers": [1, 2],
            "scenario_ids": list(replay_ids),
            "require_exact_record_and_event_equality": True,
        },
        "failure_policy": {
            "any_missing_duplicate_extra_failure_nan_gate_or_replay_mismatch": "FAILED_INCOMPLETE",
            "complete_case_confirmation_forbidden": True,
            "automatic_retry_forbidden": True,
            "seed_replacement_forbidden": True,
        },
        "implementation_status": "DESIGN_ONLY_NOT_IMPLEMENTED",
        "execution_authorized": False,
        "d2_authorized": False,
    }
    manifest["manifest_digest"] = canonical_digest(manifest)
    return manifest


if __name__ == "__main__":
    write_json_atomic(OUTPUT, build_manifest())
    print(OUTPUT)
