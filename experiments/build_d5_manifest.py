"""Build the registered D5 raw/warped x retain/rebuild design."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from uav_lifecycle.artifacts import sha256_file, write_json_atomic
from uav_lifecycle.dynamic_d3 import ALLOCATION_PRESSURE_CONDITIONS, D3_SCALE_CELLS


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "docs/superpowers/specs/2026-07-17-d5-factorial-ablation-design.md"
OUTPUT = ROOT / "results/dynamic_mainline/d5_factorial_ablation/design/d5_manifest.json"
METHODS = ("V00", "V01", "V10", "V11")


def _digest(value: object) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")).hexdigest()


def build_manifest() -> dict[str, object]:
    scenarios = []
    for condition in ALLOCATION_PRESSURE_CONDITIONS:
        for index in range(9000, 9128):
            scenarios.append({
                "scenario_id": f"D5P-{condition}-N6-M15-allocation-pressure-S{index:04d}",
                "suite": "pressure", "condition": condition,
                "cell_id": "N6-M15-allocation-pressure", "index": index,
            })
    for cell in D3_SCALE_CELLS:
        for index in range(8000, 8064):
            scenarios.append({
                "scenario_id": f"D5S-{cell.cell_id}-S{index:04d}",
                "suite": "scale", "condition": "scaled",
                "cell_id": cell.cell_id, "index": index,
            })
    rectangle = [[row["scenario_id"], method] for row in scenarios for method in METHODS]
    replay_ids = [
        row["scenario_id"] for row in scenarios
        if row["index"] in {8000, 8063, 9000, 9127}
    ]
    manifest: dict[str, object] = {
        "manifest_version": "dynamic_lifecycle_d5_factorial_v1",
        "design_sha256": sha256_file(DESIGN),
        "status": "DESIGN_ONLY_NOT_EXECUTED",
        "methods": list(METHODS),
        "factor_levels": {
            "V00": ["raw", "retain_release"],
            "V01": ["raw", "full_reconstruction"],
            "V10": ["warped", "retain_release"],
            "V11": ["warped", "full_reconstruction"],
        },
        "scenario_count": len(scenarios),
        "expected_record_count": len(rectangle),
        "scenario_ids": [row["scenario_id"] for row in scenarios],
        "scenario_metadata": {row["scenario_id"]: row for row in scenarios},
        "expected_rectangle": rectangle,
        "fresh_namespaces": ["d5-factorial-pressure-v1", "d5-factorial-scale-v1"],
        "formal_seed_ranges": {"pressure": [9000, 9128], "scale": [8000, 8064]},
        "common_random_numbers": True,
        "algorithm_and_parameters_frozen": True,
        "analysis": {
            "primary": [
                "cycle_or_round_cap", "legal_commit_rate", "allocation_stall_rate",
                "normalized_utility",
            ],
            "secondary": [
                "winner_conflicts", "rounds_per_successful_commit",
                "message_packets_per_successful_commit", "planning_time_per_successful_commit",
                "warping_activations", "raw_prefix_increases",
            ],
            "bootstrap_iterations": 10000,
            "resampling_unit": "paired scenario within registered stratum",
            "failed_epoch_objective_is_diagnostic_only": True,
            "report_all_strata": True,
        },
        "replay_audit": {
            "workers": [1, 2], "canonical_workers": 22,
            "scenario_ids": replay_ids,
            "require_exact_record_and_event_equality": True,
        },
        "failure_policy": {
            "missing_duplicate_extra_failure_nan_gate_or_replay_mismatch": "FAILED_INCOMPLETE",
            "retry_seed_replacement_effect_tuning": "FORBIDDEN",
        },
    }
    manifest["manifest_digest"] = _digest(manifest)
    return manifest


def main() -> None:
    write_json_atomic(OUTPUT, build_manifest())
    print(OUTPUT)


if __name__ == "__main__":
    main()
