"""Build and seal the frozen D4 sensitivity design."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.build_d3_manifests import canonical_digest
from uav_lifecycle.artifacts import write_json_atomic
from uav_lifecycle.dynamic_d4 import BATTLEFIELD_STRUCTURES, REACHABILITY_SCALES, WRECK_RATES


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/dynamic_mainline/d4_design/d4_manifest.json"
AUTHORIZATION = ROOT / "results/dynamic_mainline/d4_design/execution_authorization.json"
BATTLEFIELD_METHODS = ("P", "DVCBBA", "B6", "B4", "B1m", "SCBBA")
REACHABILITY_METHODS = ("P", "DVCBBA", "B6", "SCBBA")


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    value["manifest_digest"] = canonical_digest(value)
    return value


def build_d4_manifest() -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for structure in BATTLEFIELD_STRUCTURES:
        for wreck_rate in WRECK_RATES:
            rate = int(round(100.0 * wreck_rate))
            for seed in range(8000, 8064):
                scenario_id = f"D4A-{structure}-D{rate:02d}-N6-M20-D4-battlefield-S{seed:04d}"
                scenarios.append({
                    "scenario_id": scenario_id, "suite": "battlefield_structure",
                    "condition": f"{structure}/D{rate:02d}", "structure": structure,
                    "wreck_rate": wreck_rate, "cell_id": "N6-M20-D4-battlefield", "index": seed,
                })
    for map_scale in REACHABILITY_SCALES:
        for time_scale in REACHABILITY_SCALES:
            m, t = int(round(100.0 * map_scale)), int(round(100.0 * time_scale))
            for seed in range(9000, 9064):
                scenario_id = f"D4B-L{m:03d}-T{t:03d}-N6-M20-D4-reachability-S{seed:04d}"
                scenarios.append({
                    "scenario_id": scenario_id, "suite": "reachability",
                    "condition": f"L{m:03d}/T{t:03d}", "map_scale": map_scale,
                    "time_scale": time_scale, "cell_id": "N6-M20-D4-reachability", "index": seed,
                })
    rectangle = [
        [item["scenario_id"], method]
        for item in scenarios
        for method in (BATTLEFIELD_METHODS if item["suite"] == "battlefield_structure" else REACHABILITY_METHODS)
    ]
    replay_ids = [
        item["scenario_id"] for item in scenarios
        if item["index"] in {8000, 9000}
    ]
    return _seal({
        "manifest_version": "dynamic_lifecycle_d4_sensitivity_v1",
        "status": "FROZEN_AUTHORIZED_BY_USER_2026-07-15",
        "primary_algorithm_frozen": "P",
        "source_d3_manifest_digest": "e05d85e28cf62a081214001ff76f5bcba9ccf9cb627acdac5cb25bf91ae82aeb",
        "algorithm_or_hyperparameter_change": False,
        "methods": sorted(set(BATTLEFIELD_METHODS + REACHABILITY_METHODS)),
        "suite_methods": {
            "battlefield_structure": list(BATTLEFIELD_METHODS),
            "reachability": list(REACHABILITY_METHODS),
        },
        "suites": {
            "battlefield_structure": {
                "agents": 6, "targets": 20, "structures": list(BATTLEFIELD_STRUCTURES),
                "wreck_rates": list(WRECK_RATES), "scenarios_per_condition": 64,
                "index_range": [8000, 8064], "matched_prior": True,
                "belief_rule": "P(C|spatial stratum)*P(S|wreck rate); truth never exposed",
                "value_correlated_rule": {
                    "P(cluster_stratum)": 0.5, "P(H|cluster_stratum)": 0.65,
                    "P(H|dispersed_stratum)": 0.15, "marginal_P(H)": 0.4,
                },
                "mixed_cluster_probability": 0.6,
                "cluster_sigma_over_map_side": 0.08,
                "nested_wreck_crn": "D iff common U_damage < wreck_rate",
            },
            "reachability": {
                "agents": 6, "targets": 20, "map_scales": list(REACHABILITY_SCALES),
                "time_scales": list(REACHABILITY_SCALES), "scenarios_per_condition": 64,
                "index_range": [9000, 9064], "structure": "uniform", "wreck_rate": 0.2,
                "base_half_width": 12.0, "base_horizon": 51,
                "range_rule": "1.25*horizon", "ammo_per_agent": 4,
            },
        },
        "scenario_count": len(scenarios), "expected_record_count": len(rectangle),
        "scenario_ids": [item["scenario_id"] for item in scenarios],
        "scenario_metadata": {item["scenario_id"]: item for item in scenarios},
        "expected_rectangle": rectangle,
        "analysis": {
            "paired_within_condition": True, "bootstrap_iterations": 10000,
            "confidence_level": 0.95, "all_conditions_reported": True,
            "effect_driven_retuning_forbidden": True,
            "primary_contrasts": ["P-DVCBBA", "P-B6", "P-B4", "P-B1m", "P-SCBBA"],
            "figures_generated_during_run": False,
        },
        "replay_audit": {"workers": [1, 2], "scenario_ids": replay_ids, "require_exact_equality": True},
        "failure_policy": {
            "any_missing_duplicate_extra_failure_nan_gate_or_replay_mismatch": "FAILED_INCOMPLETE",
            "automatic_retry_forbidden": True, "seed_replacement_forbidden": True,
        },
        "execution_authorized": False,
    })


def main() -> None:
    manifest = build_d4_manifest()
    write_json_atomic(OUTPUT, manifest)
    write_json_atomic(AUTHORIZATION, {
        "authorized": True,
        "authorized_by": "user_explicit_consent_in_codex_thread",
        "authorized_date": "2026-07-15",
        "manifest_digest": manifest["manifest_digest"],
    })
    print(OUTPUT)
    print(manifest["manifest_digest"])


if __name__ == "__main__":
    main()
