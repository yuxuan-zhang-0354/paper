"""Freeze the post-D2 diagnostic plan and the non-executable D3 design."""

from __future__ import annotations

from hashlib import sha256
import json
from math import ceil, sqrt
from pathlib import Path
from typing import Any

from uav_lifecycle.artifacts import sha256_file, write_json_atomic
from uav_lifecycle.dynamic_scenarios import D1_CELLS


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "docs/superpowers/specs/2026-07-14-d3-external-validation-design.md"
DIAGNOSTIC_OUTPUT = ROOT / "results/dynamic_mainline/d2_diagnostics/design_manifest.json"
D3_OUTPUT = ROOT / "results/dynamic_mainline/d3_design/d3_manifest.json"
D2_ROOT = ROOT / "results/dynamic_mainline/d2_confirmation"
METHODS = ("P", "B1m", "B4", "B5(4)", "B6", "SCBBA", "DVCBBA")
CBBA_ISOLATION_METHODS = ("P", "SCBBA", "DVCBBA", "CEX")
ALLOCATION_PRESSURE_METHODS = ("P", "SCBBA", "DVCBBA", "B6")
D2_CELLS = tuple(sorted(cell.cell_id for cell in D1_CELLS))
SCALE_CELLS = ((4, 10), (6, 15), (8, 20), (6, 10), (6, 20), (6, 30), (8, 30))
MISMATCH_CONDITIONS = (
    "nominal",
    "sensor_m20", "sensor_m10", "sensor_p10", "sensor_p20",
    "attack_m20", "attack_m10", "attack_p10", "attack_p20",
)
WEIGHT_PROFILES = ("value_priority", "balanced", "resource_saving")
ALLOCATION_PRESSURE_CONDITIONS = (
    "reference", "shared_high_value", "target_clustered",
    "tight_resources", "long_routes", "combined_stress",
)


def canonical_digest(value: object) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")).hexdigest()


def _seal(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["manifest_digest"] = canonical_digest(manifest)
    return manifest


def build_d2_diagnostic_manifest() -> dict[str, Any]:
    sources = {
        name: {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(path),
        }
        for name, path in {
            "records": D2_ROOT / "canonical/records.csv",
            "public_events": D2_ROOT / "canonical/public_events.csv",
            "private_audit_events": D2_ROOT / "canonical/private_audit_events.csv",
            "summary": D2_ROOT / "summary.json",
        }.items()
    }
    return _seal({
        "manifest_version": "dynamic_lifecycle_d2_diagnostics_v1",
        "design_sha256": sha256_file(DESIGN),
        "status": "DESIGN_ONLY_NOT_EXECUTED",
        "source_is_frozen_d2": True,
        "source_manifest_digest": "1b45eea03bd606bf4a512f9e00e4bed48c0b984b24209002e92d7f9eb4baa5d9",
        "source_record_count": 40960,
        "sources": sources,
        "analysis_label": "post_hoc_explanatory_not_confirmatory",
        "no_algorithm_or_parameter_change": True,
        "artifact_only_analyses": {
            "method_profiles": [
                "normalized_utility", "destroyed_value", "service_cost",
                "distance_cost", "ammo_cost", "makespan", "final_joint_brier_score",
                "recon_count", "bda_count", "continuous_attack_count", "handoff_count",
                "replan_count", "cbba_round_count",
            ],
            "paired_contrasts": [
                "P-B1m", "P-B2", "P-B3", "P-B4", "P-B5(2)",
                "P-B5(4)", "P-B5(8)", "P-B6", "P-CEX",
            ],
            "paired_decomposition": (
                "delta_destroyed_value-delta_service_cost-"
                "delta_distance_cost-delta_ammo_cost"
            ),
            "distribution_outputs": [
                "mean", "median", "p05", "p25", "p75", "p95",
                "win_tie_loss", "ecdf", "boxplot_values",
            ],
        },
        "trace_replay": {
            "methods": ["P", "B4", "B5(2)", "B5(4)", "B5(8)"],
            "scenario_ids": "all 4096 frozen D2 scenario IDs",
            "acceptance": (
                "terminal records and public/private events must exactly match canonical D2"
            ),
            "captured_per_planning_call": [
                "tick", "trigger", "positive_pair_count", "proposals",
                "planned_paths", "path_scores", "cbba_rounds", "no_commit",
            ],
            "performance_results_are_not_reestimated": True,
        },
        "bda_diagnostics": {
            "comparison": "paired P-B4",
            "scenario_groups_by_p_bda_count": ["0", "1", "2_plus"],
            "event_fields": [
                "belief_before", "belief_after", "delta_probability_alive",
                "delta_entropy", "prior_attacks_same_target", "observation",
                "next_same_target_action", "resource_tier",
            ],
            "fixed_probability_bins": [0.0, 0.25, 0.5, 0.75, 1.0],
            "causal_language_forbidden": True,
        },
        "periodic_diagnostics": {
            "methods": ["B5(2)", "B5(4)", "B5(8)"],
            "outputs": [
                "completion_to_next_grid_wait", "action_counts", "replan_count",
                "cbba_round_count", "resource_consumption", "selected_path_score",
                "no_commit_count",
            ],
        },
        "bootstrap": {
            "iterations": 10000,
            "resampling": "paired scenario within cell",
            "quantile": "linear_type7",
            "confidence_level": 0.95,
        },
        "planned_tables": [
            "diagnostic_method_summary.csv", "diagnostic_paired_decomposition.csv",
            "diagnostic_bda_events.csv", "diagnostic_periodic_trace.csv",
        ],
        "execution_authorized": False,
    })


def _scale_resources(n: int, m: int) -> dict[str, float | int]:
    horizon = ceil(12.0 * sqrt(m / 5.0) + 8.0 * m / n)
    return {
        "arena_half_width": 6.0 * sqrt(m / 5.0),
        "ammo_per_agent": ceil(1.2 * m / n),
        "t_max": horizon,
        "range_per_agent": 1.25 * horizon,
    }


def build_d3_manifest() -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []

    for n, m in SCALE_CELLS:
        cell = f"N{n}-M{m}-Rscaled"
        for index in range(2000, 2096):
            scenarios.append({
                "scenario_id": f"D3S-{cell}-S{index:04d}",
                "suite": "scale", "condition": "scaled", "cell_id": cell,
                "index": index,
            })
    for cell in D2_CELLS:
        for index in range(3000, 3128):
            scenarios.append({
                "scenario_id": f"D3C-{cell}-S{index:04d}",
                "suite": "continuous_belief", "condition": "dirichlet_1111",
                "cell_id": cell, "index": index,
            })
    for condition in MISMATCH_CONDITIONS:
        for cell in D2_CELLS:
            for index in range(4000, 4064):
                scenarios.append({
                    "scenario_id": f"D3M-{condition}-{cell}-S{index:04d}",
                    "suite": "model_mismatch", "condition": condition,
                    "cell_id": cell, "index": index,
                })
    for profile in WEIGHT_PROFILES:
        for cell in D2_CELLS:
            for index in range(5000, 5064):
                scenarios.append({
                    "scenario_id": f"D3W-{profile}-{cell}-S{index:04d}",
                    "suite": "utility_profile", "condition": profile,
                    "cell_id": cell, "index": index,
                })

    for cell in D2_CELLS:
        for index in range(6000, 6064):
            scenarios.append({
                "scenario_id": f"D3A-{cell}-S{index:04d}",
                "suite": "cbba_isolation", "condition": "nominal",
                "cell_id": cell, "index": index,
            })

    pressure_cell = "N6-M15-allocation-pressure"
    for condition in ALLOCATION_PRESSURE_CONDITIONS:
        for index in range(7000, 7064):
            scenarios.append({
                "scenario_id": f"D3P-{condition}-{pressure_cell}-S{index:04d}",
                "suite": "allocation_pressure", "condition": condition,
                "cell_id": pressure_cell, "index": index,
            })

    rectangle = [
        [scenario["scenario_id"], method]
        for scenario in scenarios
        for method in (
            CBBA_ISOLATION_METHODS
            if scenario["suite"] == "cbba_isolation"
            else ALLOCATION_PRESSURE_METHODS
            if scenario["suite"] == "allocation_pressure"
            else METHODS
        )
    ]
    replay_ids = [
        scenario["scenario_id"] for scenario in scenarios
        if scenario["index"] in {2000, 3000, 4000, 5000, 6000, 7000}
    ]
    scale = {
        f"N{n}-M{m}-Rscaled": {
            "agents": n, "targets": m, **_scale_resources(n, m),
        }
        for n, m in SCALE_CELLS
    }
    return _seal({
        "manifest_version": "dynamic_lifecycle_d3_external_validation_v2",
        "design_sha256": sha256_file(DESIGN),
        "status": "DESIGN_ONLY_NOT_IMPLEMENTED",
        "primary_algorithm_frozen": "P",
        "source_d2_manifest_digest": "1b45eea03bd606bf4a512f9e00e4bed48c0b984b24209002e92d7f9eb4baa5d9",
        "methods": sorted(set(METHODS + CBBA_ISOLATION_METHODS + ALLOCATION_PRESSURE_METHODS)),
        "suite_methods": {
            "external_validation": list(METHODS),
            "cbba_isolation": list(CBBA_ISOLATION_METHODS),
            "allocation_pressure": list(ALLOCATION_PRESSURE_METHODS),
        },
        "cex_excluded_from_scalable_suites": True,
        "suites": {
            "scale": {
                "cells": scale,
                "scenarios_per_cell": 96,
                "index_range": [2000, 2096],
                "generator_namespace": "dynamic-lifecycle-mainline-v2/d3-scale-generator-v1",
                "coordinate_distribution": "uniform square",
                "belief_distribution": "four frozen D2 archetypes rotated by target and index",
                "truth_conditioned_on_belief": True,
                "density_rule": "arena_half_width=6*sqrt(M/5)",
                "resource_rules": {
                    "ammo_per_agent": "ceil(1.2*M/N)",
                    "t_max": "ceil(12*sqrt(M/5)+8*M/N)",
                    "range_per_agent": "1.25*t_max",
                },
                "design_axes": {
                    "constant_load_M_over_N_2.5": [[4, 10], [6, 15], [8, 20]],
                    "fixed_N_6_load_sweep": [[6, 10], [6, 20], [6, 30]],
                    "joint_stress": [[8, 30]],
                },
            },
            "continuous_belief": {
                "cells": list(D2_CELLS), "scenarios_per_cell": 128,
                "index_range": [3000, 3128],
                "generator_namespace": "dynamic-lifecycle-mainline-v2/d3-continuous-generator-v1",
                "belief_distribution": "Dirichlet(1,1,1,1) via normalized -log(U)",
                "truth_conditioned_on_generated_belief": True,
            },
            "model_mismatch": {
                "cells": list(D2_CELLS), "scenarios_per_condition_cell": 64,
                "index_range": [4000, 4064],
                "generator_namespace": "dynamic-lifecycle-mainline-v2/d3-mismatch-generator-v1",
                "conditions": list(MISMATCH_CONDITIONS),
                "belief_distribution": "four frozen D2 archetypes rotated by target and index",
                "planner_model": "frozen nominal D2 model for every condition",
                "environment_sensor_transform": {
                    "positive_q": "O_true=(1-q)*O_nominal+q*I",
                    "negative_q": "O_true=(1-|q|)*O_nominal+|q|*U; U columns=(0.5,0.5)",
                    "q_values": [-0.2, -0.1, 0.1, 0.2],
                    "applies_to": [
                        "recon_category_matrix", "recon_damage_matrix", "bda_damage_matrix",
                    ],
                },
                "environment_attack_transform": {
                    "formula": "p_true=clip(p_nominal*(1+delta),0,1)",
                    "delta_values": [-0.2, -0.1, 0.1, 0.2],
                },
                "one_factor_at_a_time": True,
                "cross_condition_crn": True,
                "crn_key_excludes_condition_but_scenario_id_includes_condition": True,
            },
            "utility_profile": {
                "cells": list(D2_CELLS), "scenarios_per_profile_cell": 64,
                "index_range": [5000, 5064],
                "generator_namespace": "dynamic-lifecycle-mainline-v2/d3-weight-generator-v1",
                "profiles": {
                    "value_priority": {
                        "service_cost_multiplier": 0.5,
                        "distance_cost_multiplier": 0.5,
                        "ammo_cost_multiplier": 0.5,
                    },
                    "balanced": {
                        "service_cost_multiplier": 1.0,
                        "distance_cost_multiplier": 1.0,
                        "ammo_cost_multiplier": 1.0,
                    },
                    "resource_saving": {
                        "service_cost_multiplier": 1.5,
                        "distance_cost_multiplier": 3.0,
                        "ammo_cost_multiplier": 4.0,
                    },
                },
                "planner_and_evaluator_use_same_profile": True,
                "belief_distribution": "four frozen D2 archetypes rotated by target and index",
                "physical_model_and_constraints_unchanged": True,
                "cross_profile_crn": True,
                "crn_key_excludes_profile_but_scenario_id_includes_profile": True,
            },
            "cbba_isolation": {
                "cells": list(D2_CELLS), "scenarios_per_cell": 64,
                "index_range": [6000, 6064],
                "generator_namespace": "dynamic-lifecycle-mainline-v2/d3-cbba-isolation-v1",
                "methods": list(CBBA_ISOLATION_METHODS),
                "purpose": "isolate allocator replacement on the same dynamic interface",
                "cex_scope": "small current-epoch allocation only",
                "baseline_nonconvergence_is_outcome_not_infrastructure_failure": True,
            },
            "allocation_pressure": {
                "cell": {"agents": 6, "targets": 15},
                "conditions": list(ALLOCATION_PRESSURE_CONDITIONS),
                "scenarios_per_condition": 64,
                "index_range": [7000, 7064],
                "generator_namespace": "dynamic-lifecycle-mainline-v2/d3-allocation-pressure-v1",
                "methods": list(ALLOCATION_PRESSURE_METHODS),
                "purpose": "test whether small low-conflict D2 cells compress allocator differences",
                "axes": {
                    "reference": "scaled N6-M15 geometry and resources",
                    "shared_high_value": "alive-high-value beliefs and centrally colocated agents",
                    "target_clustered": "two compact target clusters with dispersed agents",
                    "tight_resources": "reduced ammo, horizon, and range",
                    "long_routes": "expanded arena with feasible but strongly coupled routes",
                    "combined_stress": "shared-high-value demand plus clustering and tight resources",
                },
                "pre_frozen_no_effect_driven_tuning": True,
                "cex_excluded_due_to_search_scale": True,
            },
        },
        "scenario_count": len(scenarios),
        "expected_record_count": len(rectangle),
        "scenario_ids": [scenario["scenario_id"] for scenario in scenarios],
        "scenario_metadata": {scenario["scenario_id"]: scenario for scenario in scenarios},
        "expected_rectangle": rectangle,
        "analysis": {
            "label": "registered_external_validation",
            "d2_remains_only_primary_confirmation": True,
            "paired_contrasts": [
                "P-B1m", "P-B4", "P-B5(4)", "P-B6", "P-SCBBA", "P-DVCBBA",
            ],
            "bootstrap_iterations": 10000,
            "confidence_level": 0.95,
            "resampling": "paired scenario within registered stratum",
            "report_all_cells_conditions_and_profiles": True,
            "cell_or_condition_removal_forbidden": True,
            "formal_outcomes": [
                "normalized_utility", "destroyed_value", "service_cost",
                "distance_cost", "ammo_cost", "initially_alive_value_neutralized_ratio",
                "invalid_attack_rate", "makespan", "action_counts", "replan_count",
                "cbba_round_count", "planning_process_time_ns",
                "allocator_convergence_rate", "cycle_count", "round_cap_count",
                "winner_conflicts", "message_packets", "message_scalars",
                "allocation_objective", "positive_pair_density",
            ],
            "cbba_isolation_outcomes": [
                "fixed_screened_task_exact_gap", "all_mode_cex_gap",
                "final_normalized_utility",
            ],
            "allocation_pressure_interpretation": (
                "compare condition-wise gaps; no condition may be removed or retuned after effects are seen"
            ),
        },
        "implementation_requirements": {
            "separate_planner_and_environment_models": True,
            "sensor_observations_use_environment_model": True,
            "physical_attack_outcomes_use_environment_model": True,
            "belief_updates_and_attack_predictions_use_planner_model": True,
            "utility_profile_changes_no_physical_parameter": True,
            "algorithm_control_flow_unchanged": True,
        },
        "pre_execution_structural_smoke": {
            "namespace": "dynamic-lifecycle-mainline-v2/d3-structural-smoke-v1",
            "formal_scenarios_reused": False,
            "minimum_cases": [
                "one scenario per scale cell", "all mismatch transforms",
                "all utility profiles", "continuous belief simplex and truth calibration",
            ],
            "effect_statistics_forbidden": True,
            "checks": [
                "completion", "zero_gate", "finite_values", "action_region_coverage",
                "rng_key_separation", "cross_condition_crn", "runtime_feasibility",
            ],
            "failure_requires_new_manifest_version_before_formal_execution": True,
        },
        "runtime": {
            "planning_process_time_ns_per_call": True,
            "episode_planning_time_sum": True,
            "excluded_from_utility": True,
            "serial_benchmark": {
                "scenario_selection": "first and last index of every scale cell",
                "warmups": 1, "measured_repetitions": 3, "workers": 1,
                "summary": "median and p95 by method and scale cell",
            },
        },
        "replay_audit": {
            "workers": [1, 2], "scenario_ids": replay_ids,
            "canonical_workers": 22,
            "require_exact_record_and_event_equality": True,
        },
        "failure_policy": {
            "any_missing_duplicate_extra_failure_nan_gate_or_replay_mismatch": "FAILED_INCOMPLETE",
            "automatic_retry_forbidden": True,
            "seed_replacement_forbidden": True,
            "effect_based_tuning_forbidden": True,
        },
        "planned_outputs": [
            "d3_records.csv", "d3_public_events.csv", "d3_private_audit_events.csv",
            "d3_runtime.csv", "summary.json", "replay_summary", "artifact_inventory.json",
        ],
        "execution_authorized": False,
    })


def main() -> None:
    write_json_atomic(DIAGNOSTIC_OUTPUT, build_d2_diagnostic_manifest())
    write_json_atomic(D3_OUTPUT, build_d3_manifest())
    print(DIAGNOSTIC_OUTPUT)
    print(D3_OUTPUT)


if __name__ == "__main__":
    main()
