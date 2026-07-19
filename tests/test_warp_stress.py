import math
import json
import subprocess
import sys

from experiments.run_warp_stress import summarize
import numpy as np

from uav_lifecycle.second_batch import tier0_instances
from uav_lifecycle.warp_stress import (
    calibration_bases,
    calibration_cases,
    confirmation_cases,
    confirmation_coverage,
    construct_competition,
    detect_rises,
    evaluate_stress_case,
    select_strata,
    SelectedStratum,
    stratum_key,
)


def witness_instance():
    return next(instance for instance in tier0_instances() if instance.instance_id == "warp_activation")


def test_detect_rise_recovers_audited_witness_numbers():
    rises = detect_rises(witness_instance())
    assert len(rises) == 1
    rise = rises[0]
    assert rise.target_task == 2
    np.testing.assert_allclose(rise.raw, 49.823433822724525)
    np.testing.assert_allclose(rise.warped, 49.5705720963674)
    np.testing.assert_allclose(rise.gap, rise.raw - rise.warped)


def test_competitor_distance_places_bid_inside_warp_interval():
    base = detect_rises(witness_instance())[0]
    case = construct_competition(base, rho=0.5, slack=0.25, instance_id="case")
    assert case is not None
    expected_bid = base.warped + 0.5 * base.gap
    np.testing.assert_allclose(case.competitor_bid, expected_bid)
    target = case.instance.tasks[base.target_task]
    competitor = case.instance.agents[1]
    distance = math.dist(competitor.origin, target.position)
    np.testing.assert_allclose(target.value * case.instance.discount**distance, expected_bid)
    assert base.warped < case.competitor_bid < base.raw


def test_competitor_is_rejected_if_large_slack_makes_another_task_preferred():
    base = detect_rises(witness_instance())[0]
    assert construct_competition(base, rho=0.5, slack=100.0, instance_id="bad") is None


def test_calibration_base_generation_is_deterministic_and_positive_gap():
    first = list(calibration_bases(limit=3))
    second = list(calibration_bases(limit=3))
    assert first == second
    assert len(first) == 3
    assert all(base.gap > 1e-9 for base in first)


def test_calibration_cases_use_only_frozen_rho_and_slack_grid():
    cases = list(calibration_cases(limit=10))
    assert len(cases) == 10
    assert all(case.rho in {0.1, 0.3, 0.5, 0.7, 0.9} for case in cases)
    assert all(case.slack in {0.05, 0.25, 0.75} for case in cases)
    assert len({case.case_id for case in cases}) == 10


def test_warp_competition_changes_target_winner():
    base = detect_rises(witness_instance())[0]
    case = construct_competition(base, rho=0.5, slack=0.25, instance_id="decisive")
    record = evaluate_stress_case(case)
    assert record["target_winner_changed"] is True
    assert record["allocation_changed"] is True
    assert record["warping_decisive"] is True
    assert record["johnson_gate_e_failures"] == 0
    np.testing.assert_allclose(record["predicted_delta_j"], -(1.0 - case.rho) * base.gap)
    np.testing.assert_allclose(record["delta_identity_error"], 0.0, atol=1e-12)


def test_stratum_boundaries_are_frozen():
    assert stratum_key({"discount": 0.97, "primary_deadline": 27, "relative_gap": 0.02, "slack": 0.25}) == (
        "low", "tight", "small", "small"
    )
    assert stratum_key({"discount": 0.98, "primary_deadline": 30, "relative_gap": 0.03, "slack": 0.75}) == (
        "high", "medium", "medium", "large"
    )
    assert stratum_key({"discount": 0.99, "primary_deadline": 36, "relative_gap": 0.08, "slack": 0.75}) == (
        "high", "loose", "large", "large"
    )


def test_stratum_selection_does_not_read_delta_j():
    records = []
    for index in range(30):
        records.append({
            "base_id": f"b{index}", "discount": 0.98, "primary_deadline": 30,
            "relative_gap": 0.03, "slack": 0.25, "allocation_changed": index < 20,
            "warping_decisive": index < 15, "johnson_gate_e_failures": 0, "delta_j": float(index),
        })
    first = select_strata(records, top_k=12, minimum_count=30)
    for record in records:
        record["delta_j"] *= -10_000
    second = select_strata(records, top_k=12, minimum_count=30)
    assert first == second
    assert len(first) == 1


def test_confirmation_split_is_disjoint_deterministic_and_in_selected_stratum():
    selected = (SelectedStratum(("high", "medium", "small", "small"), 30, 0.5, 0.5, 10),)
    first = confirmation_cases(selected, per_stratum=5, seed=7132026)
    second = confirmation_cases(selected, per_stratum=5, seed=7132026)
    assert first == second
    assert len(first) == 5
    for case in first:
        assert case.rho in {0.2, 0.4, 0.6, 0.8}
        assert case.rho not in {0.1, 0.3, 0.5, 0.7, 0.9}
        assert case.slack == 0.10
        assert case.instance.discount == 0.985
        assert case.instance.discount not in {0.96, 0.97, 0.98, 0.99}
        assert set(task.value for task in case.instance.tasks) <= {50.0, 70.0, 90.0, 110.0}
        record_stub = {
            "discount": case.instance.discount,
            "primary_deadline": case.instance.agents[0].deadline,
            "relative_gap": case.base.relative_gap,
            "slack": case.slack,
        }
        assert stratum_key(record_stub) == selected[0].key


def test_confirmation_coverage_reports_unfilled_strata():
    selected = (
        SelectedStratum(("high", "medium", "small", "small"), 30, 0.5, 0.5, 10),
        SelectedStratum(("low", "loose", "large", "large"), 30, 0.5, 0.5, 10),
    )
    cases = confirmation_cases(selected[:1], per_stratum=5, seed=7132026)
    coverage = confirmation_coverage(selected, cases, per_stratum=5)
    assert coverage["requested_count"] == 10
    assert coverage["generated_count"] == 5
    assert coverage["covered_strata"] == 1
    assert coverage["shortages"]["low|loose|large|large"] == 5


def test_stress_runner_smoke_writes_manifest(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.run_warp_stress", "--stage", "smoke", "--workers", "2", "--smoke-count", "20", "--output", str(tmp_path)],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((tmp_path / "stress_manifest.json").read_text(encoding="utf-8"))
    assert manifest["verdict"] == "PASS"
    assert manifest["stages"]["smoke"]["johnson_gate_e_failures"] == 0


def test_summary_reports_local_delta_identity_and_interaction_residual():
    base = {
        "gap": 1.0, "allocation_changed": True, "target_winner_changed": True,
        "warping_decisive": True, "johnson_gate_e_failures": 0,
    }
    records = [
        {**base, "delta_j": -0.5, "delta_identity_error": 0.0},
        {**base, "delta_j": -2.0, "delta_identity_error": -1.5},
    ]
    summary = summarize(records)
    assert summary["delta_identity_exact_rate"] == 0.5
    assert summary["interaction_residual_negative_count"] == 1
    assert summary["interaction_residual_min"] == -1.5
