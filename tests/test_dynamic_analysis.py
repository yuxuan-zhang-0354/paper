"""Effect-blind calibration and exploratory D1 analysis tests."""

from __future__ import annotations

from dataclasses import fields, replace
import math

import pytest

from uav_lifecycle.dynamic_analysis import (
    CalibrationBatch,
    CalibrationCoverageRecord,
    MethodBlindDiagnostics,
    PublicTick0Coverage,
    paired_d1_summary,
    project_calibration_coverage,
    select_first_passing_calibration,
    summarize_effect_blind,
    validate_method_matrix,
    write_dynamic_verdict,
)
from uav_lifecycle.dynamic_planning import tick0_target_screening
from uav_lifecycle.dynamic_scenarios import (
    D1_CELLS,
    dynamic_config_registry,
    generate_d1_scenario,
)
from uav_lifecycle.dynamic_simulator import initialize_state


FORBIDDEN = {
    "method", "reward", "normalized_utility", "private_digest",
    "pairwise_difference", "win", "loss", "rank", "ci",
}


def _coverage() -> tuple[CalibrationCoverageRecord, ...]:
    return (
        CalibrationCoverageRecord("S0", "tight", ("R", "A", "B"), True, False, 0, 0, 0.1),
        CalibrationCoverageRecord("S1", "tight", ("Defer",), False, True, 0, 0, 0.1),
        CalibrationCoverageRecord("S2", "loose", ("R", "A", "B"), True, False, 0, 0, 0.1),
        CalibrationCoverageRecord("S3", "loose", ("Defer",), False, True, 0, 0, 0.1),
    )


def _batch(run_id="r0", records=None, d0_passed=True):
    selected = _coverage() if records is None else tuple(records)
    return CalibrationBatch(
        run_id, "cfg", tuple(row.scenario_id for row in selected),
        {row.scenario_id: len(row.initial_action_regions) for row in selected},
        selected, d0_passed,
    )


def test_calibration_record_is_structurally_blind_and_frozen() -> None:
    record = _coverage()[0]
    names = {field.name for field in fields(CalibrationCoverageRecord)}
    assert names.isdisjoint(FORBIDDEN)
    for name in FORBIDDEN:
        assert not hasattr(record, name)
    with pytest.raises(Exception):
        record.gate_count = 2


def test_projector_accepts_only_typed_public_coverage_and_diagnostics() -> None:
    public = tuple(
        PublicTick0Coverage(
            row.scenario_id, row.resource_tier, row.initial_action_regions,
            row.positive_single_task, row.resource_blocked,
        ) for row in _coverage()
    )
    diagnostics = {row.scenario_id: MethodBlindDiagnostics(0, 0, 0.1) for row in public}
    projected = project_calibration_coverage(
        public, diagnostics, tuple(row.scenario_id for row in public),
        {row.scenario_id: len(row.initial_action_regions) for row in public},
    )
    assert projected == _coverage()
    with pytest.raises(TypeError, match="PublicTick0Coverage"):
        project_calibration_coverage(({"method": "P"},), {}, ("S0",), {"S0": 1})


def test_projector_rejects_duplicate_or_missing_scenario() -> None:
    public = tuple(PublicTick0Coverage(
        row.scenario_id, row.resource_tier, row.initial_action_regions,
        row.positive_single_task, row.resource_blocked,
    ) for row in _coverage())
    diagnostics = {row.scenario_id: MethodBlindDiagnostics(0, 0, 0.1) for row in public}
    with pytest.raises(ValueError, match="duplicate"):
        project_calibration_coverage(public + public[:1], diagnostics, tuple(diagnostics), {row.scenario_id: len(row.initial_action_regions) for row in public})
    with pytest.raises(ValueError, match="missing"):
        project_calibration_coverage(public[:-1], diagnostics, tuple(diagnostics), {row.scenario_id: len(row.initial_action_regions) for row in public})


def test_selector_accepts_only_blind_batches() -> None:
    with pytest.raises(TypeError, match="CalibrationBatch"):
        select_first_passing_calibration(({"method": "P", "reward": 1.0},))
    assert select_first_passing_calibration((_batch(),)).selected_run_id == "r0"


@pytest.mark.parametrize("region", ["R", "A", "B", "Defer"])
def test_each_action_region_must_reach_one_percent(region) -> None:
    records = tuple(replace(row, initial_action_regions=tuple(
        "R" if item == region and region != "R" else "A" if item == "R" and region == "R" else item
        for item in row.initial_action_regions
    )) for row in _coverage())
    selection = select_first_passing_calibration((_batch(records=records),))
    assert selection.status == "NO_PASS_REQUIRES_USER_AUTHORIZATION"


@pytest.mark.parametrize("tier,field", [
    ("tight", "positive_single_task"), ("tight", "resource_blocked"),
    ("loose", "positive_single_task"), ("loose", "resource_blocked"),
])
def test_each_tier_requires_feasible_and_blocked_scenarios(tier, field) -> None:
    records = tuple(replace(row, **{field: False}) if row.resource_tier == tier else row for row in _coverage())
    assert select_first_passing_calibration((_batch(records=records),)).selected_run_id is None


def test_loose_resource_block_uses_cumulative_two_task_path() -> None:
    config_id, config = next(iter(dynamic_config_registry().items()))
    cell = next(
        item for item in D1_CELLS
        if item.agent_count == 2 and item.target_count == 5
        and item.resource_tier == "loose"
    )
    scenario = generate_d1_scenario(cell, 13, config_id)
    screening = tick0_target_screening(
        initialize_state(scenario).snapshot(), config, scenario.t_max_tick,
    )
    assert any(row.resource_blocked for row in screening)


def test_gate_or_numerical_failure_blocks_calibration() -> None:
    for field in ("gate_count", "numerical_failure_count"):
        records = (replace(_coverage()[0], **{field: 1}), *_coverage()[1:])
        assert select_first_passing_calibration((_batch(records=records),)).selected_run_id is None
    assert select_first_passing_calibration((_batch(d0_passed=False),)).selected_run_id is None


def test_first_passing_of_at_most_three_is_locked() -> None:
    failed = _batch("r0", d0_passed=False)
    selected = select_first_passing_calibration((failed, _batch("r1"), _batch("r2")))
    assert selected.selected_run_id == "r1"
    assert selected.status == "LOCK_FIRST_PASS"
    with pytest.raises(ValueError, match="at most three"):
        select_first_passing_calibration((failed, failed, failed, failed))


def test_effect_blind_summary_contains_no_method_effect_fields() -> None:
    summary = summarize_effect_blind(
        _coverage(), tuple(row.scenario_id for row in _coverage()),
        {row.scenario_id: len(row.initial_action_regions) for row in _coverage()},
    )
    assert not (set(summary) & FORBIDDEN)
    assert summary["passes_frozen_rule"] is True


METHODS = ("P", "B1m")


def _row(scenario, cell, method, value, **changes):
    row = {
        "scenario_id": scenario, "cell_id": cell, "method": method,
        "status": "complete", "normalized_utility": value,
        "realized_utility": value * 10, "gross_scenario_value": 10.0,
        "allocator_gates": [], "replay_audit": "match", "error_type": "",
        "initial_truth_digest": f"truth-{scenario}",
        "public_initial_digest": f"public-{scenario}",
    }
    row.update(changes)
    return row


def _matrix():
    rows = []
    expected = []
    for scenario, cell, p, b in (
        ("A0", "cell-a", 1.0, 0.0),
        ("B0", "cell-b", 0.0, 1.0),
        ("B1", "cell-b", 0.0, 1.0),
        ("B2", "cell-b", 0.0, 1.0),
    ):
        for method, value in (("P", p), ("B1m", b)):
            rows.append(_row(scenario, cell, method, value))
            expected.append((scenario, method))
    return tuple(rows), tuple(expected)


def _cell_args():
    return {
        "expected_cells": {"A0": "cell-a", "B0": "cell-b", "B1": "cell-b", "B2": "cell-b"},
        "required_cells": ("cell-a", "cell-b"),
    }


@pytest.mark.parametrize("fault", [
    "missing", "duplicate", "extra", "crash", "nan", "timeout", "gate", "replay",
    "not_checked", "cell", "truth", "public", "gross", "zero_gross", "normalized",
])
def test_any_matrix_fault_is_failed_incomplete(fault) -> None:
    records, expected = _matrix()
    records = list(records)
    if fault == "missing":
        records.pop()
    elif fault == "duplicate":
        records.append(dict(records[0]))
    elif fault == "extra":
        records.append(_row("EXTRA", "cell-a", "P", 0.0))
    elif fault == "crash":
        records[0]["status"] = "crash"
    elif fault == "nan":
        records[0]["normalized_utility"] = math.nan
    elif fault == "timeout":
        records[0].update(status="failed", error_type="TimeoutError")
    elif fault == "gate":
        records[0]["allocator_gates"] = [{"gate": "scheduler"}]
    elif fault == "replay":
        records[0]["replay_audit"] = "mismatch"
    elif fault == "not_checked":
        records[0]["replay_audit"] = "not_checked"
    elif fault == "cell":
        records[0]["cell_id"] = "wrong-cell"
    elif fault == "truth":
        records[0]["initial_truth_digest"] = "wrong-truth"
    elif fault == "public":
        records[0]["public_initial_digest"] = "wrong-public"
    elif fault == "gross":
        records[0]["gross_scenario_value"] = 11.0
    elif fault == "zero_gross":
        records[0]["gross_scenario_value"] = 0.0
    elif fault == "normalized":
        records[0]["normalized_utility"] += 0.1
    validation = validate_method_matrix(tuple(records), expected, **_cell_args())
    assert validation.status == "FAILED/INCOMPLETE"
    assert validation.reasons
    assert paired_d1_summary(tuple(records), expected, bootstrap_iterations=20, **_cell_args())["status"] == "FAILED/INCOMPLETE"


def test_complete_matrix_passes() -> None:
    records, expected = _matrix()
    validation = validate_method_matrix(records, expected, **_cell_args())
    assert validation.status == "COMPLETE"
    assert validation.gate_count == 0


def test_external_replay_gate_allows_not_checked_but_never_mismatch() -> None:
    records, expected = _matrix()
    records = [dict(row, replay_audit="not_checked") for row in records]
    assert validate_method_matrix(records, expected, replay_verified=True, **_cell_args()).status == "COMPLETE"
    records[0]["replay_audit"] = "mismatch"
    assert validate_method_matrix(records, expected, replay_verified=True, **_cell_args()).status == "FAILED/INCOMPLETE"
    assert validate_method_matrix(records, expected, replay_verified=False, **_cell_args()).status == "FAILED/INCOMPLETE"


def test_primary_effect_uses_equal_cell_not_episode_weighting() -> None:
    records, expected = _matrix()
    summary = paired_d1_summary(records, expected, bootstrap_iterations=100, **_cell_args())
    primary = summary["contrasts"]["P-B1m"]
    assert primary["mean"] == pytest.approx(0.0)
    assert primary["episode_weighted_mean"] == pytest.approx(-0.5)
    assert primary["label"] == "exploratory"


def test_bootstrap_is_deterministic_and_independently_keyed() -> None:
    records, expected = _matrix()
    first = paired_d1_summary(records, expected, bootstrap_iterations=10_000, manifest_digest="manifest-a", **_cell_args())
    replay = paired_d1_summary(records, expected, bootstrap_iterations=10_000, manifest_digest="manifest-a", **_cell_args())
    other = paired_d1_summary(records, expected, bootstrap_iterations=10_000, manifest_digest="manifest-b", **_cell_args())
    assert first == replay
    assert first["manifest_digest"] != other["manifest_digest"]
    assert first["quantile_convention"] == "linear_type7"
    assert "{cell}|{replicate}|{draw}" in first["bootstrap_key_template"]
    assert set(first["contrasts"]["P-B1m"]["delta_quantiles"]) == {"median", "p05", "p95"}


def test_d1_never_claims_confirmatory_success_and_reports_future_rule() -> None:
    records, expected = _matrix()
    summary = paired_d1_summary(records, expected, bootstrap_iterations=50, **_cell_args())
    assert summary["analysis_label"] == "exploratory"
    assert summary["d1_confirmatory_success"] is False
    assert summary["future_d2_rule"] == {
        "effect_at_least": 0.01, "ci_lower_above": 0.0,
        "complete_matrix": True, "zero_gates": True,
    }


def test_verdict_writer_labels_exploratory(tmp_path) -> None:
    records, expected = _matrix()
    summary = paired_d1_summary(records, expected, bootstrap_iterations=20, **_cell_args())
    path = tmp_path / "verdict.md"
    write_dynamic_verdict(path, summary)
    text = path.read_text(encoding="utf-8")
    assert "exploratory" in text.lower()
    assert "does not authorize D2" in text


def test_review_pools_all_m3_and_m5_target_regions() -> None:
    records = (
        CalibrationCoverageRecord(
            "S3", "tight", ("R", "A", "B"), True, True, 0, 0, 0.1,
        ),
        CalibrationCoverageRecord(
            "S5", "loose", ("Defer", "Defer", "R", "A", "B"),
            True, True, 0, 0, 0.2,
        ),
    )
    summary = summarize_effect_blind(records, ("S3", "S5"), {"S3": 3, "S5": 5})
    assert summary["public_scenario_count"] == 2
    assert summary["public_target_count"] == 8
    assert summary["action_region_shares"] == pytest.approx({
        "R": 2 / 8, "A": 2 / 8, "B": 2 / 8, "Defer": 2 / 8,
    })


def test_review_rejects_complete_keys_disguised_as_one_cell() -> None:
    records, expected = _matrix()
    disguised = tuple(dict(row, cell_id="cell-a") for row in records)
    expected_cells = {"A0": "cell-a", "B0": "cell-b", "B1": "cell-b", "B2": "cell-b"}
    validation = validate_method_matrix(
        disguised, expected, expected_cells=expected_cells,
        required_cells=("cell-a", "cell-b"),
    )
    assert validation.status == "FAILED/INCOMPLETE"
