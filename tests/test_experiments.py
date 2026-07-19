import csv
import json

from experiments.run_belief_sweep import (
    CSV_COLUMNS,
    audit_decision_csv,
    evaluate_config,
    run_belief_sweep,
)
from experiments.run_properties import run_property_checks
from uav_lifecycle.scenarios import validation_parameter_sets
from uav_lifecycle.simplex import simplex_grid


def test_small_property_check_has_zero_failures():
    summary = run_property_checks(sample_count=50, seed=7132026)
    assert summary["sample_count"] == 50
    assert summary["total_failures"] == 0
    assert summary["gate_a_pass"] is True
    assert summary["gate_b_pass"] is True


def test_config_evaluation_is_deterministic_and_schema_complete():
    config = validation_parameter_sets()[0]
    grid = simplex_grid(0.5)
    first = evaluate_config(config, grid)
    second = evaluate_config(config, grid)
    assert first == second
    assert len(first) == 10
    assert all(len(row) == len(CSV_COLUMNS) for row in first)


def test_serial_sweep_writes_expected_record_count(tmp_path):
    output = tmp_path / "sweep"
    configs = validation_parameter_sets()[:2]
    summary = run_belief_sweep(
        step=0.5,
        output=output,
        workers=1,
        configs=configs,
    )
    assert summary["actual_record_count"] == 20
    with (output / "decision_regions.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        assert sum(1 for _ in csv.DictReader(source)) == 20
    counts = json.loads((output / "action_counts.json").read_text(encoding="utf-8"))
    assert set(counts["by_config"]) == {config.config_id for config in configs}
    assert summary["persisted_audit"]["integrity_pass"] is True


def test_persisted_audit_rejects_equal_length_duplicate_grid_row(tmp_path):
    output = tmp_path / "tampered"
    config = validation_parameter_sets()[0]
    grid = simplex_grid(0.5)
    run_belief_sweep(0.5, output, workers=1, configs=(config,))
    csv_path = output / "decision_regions.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[2] = lines[1]
    csv_path.write_text("".join(lines), encoding="utf-8", newline="")
    audit = audit_decision_csv(csv_path, (config,), grid)
    assert audit["row_count"] == 10
    assert audit["sequence_mismatches"] > 0
    assert audit["integrity_pass"] is False
