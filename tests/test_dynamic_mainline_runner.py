"""High-value tests for the small D0/D1 research runner."""

from __future__ import annotations

import csv
from dataclasses import fields
import json
from pathlib import Path

import pytest

import experiments.run_dynamic_mainline as runner
from experiments.run_dynamic_mainline import (
    ALLOWED_STAGES,
    D1_METHODS,
    WorkerInput,
    build_d1_manifest,
    run_dynamic_mainline,
)
from uav_lifecycle.dynamic_scenarios import D1_CELLS, generate_d1_scenario


CONFIG_ID = "recon_damage_plus_010_r2_a6_b3"


def _scenarios(count: int = 2):
    return tuple(
        generate_d1_scenario(D1_CELLS[0], index, CONFIG_ID)
        for index in range(count)
    )


def _one_per_cell():
    return tuple(generate_d1_scenario(cell, 0, CONFIG_ID) for cell in D1_CELLS)


def _sealed(root: Path, run_id: str = "r0") -> Path:
    return root / "calibration" / f"run_{run_id}" / "sealed"


def test_runner_stays_small_and_has_no_service_hardening() -> None:
    assert len(Path(runner.__file__).read_text(encoding="utf-8").splitlines()) < 530
    assert len(Path(__file__).read_text(encoding="utf-8").splitlines()) < 350
    for removed in (
        "_terminate_executor_processes",
        "_test_hard_exit_worker",
        "_test_blocking_worker",
        "_validate_public_private_separation",
        "PublicEventProjection",
    ):
        assert not hasattr(runner, removed)


def test_frozen_manifest_is_160_by_10_and_run_id_is_metadata_only() -> None:
    manifest, scenarios = build_d1_manifest(CONFIG_ID, 1, "round-a")
    other, _ = build_d1_manifest(CONFIG_ID, 1, "round-b")
    assert len(scenarios) == 160
    assert D1_METHODS == (
        "P", "B1m", "B2", "B3", "B4", "B5(4)", "B5(2)", "B5(8)", "B6", "CEX",
    )
    assert len(manifest["expected_rectangle"]) == 1600
    assert len({tuple(key) for key in manifest["expected_rectangle"]}) == 1600
    assert manifest["generator_digest"] == other["generator_digest"]
    assert "round-a" not in json.dumps(manifest["crn_contract"])
    assert manifest["d2_authorized"] is False
    assert [field.name for field in fields(WorkerInput)] == ["scenario", "config", "method"]


@pytest.mark.parametrize("stage", ["d2", "unknown"])
def test_rejects_d2_and_unknown_stage(stage, tmp_path) -> None:
    with pytest.raises(ValueError, match="stage"):
        run_dynamic_mainline(stage, tmp_path, 1, "r0", CONFIG_ID)
    assert ALLOWED_STAGES == {"d0", "d1-calibrate", "d1-reveal", "all-d0-d1"}


@pytest.mark.parametrize("workers", [0, 23])
def test_rejects_workers_outside_frozen_range(workers, tmp_path) -> None:
    with pytest.raises(ValueError, match=r"\[1, 22\]"):
        run_dynamic_mainline("d1-calibrate", tmp_path, workers, "r0", CONFIG_ID)


def test_manifest_rejects_duplicate_scenario_or_method() -> None:
    scenarios = _scenarios()
    with pytest.raises(ValueError, match="duplicate scenario"):
        build_d1_manifest(
            CONFIG_ID, 1, "r0", scenarios=scenarios + scenarios[:1], methods=("P",),
        )
    with pytest.raises(ValueError, match="duplicate method"):
        build_d1_manifest(
            CONFIG_ID, 1, "r0", scenarios=scenarios, methods=("P", "P"),
        )


def test_serial_parallel_and_upper_smoke_have_identical_algorithm_artifacts(tmp_path) -> None:
    scenarios = _scenarios()
    methods = ("P", "B6")
    summaries = {}
    for name, workers, selected in (
        ("serial", 1, scenarios),
        ("parallel", 2, scenarios),
        ("upper", 22, scenarios[:1]),
    ):
        summaries[name] = run_dynamic_mainline(
            "d1-calibrate", tmp_path / name, workers, "r0", CONFIG_ID,
            scenarios=selected, methods=methods,
        )
    assert summaries["serial"]["algorithm_digest"] == summaries["parallel"]["algorithm_digest"]
    assert summaries["serial"]["record_count"] == summaries["serial"]["terminal_count"] == 4
    assert summaries["serial"]["failure_count"] == summaries["serial"]["gate_count"] == 0
    assert summaries["upper"]["record_count"] == 2
    for filename in ("records.csv", "public_events.csv", "private_audit_events.csv", "dynamic_verdict.md"):
        assert (_sealed(tmp_path / "serial") / filename).read_bytes() == (
            _sealed(tmp_path / "parallel") / filename
        ).read_bytes()


def test_public_and_private_csv_use_fixed_physical_schemas(tmp_path) -> None:
    run_dynamic_mainline(
        "d1-calibrate", tmp_path, 1, "r0", CONFIG_ID,
        scenarios=_scenarios(1), methods=("P",),
    )
    with (_sealed(tmp_path) / "public_events.csv").open(encoding="utf-8", newline="") as source:
        public_columns = csv.DictReader(source).fieldnames
    with (_sealed(tmp_path) / "private_audit_events.csv").open(encoding="utf-8", newline="") as source:
        private_columns = csv.DictReader(source).fieldnames
    assert public_columns == list(runner.PUBLIC_COLUMNS)
    assert private_columns == list(runner.PRIVATE_COLUMNS)
    assert set(public_columns).isdisjoint({
        "draw", "true_category", "damage_before", "damage_after",
        "physical_success", "realized_reward", "counter_key",
    })


def test_policy_exception_becomes_one_terminal_failure_row(tmp_path, monkeypatch) -> None:
    def fail_policy(method, config):
        del method, config
        raise RuntimeError("synthetic policy failure")

    monkeypatch.setattr(runner, "make_policy", fail_policy)
    summary = run_dynamic_mainline(
        "d1-calibrate", tmp_path, 1, "r0", CONFIG_ID,
        scenarios=_scenarios(1), methods=("P",),
    )
    with (_sealed(tmp_path) / "records.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["error_type"] == "RuntimeError"
    assert summary["terminal_count"] == summary["failure_count"] == 1
    assert summary["status"] == "FAILED/INCOMPLETE"


def test_manifest_cannot_be_reused_with_different_rectangle(tmp_path) -> None:
    run_dynamic_mainline(
        "d1-calibrate", tmp_path, 1, "r0", CONFIG_ID,
        scenarios=_scenarios(1), methods=("P",),
    )
    with pytest.raises(RuntimeError, match="manifest"):
        run_dynamic_mainline(
            "d1-calibrate", tmp_path, 1, "r0", CONFIG_ID,
            scenarios=_scenarios(1), methods=("B6",),
        )


def test_reveal_only_accepts_passing_digest_locked_calibration(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="locked calibration"):
        run_dynamic_mainline("d1-reveal", tmp_path, 1, "r0", CONFIG_ID)
    run_dynamic_mainline(
        "d1-calibrate", tmp_path, 1, "r0", CONFIG_ID,
        scenarios=_scenarios(1), methods=("P",),
    )
    manifest = json.loads((_sealed(tmp_path) / "dynamic_manifest.json").read_text(encoding="utf-8"))
    lock = tmp_path / "calibration" / "run_r0" / "calibration_lock.json"
    lock.write_text(json.dumps({
        "locked": True, "status": "passed", "manifest_digest": manifest["manifest_digest"],
    }), encoding="utf-8")
    reveal = run_dynamic_mainline("d1-reveal", tmp_path, 1, "r0", CONFIG_ID)
    assert reveal == {
        "stage": "d1-reveal", "status": "LOCK_VERIFIED",
        "analysis_performed": False, "manifest_digest": manifest["manifest_digest"],
        "d2_authorized": False,
    }


def test_parent_writes_blind_coverage_and_complete_artifacts_atomically(tmp_path) -> None:
    run_dynamic_mainline(
        "d1-calibrate", tmp_path, 1, "r0", CONFIG_ID,
        scenarios=_scenarios(1), methods=("P",),
    )
    assert {path.name for path in _sealed(tmp_path).iterdir()} == {
        "dynamic_manifest.json", "records.csv", "public_events.csv",
        "private_audit_events.csv", "summary.json", "dynamic_verdict.md",
        "public_coverage.json",
    }
    coverage = json.loads((_sealed(tmp_path) / "public_coverage.json").read_text())
    assert coverage["summary"]["passes_frozen_rule"] is False
    assert coverage["summary"]["public_scenario_count"] == 1
    assert coverage["summary"]["public_target_count"] == 3
    assert len(coverage["records"][0]["initial_action_regions"]) == 3
    assert not any(name in json.dumps(coverage) for name in ("method", "normalized_utility", "reward"))
    assert not list(_sealed(tmp_path).glob("*.tmp"))


def test_reveal_writes_exploratory_summary_after_external_replay_lock(tmp_path) -> None:
    run_dynamic_mainline(
        "d1-calibrate", tmp_path, 1, "r0", CONFIG_ID,
        scenarios=_one_per_cell(), methods=("P", "B1m"),
    )
    records_path = _sealed(tmp_path) / "records.csv"
    with records_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
        columns = tuple(rows[0])
    for row in rows:
        row["status"], row["allocator_gates"] = "complete", "[]"
    with records_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = json.loads((_sealed(tmp_path) / "dynamic_manifest.json").read_text())
    lock = tmp_path / "calibration" / "run_r0" / "calibration_lock.json"
    lock.write_text(json.dumps({
        "locked": True, "status": "passed", "replay_verified": True,
        "manifest_digest": manifest["manifest_digest"],
    }), encoding="utf-8")
    reveal = run_dynamic_mainline("d1-reveal", tmp_path, 1, "r0", CONFIG_ID)
    summary = json.loads((tmp_path / "d1_pilot/d1_exploratory_summary.json").read_text())
    assert reveal["analysis_performed"] is True
    assert summary["status"] == "COMPLETE"
    assert summary["replay_evidence"] == "external_worker_replay"
    assert summary["d1_confirmatory_success"] is summary["d2_authorized"] is False
