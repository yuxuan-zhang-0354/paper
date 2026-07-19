import csv
import json
from dataclasses import replace

import pytest
import experiments.run_ranked_fallback as fallback_runner

from uav_lifecycle.third_batch import (
    evaluate_fallback_instance,
    evaluate_mode_instance,
    fallback_gate_telemetry,
    random_mode_instance,
    tier0_mode_instances,
)
from uav_lifecycle.mode_allocation import Mode, ModeAgent, ModeInstance, ModeTask
from uav_lifecycle.mode_fallback import (
    finalize_fallback_attempts,
    rank_mode_candidates,
    run_ranked_fallback,
)
from experiments.run_third_batch import _parse_cells, _summarize, run_third_batch
from experiments.run_ranked_fallback import (
    _f0_hard_gate_status,
    _f2_decision,
    _f2_summary,
    _compare_f0_replay,
    _replay_manifest_value,
    _runner_exit_code,
    _validate_f0_prerequisite,
    build_ranked_fallback_jobs,
    rewrite_existing_fallback_artifacts,
    run_ranked_fallback_stage,
)


def test_tier0_has_eight_distinct_registered_witnesses():
    instances = tier0_mode_instances()
    labels = {instance.instance_id for instance in instances}
    assert len(instances) >= 8
    assert len(labels) == len(instances)
    assert {
        "recon_selected",
        "attack_selected",
        "bda_selected",
        "defer",
        "shared_witness_ammo",
        "horizon_conflict",
        "range_conflict",
        "mode_substitution",
    } <= labels


def test_random_mode_instance_replays_exactly_from_seed():
    first = random_mode_instance(2, 3, 17, "tight", "medium", "optimistic")
    second = random_mode_instance(2, 3, 17, "tight", "medium", "optimistic")
    assert first == second


def test_stratified_profile_covers_registered_intrinsic_action_regions():
    instance = random_mode_instance(
        2, 4, 3, "loose", "loose", "optimistic", belief_profile="stratified"
    )
    best = []
    for tasks in instance.tasks_by_target:
        values = {task.mode.value: task.utility for task in tasks}
        values["defer"] = 0.0
        best.append(max(values, key=values.get))
    assert set(best) == {"recon", "attack", "bda", "defer"}


@pytest.mark.parametrize("instance", tier0_mode_instances(), ids=lambda x: x.instance_id)
def test_tier0_loss_decomposition_and_gates(instance):
    record = evaluate_mode_instance(instance)
    assert record["all_mode_score"] >= record["fixed_mode_score"] - 1e-9
    assert record["screening_loss"] == pytest.approx(
        record["all_mode_score"] - record["fixed_mode_score"]
    )
    assert record["allocation_loss"] == pytest.approx(
        record["fixed_mode_score"] - record["full_raw_score"]
    )
    assert record["warping_loss"] == pytest.approx(
        record["full_raw_score"] - record["johnson_score"]
    )
    assert record["johnson_gate_failures"] == 0


def test_mode_substitution_witness_has_positive_screening_gap():
    instance = next(i for i in tier0_mode_instances() if i.instance_id == "mode_substitution")
    record = evaluate_mode_instance(instance)
    assert record["mode_substitutions"] >= 1
    assert record["screening_loss"] > 0


def test_tier0_runner_writes_auditable_manifest_and_records(tmp_path):
    summary = run_third_batch("tier0", tmp_path, workers=1)
    manifest = json.loads((tmp_path / "third_batch_manifest.json").read_text(encoding="utf-8"))
    with (tmp_path / "iteration0" / "records.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert summary["gate_g1_pass"]
    assert summary["gate_g2_pass"]
    assert manifest["stage"] == "tier0"
    assert manifest["record_count"] == len(tier0_mode_instances()) == len(rows)


def test_full_raw_cycle_is_reported_but_does_not_fail_johnson_gate_g2():
    record = {
        "all_mode_score": 10.0,
        "fixed_mode_score": 9.0,
        "full_raw_score": 8.0,
        "johnson_score": 8.0,
        "screening_loss": 1.0,
        "allocation_loss": 1.0,
        "warping_loss": 0.0,
        "johnson_ratio": 0.8,
        "screened_task_count": 1,
        "orphan_count": 0,
        "central_modes": ["attack"],
        "mode_substitutions": 0,
        "full_raw_gate_failures": 1,
        "johnson_gate_failures": 0,
        "full_raw_status": "cycle",
        "decomposition_valid": False,
    }
    summary = _summarize([record])
    assert summary["gate_g2_pass"]
    assert summary["full_raw_nonconvergence_rate"] == 1.0
    assert summary["decomposition_valid_rate"] == 0.0
    assert summary["mean_allocation_loss"] is None
    assert summary["mean_warping_loss"] is None


def test_pilot_runner_records_stratified_calibration_profile(tmp_path):
    run_third_batch(
        "pilot",
        tmp_path,
        workers=1,
        per_cell=1,
        cells=((2, 3),),
        belief_profile="stratified",
        seed_start=100,
    )
    manifest = json.loads((tmp_path / "third_batch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["belief_profile"] == "stratified"
    assert manifest["seed_start"] == 100
    assert manifest["record_count"] == 8 + 27


def test_cli_cell_parser_is_deterministic():
    assert _parse_cells("2x4,3x4,3x5") == ((2, 4), (3, 4), (3, 5))
    with pytest.raises(ValueError):
        _parse_cells("2x6")


def _fallback_resolution_witness():
    def task(target_id, mode, utility, ammo=0):
        return ModeTask(target_id, mode, (0, 0), 0, ammo, utility)

    return ModeInstance(
        agents=(ModeAgent(0, (0, 0), 10, 10, 1),),
        tasks_by_target=(
            (task(0, Mode.ATTACK, 10, ammo=1),),
            (task(1, Mode.ATTACK, 9, ammo=1), task(1, Mode.RECON, 8)),
        ),
        beta=0,
        distance_cost=0,
        ammo_cost=0,
        instance_id="fallback-resolution",
    )


def test_fallback_record_preserves_base_and_reports_set_differences():
    record = evaluate_fallback_instance(_fallback_resolution_witness())
    assert record["fallback_gain"] == pytest.approx(
        record["fallback_score"] - record["johnson_score"]
    )
    assert set(record["resolved_targets"]).isdisjoint(
        record["newly_unassigned_targets"]
    )
    assert record["fallback_score"] >= record["johnson_score"] - 1e-12
    assert record["screening_loss"] == pytest.approx(
        record["all_mode_score"] - record["fixed_mode_score"]
    )
    assert record["allocation_loss"] == pytest.approx(
        record["fixed_mode_score"] - record["full_raw_score"]
    )
    assert record["warping_loss"] == pytest.approx(
        record["full_raw_score"] - record["johnson_score"]
    )


def test_fallback_record_exports_per_gate_attempt_base_and_late_counts():
    record = evaluate_fallback_instance(_fallback_resolution_witness())
    gate_keys = (
        "winner_conflicts",
        "infeasible_paths",
        "bundle_path_mismatches",
        "warped_monotonicity_violations",
        "replay_mismatches",
        "cycle_or_timeout",
    )

    assert sum(int(record[f"fallback_{key}"]) for key in gate_keys) == int(
        record["fallback_gate_failures"]
    )
    assert sum(int(record[f"fallback_base_{key}"]) for key in gate_keys) == int(
        record["fallback_base_gate_failures"]
    )
    assert sum(int(record[f"fallback_late_{key}"]) for key in gate_keys) == int(
        record["fallback_late_gate_failures"]
    )


def test_fallback_per_gate_telemetry_preserves_nonzero_failure_classes():
    completed = run_ranked_fallback(_fallback_resolution_witness())
    base, resolved = completed.iterations
    failed = replace(
        resolved,
        legal=False,
        gate_report=(
            ("bundle_path_mismatches", 0),
            ("cycle_or_timeout", 2),
            ("infeasible_paths", 3),
            ("replay_mismatches", 0),
            ("warped_monotonicity_violations", 0),
            ("winner_conflicts", 0),
        ),
    )
    result = finalize_fallback_attempts(
        (base, failed), rank_mode_candidates(_fallback_resolution_witness())
    )

    telemetry = fallback_gate_telemetry(result)

    assert telemetry["fallback_cycle_or_timeout"] == 2
    assert telemetry["fallback_infeasible_paths"] == 3
    assert telemetry["fallback_gate_failures"] == 5
    assert telemetry["fallback_base_gate_failures"] == 0
    assert telemetry["fallback_late_gate_failures"] == 5


def test_fallback_per_instance_rates_are_none_for_empty_denominators():
    no_targets = ModeInstance(
        agents=(ModeAgent(0, (0, 0), 10, 10, 0),),
        tasks_by_target=(),
        beta=0,
        distance_cost=0,
        ammo_cost=0,
        instance_id="no-targets",
    )
    empty = evaluate_fallback_instance(no_targets)
    assert empty["base_orphan_rate"] is None
    assert empty["fallback_unresolved_rate"] is None
    assert empty["resolved_rate"] is None

    no_orphans = evaluate_fallback_instance(tier0_mode_instances()[0])
    assert no_orphans["base_orphan_count"] == 0
    assert no_orphans["resolved_rate"] is None


def test_fallback_summary_uses_corpus_count_denominators():
    records = [
        {
            "fallback_base_target_count": 1,
            "base_orphan_count": 1,
            "fallback_unresolved_count": 0,
            "resolved_count": 1,
            "fallback_gain": 1.0,
            "fallback_gate_failures": 0,
            "selected_mode_switches": 1,
            "selected_defer_count": 1,
            "search_advances": 2,
            "fallback_wall_clock_seconds": 0.25,
        },
        {
            "fallback_base_target_count": 9,
            "base_orphan_count": 0,
            "fallback_unresolved_count": 0,
            "resolved_count": 0,
            "fallback_gain": 0.0,
            "fallback_gate_failures": 1,
            "selected_mode_switches": 0,
            "selected_defer_count": 0,
            "search_advances": 0,
            "fallback_wall_clock_seconds": 0.75,
        },
    ]
    for record in records:
        record.update(
            {
                "all_mode_score": 1.0,
                "fixed_mode_score": 1.0,
                "screening_loss": 0.0,
                "johnson_gate_failures": 0,
                "full_raw_status": "converged",
                "screened_task_count": 1,
                "orphan_count": 0,
                "central_modes": ["recon"],
                "mode_substitutions": 0,
                "johnson_ratio": 1.0,
                "allocation_loss": 0.0,
                "warping_loss": 0.0,
            }
        )
    summary = _summarize(records)
    assert summary["base_orphan_rate"] == pytest.approx(0.1)
    assert summary["fallback_unresolved_rate"] == pytest.approx(0.0)
    assert summary["resolution_rate"] == pytest.approx(1.0)
    assert summary["fallback_win_count"] == 1
    assert summary["fallback_tie_count"] == 1
    assert summary["fallback_loss_count"] == 0
    assert summary["fallback_gate_failures"] == 1
    assert summary["selected_defer_count"] == 1


def test_fallback_tier_f0_writes_parent_owned_auditable_artifacts(tmp_path):
    summary = run_ranked_fallback_stage("tier_f0", tmp_path, workers=1)
    manifest = json.loads(
        (tmp_path / "fallback_manifest.json").read_text(encoding="utf-8")
    )
    with (tmp_path / "records.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))

    assert manifest["fallback_trigger"] == "base task orphan rate > 1%"
    assert manifest["ranked_fallback_implemented"] is True
    assert manifest["record_count"] == len(tier0_mode_instances()) == len(rows)
    assert manifest["parameters"]["belief_profile"] == "directed"
    assert manifest["tolerances"]["minimum_gain"] == -1e-9
    assert summary["fallback_gate_failures"] == 0
    assert summary["deterministic_replay_failures"] == 0
    assert summary["finite_iteration_bound_failures"] == 0
    assert manifest["status"] == "COMPLETE"
    assert manifest["deterministic_replay_failures"] == 0
    assert all(
        int(row["fallback_total_johnson_calls"])
        <= int(row["theoretical_call_bound"])
        == 1 + int(row["candidate_count"])
        for row in rows
    )
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "fallback_verdict.md").is_file()


def test_fallback_tier_f1_jobs_are_isolated_to_prior_holdout_seeds():
    jobs = build_ranked_fallback_jobs("tier_f1")
    metadata = [job[1] for job in jobs]

    assert len(jobs) == 3 * 3 * 3 * 3 * 20 == 1620
    assert {tuple(item["cell"]) for item in metadata} == {(2, 4), (3, 4), (3, 5)}
    assert {item["belief_profile"] for item in metadata} == {"stratified"}
    assert {item["seed"] for item in metadata} == set(range(1000, 1020))
    assert {item["stage"] for item in metadata} == {"tier_f1"}


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        ({"fallback_gate_failures": 1, "deterministic_replay_failures": 0, "finite_iteration_bound_failures": 0}, "STOPPED_GATE_FAILURE"),
        ({"fallback_gate_failures": 0, "deterministic_replay_failures": 1, "finite_iteration_bound_failures": 0}, "STOPPED_REPLAY_MISMATCH"),
        ({"fallback_gate_failures": 0, "deterministic_replay_failures": 0, "finite_iteration_bound_failures": 1}, "STOPPED_FINITE_BOUND_FAILURE"),
        ({"fallback_gate_failures": 0, "deterministic_replay_failures": 0, "finite_iteration_bound_failures": 0}, "COMPLETE"),
    ],
)
def test_f0_hard_gate_has_distinct_failure_statuses(summary, expected):
    assert _f0_hard_gate_status(summary) == expected


def test_tier_f1_requires_a_complete_passing_f0_manifest(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        _validate_f0_prerequisite(missing)
    with pytest.raises(FileNotFoundError):
        run_ranked_fallback_stage(
            "tier_f1",
            tmp_path / "output",
            workers=1,
            prerequisite_manifest=missing,
        )

    failed = tmp_path / "failed.json"
    failed.write_text(
        json.dumps(
            {
                "stage": "tier_f0",
                "status": "STOPPED_FINITE_BOUND_FAILURE",
                "summary": {
                    "fallback_gate_failures": 0,
                    "deterministic_replay_failures": 0,
                    "finite_iteration_bound_failures": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="passing Tier F0"):
        _validate_f0_prerequisite(failed)


def test_replay_audit_is_explicitly_not_run_for_standalone_tier_f1():
    assert _replay_manifest_value("tier_f1", 0) is None
    assert _replay_manifest_value("tier_f0", 0) == 0


def test_f2_does_not_invent_an_effect_threshold_after_observing_one_resolution():
    decision = _f2_decision(
        "tier_f1",
        completed=True,
        summary={
            "fallback_gate_failures": 0,
            "minimum_fallback_gain": 0.0,
            "base_orphan_count": 236,
            "resolved_count": 1,
            "base_orphan_rate": 0.07398119122257053,
            "fallback_unresolved_rate": 0.07366771159874608,
            "runtime_distribution_seconds": {"mean": 0.001},
        },
    )
    assert decision == {
        "proceed_to_locked_confirmation": False,
        "f2_decision": "effect_criterion_not_preregistered",
        "effect_evidence": "strict decrease only: 1/236",
    }


def test_existing_f0_artifact_rewrite_preserves_measured_values(tmp_path):
    run_ranked_fallback_stage("tier_f0", tmp_path, workers=1)
    before_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    with (tmp_path / "records.csv").open(encoding="utf-8", newline="") as source:
        before_rows = list(csv.DictReader(source))

    rewrite_existing_fallback_artifacts("tier_f0", tmp_path)

    after_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    with (tmp_path / "records.csv").open(encoding="utf-8", newline="") as source:
        after_rows = list(csv.DictReader(source))
    assert after_summary["fallback_wall_clock_seconds"] == before_summary["fallback_wall_clock_seconds"]
    assert [row["fallback_score"] for row in after_rows] == [
        row["fallback_score"] for row in before_rows
    ]
    assert all(int(row["theoretical_call_bound"]) == 1 + int(row["candidate_count"]) for row in after_rows)
    assert all("fallback_winner_conflicts" in row for row in after_rows)
    assert "selected_defer_count" in after_summary


def _fallback_summary_record(stage, base_targets, base_orphans, unresolved, resolved):
    return {
        "stage": stage,
        "fallback_base_target_count": base_targets,
        "base_orphan_count": base_orphans,
        "fallback_unresolved_count": unresolved,
        "resolved_count": resolved,
        "newly_unassigned_count": 0,
        "fallback_gain": 0.0,
        "fallback_gate_failures": 0,
        "selected_mode_switches": 0,
        "selected_defer_count": 0,
        "search_advances": 0,
        "fallback_wall_clock_seconds": 0.01,
        "fallback_total_johnson_calls": 1,
        "candidate_count": 1,
        "theoretical_call_bound": 2,
        "all_mode_score": 1.0,
        "fixed_mode_score": 1.0,
        "screening_loss": 0.0,
        "johnson_gate_failures": 0,
        "full_raw_status": "converged",
        "screened_task_count": 1,
        "orphan_count": 0,
        "central_modes": ["recon"],
        "mode_substitutions": 0,
        "johnson_ratio": 1.0,
        "allocation_loss": 0.0,
        "warping_loss": 0.0,
    }


def test_all_stage_f2_summary_uses_only_the_independent_f1_subset():
    combined = [
        _fallback_summary_record("tier_f0", 10, 10, 0, 10),
        _fallback_summary_record("tier_f1", 3190, 236, 235, 1),
    ]
    f2_summary = _f2_summary("all", combined)
    decision = _f2_decision("all", completed=True, summary=f2_summary)

    assert f2_summary["record_count"] == 1
    assert f2_summary["fallback_base_target_count"] == 3190
    assert f2_summary["base_orphan_count"] == 236
    assert f2_summary["runtime_distribution_seconds"]["mean"] == 0.01
    assert decision["effect_evidence"] == "strict decrease only: 1/236"


def test_short_gate_stopped_f0_replay_is_auditable_without_strict_zip_error():
    first = [{"instance_id": "a", "fallback_wall_clock_seconds": 0.1}, {"instance_id": "b", "fallback_wall_clock_seconds": 0.2}]
    replay = [{"instance_id": "a", "fallback_wall_clock_seconds": 0.3}]

    audit = _compare_f0_replay(first, replay, replay_gate_stopped=True)

    assert audit == {"replay_failures": 1, "replay_gate_failure": True}
    assert _f0_hard_gate_status(
        {
            "fallback_gate_failures": 0,
            "replay_gate_failures": 1,
            "deterministic_replay_failures": audit["replay_failures"],
            "finite_iteration_bound_failures": 0,
        }
    ) == "STOPPED_GATE_FAILURE"


def test_short_non_gate_f0_replay_is_a_replay_mismatch():
    first = [{"instance_id": "a", "fallback_wall_clock_seconds": 0.1}, {"instance_id": "b", "fallback_wall_clock_seconds": 0.2}]
    replay = [{"instance_id": "a", "fallback_wall_clock_seconds": 0.3}]

    audit = _compare_f0_replay(first, replay, replay_gate_stopped=False)

    assert audit == {"replay_failures": 1, "replay_gate_failure": False}
    assert _f0_hard_gate_status(
        {
            "fallback_gate_failures": 0,
            "replay_gate_failures": 0,
            "deterministic_replay_failures": audit["replay_failures"],
            "finite_iteration_bound_failures": 0,
        }
    ) == "STOPPED_REPLAY_MISMATCH"


@pytest.mark.parametrize(
    ("replay_gate_stopped", "expected_status"),
    [(True, "STOPPED_GATE_FAILURE"), (False, "STOPPED_REPLAY_MISMATCH")],
)
def test_all_short_replay_writes_stopped_artifacts_and_never_submits_f1(
    tmp_path, monkeypatch, replay_gate_stopped, expected_status
):
    first = [
        _fallback_summary_record("tier_f0", 1, 0, 0, 0),
        _fallback_summary_record("tier_f0", 1, 0, 0, 0),
    ]
    first[0]["instance_id"] = "a"
    first[1]["instance_id"] = "b"
    replay = [dict(first[0])]
    calls = []

    def fake_build(stage):
        return [(stage, {})] * (2 if stage == "tier_f0" else 1)

    def fake_evaluate(jobs, workers):
        calls.append(jobs[0][0])
        if len(calls) == 1:
            return first, False
        return replay, replay_gate_stopped

    monkeypatch.setattr(fallback_runner, "build_ranked_fallback_jobs", fake_build)
    monkeypatch.setattr(fallback_runner, "_evaluate_jobs", fake_evaluate)

    summary = run_ranked_fallback_stage("all", tmp_path, workers=1)
    manifest = json.loads((tmp_path / "fallback_manifest.json").read_text(encoding="utf-8"))

    assert calls == ["tier_f0", "tier_f0"]
    assert summary["runner_status"] == expected_status
    assert manifest["status"] == expected_status
    assert _runner_exit_code(expected_status) != 0
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "records.csv").is_file()
    assert (tmp_path / "fallback_verdict.md").is_file()


def test_f2_summary_manifest_payload_is_a_compact_f2_field_whitelist():
    payload = _f2_summary(
        "all", [_fallback_summary_record("tier_f1", 3190, 236, 235, 1)]
    )
    assert {
        "fallback_base_target_count",
        "base_orphan_count",
        "base_orphan_rate",
        "fallback_unresolved_count",
        "fallback_unresolved_rate",
        "resolved_count",
        "newly_unassigned_count",
        "mean_fallback_gain",
        "minimum_fallback_gain",
        "fallback_win_count",
        "fallback_tie_count",
        "fallback_loss_count",
        "runtime_distribution_seconds",
        "fallback_gate_failures",
        "finite_iteration_bound_failures",
    } <= payload.keys()
    assert {
        "mean_screening_loss",
        "mean_allocation_loss",
        "mean_warping_loss",
        "gate_g1_failures",
        "gate_g2_failures",
        "mean_johnson_ratio",
    }.isdisjoint(payload)
