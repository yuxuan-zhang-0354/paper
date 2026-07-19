"""Executable D0 witness aggregation in the frozen section 11.1 order."""

from __future__ import annotations

import pytest

from uav_lifecycle.dynamic_d0 import (
    run_all_d0,
    run_d0_witness,
    write_d0_artifacts,
)
from uav_lifecycle.dynamic_scenarios import d0_scenarios


D0_IDS = tuple(fixture.name for fixture in d0_scenarios())


@pytest.mark.parametrize("witness_id", D0_IDS, ids=D0_IDS)
def test_d0_witness(witness_id: str, tmp_path) -> None:
    record = run_d0_witness(witness_id)
    fixture = next(item for item in d0_scenarios() if item.name == witness_id)

    assert record["witness_id"] == witness_id
    assert record["order"] == D0_IDS.index(witness_id) + 1
    assert record["status"] == "passed"
    assert len(record["runs"]) == len(fixture.runs)
    for actual, expected in zip(record["runs"], fixture.runs, strict=True):
        assert actual["public_trace"] == actual["expected_public_trace"]
        assert actual["private_audit_trace"] == actual["expected_private_audit_trace"]
        assert actual["termination"]["actual"] == actual["termination"]["expected"]
        assert actual["gates"]["actual"] == actual["gates"]["expected"]
        assert actual["same_process_replay"]["fieldwise_equal"] is True
        assert actual["same_process_replay"]["canonical_bytes_equal"] is True
        assert actual["digests"]["public_trace"]
        assert actual["digests"]["private_audit_trace"]
        assert set(actual["utility_decomposition"]) == {
            "destroyed_value",
            "service_cost",
            "distance_cost",
            "ammo_cost",
            "realized_utility",
            "normalized_utility",
            "gross_scenario_value",
        }
        assert set(actual["resources"]) == {"distance_consumed", "ammo_consumed"}
        for field, expected_value in actual["expected_utility_decomposition"].items():
            assert actual["utility_decomposition"][field] == pytest.approx(expected_value, abs=1e-10)
        for field, expected_value in actual["expected_resources"].items():
            assert actual["resources"][field] == pytest.approx(expected_value, abs=1e-10)
        assert actual["run_id"] == expected.run_id
    assert record["focused_assertion_audit"]["consumed_exactly_once"] is True

    if witness_id == "counter_replay_shared_initial_truth":
        left, right = record["runs"]
        assert left["actual_counter_events"] == right["actual_counter_events"]
        assert all(item["key_matches_expected"] for item in left["actual_counter_events"])
        assert all(item["draw_matches_expected"] for item in left["actual_counter_events"])

    if witness_id == "defer_reactivated_by_event":
        diagnostics = record["runs"][0]["policy_diagnostics"]
        assert diagnostics[0]["policy"] == "DynamicPPolicy"
        assert 0 not in diagnostics[0]["positive_task_targets"]
        assert diagnostics[0]["committed_targets"] == [1]
        assert 0 in diagnostics[1]["positive_task_targets"]
        assert diagnostics[1]["committed_targets"] == [0]
        assert all(
            event["rejection_reason"] != "deferred"
            for event in record["runs"][0]["private_audit_trace"]
        )

    if witness_id == "b1m_frozen_suffix_auto_next":
        diagnostics = record["runs"][0]["policy_diagnostics"]
        assert diagnostics["policy"] == "OneShotMatchedPolicy"
        assert diagnostics["planning_calls"] == 1
        assert diagnostics["auto_next_calls"] == 2
        assert diagnostics["auto_next_commit_count"] == 1
        assert diagnostics["tick0_frozen_suffix_count"] == 1
        assert diagnostics["completion_count"] == 2
        assert diagnostics["suffix_prevented_false_termination"] is True
        assert diagnostics["active_leg_only_resources"] is True
        assert diagnostics["tick0_committed_action_count"] == 1
        assert diagnostics["tick0_reserved_ammo"] == 1.0
        assert len(diagnostics["planning"][0]["planned_paths"][0]["tasks"]) == 2
        assert diagnostics["termination"] == "normal"
        assert diagnostics["termination_tick"] == diagnostics["last_completion_tick"]
        assert diagnostics["termination_tick"] < diagnostics["t_max_tick"]

    if witness_id == "commit_next_releases_suffix":
        probe = record["focused_assertion_audit"]["commit_next_probe"]
        assert probe["input_task_count"] == 2
        assert probe["input_path_unmutated"] is True
        assert probe["returned_committed_count"] == 1
        assert probe["first_leg_only_lock"] is True
        assert probe["first_leg_only_resources"] is True
        assert probe["first_leg_only_ordinal"] is True
        assert probe["first_leg_only_event"] is True
        assert probe["suffix_has_no_lock_action_or_event"] is True

    if witness_id == "dynamic_absolute_discount_clock":
        terms = record["focused_assertion_audit"]["absolute_discount_terms"]
        assert len(terms) == 2
        assert terms[0]["start_tick"] != terms[1]["start_tick"]
        assert terms[0]["completion_tick"] != terms[1]["completion_tick"]
        assert all(term["uses_absolute_tick"] for term in terms)

    if record["order"] == len(D0_IDS):
        aggregate = run_all_d0()
        assert aggregate["summary"]["witness_count"] == 22
        assert aggregate["summary"]["run_count"] == 24
        assert aggregate["summary"]["passed"] == 22
        assert aggregate["summary"]["failed"] == 0
        assert aggregate["summary"]["gate_failures"] == 0
        assert aggregate["summary"]["status"] == "passed"
        written = write_d0_artifacts(tmp_path)
        assert written["fresh_process_replay"]["fieldwise_equal"] is True
        assert written["fresh_process_replay"]["canonical_bytes_equal"] is True
        assert (tmp_path / "d0_witnesses" / "d0_records.json").is_file()
        assert (tmp_path / "d0_witnesses" / "d0_summary.json").is_file()
        assert (tmp_path / "checkpoints" / "task07.json").is_file()
