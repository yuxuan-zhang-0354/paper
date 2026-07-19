from experiments.build_d2_manifest import build_manifest, canonical_digest


def test_d2_manifest_is_complete_new_and_non_executable() -> None:
    manifest = build_manifest()
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    assert manifest["manifest_digest"] == canonical_digest(unsigned)
    assert manifest["scenario_count"] == 4096
    assert manifest["scenarios_per_cell"] == 512
    assert manifest["expected_record_count"] == 40960
    assert len(manifest["scenario_ids"]) == len(set(manifest["scenario_ids"])) == 4096
    assert len(manifest["expected_rectangle"]) == 40960
    assert set(manifest["scenario_cells"].values()) == set(manifest["cells"])
    assert all("-S0" not in scenario_id for scenario_id in manifest["scenario_ids"])
    assert manifest["sample_size_basis"]["d1_effect_mean_used"] is False
    assert manifest["execution_authorized"] is manifest["d2_authorized"] is False


def test_d2_analysis_and_failure_rules_are_frozen() -> None:
    manifest = build_manifest()
    assert manifest["primary_contrast"] == "P-B1m"
    assert manifest["confirmation_rule"] == {
        "equal_cell_mean_at_least": 0.01,
        "bootstrap_ci_lower_above": 0.0,
        "complete_matrix": True,
        "zero_gates": True,
    }
    assert manifest["bootstrap"]["iterations"] == 10000
    assert manifest["failure_policy"]["complete_case_confirmation_forbidden"] is True
    assert manifest["failure_policy"]["automatic_retry_forbidden"] is True
