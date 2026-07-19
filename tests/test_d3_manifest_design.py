from pathlib import Path

import pytest

from experiments.build_d3_manifests import (
    ALLOCATION_PRESSURE_METHODS,
    CBBA_ISOLATION_METHODS,
    METHODS,
    build_d2_diagnostic_manifest,
    build_d3_manifest,
    canonical_digest,
)


def _valid_digest(manifest):
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    return manifest["manifest_digest"] == canonical_digest(unsigned)


def test_d2_diagnostics_are_source_pinned_and_nonconfirmatory() -> None:
    if not Path("results/dynamic_mainline/d2_confirmation/canonical/records.csv").exists():
        pytest.skip("the optional full D2 raw archive is not included in the lightweight repository")
    manifest = build_d2_diagnostic_manifest()
    assert _valid_digest(manifest)
    assert manifest["source_record_count"] == 40960
    assert manifest["analysis_label"] == "post_hoc_explanatory_not_confirmatory"
    assert manifest["execution_authorized"] is False
    assert all(len(item["sha256"]) == 64 for item in manifest["sources"].values())


def test_d3_has_exact_registered_suites_rectangle_and_new_ids() -> None:
    manifest = build_d3_manifest()
    assert _valid_digest(manifest)
    assert manifest["scenario_count"] == 8736
    assert manifest["expected_record_count"] == 58464
    assert len(manifest["scenario_ids"]) == len(set(manifest["scenario_ids"])) == 8736
    assert len(manifest["expected_rectangle"]) == 58464
    assert len({tuple(item) for item in manifest["expected_rectangle"]}) == 58464
    allowed = set(METHODS + CBBA_ISOLATION_METHODS + ALLOCATION_PRESSURE_METHODS)
    assert all(item[1] in allowed for item in manifest["expected_rectangle"])
    assert all(item.startswith(("D3S-", "D3C-", "D3M-", "D3W-", "D3A-", "D3P-")) for item in manifest["scenario_ids"])
    assert manifest["cex_excluded_from_scalable_suites"] is True
    assert manifest["execution_authorized"] is False


def test_d3_suite_counts_and_replay_are_frozen() -> None:
    manifest = build_d3_manifest()
    metadata = manifest["scenario_metadata"].values()
    counts = {
        suite: sum(item["suite"] == suite for item in metadata)
        for suite in (
            "scale", "continuous_belief", "model_mismatch", "utility_profile",
            "cbba_isolation", "allocation_pressure",
        )
    }
    assert counts == {
        "scale": 672,
        "continuous_belief": 1024,
        "model_mismatch": 4608,
        "utility_profile": 1536,
        "cbba_isolation": 512,
        "allocation_pressure": 384,
    }
    assert len(manifest["replay_audit"]["scenario_ids"]) == 125
    assert manifest["failure_policy"]["effect_based_tuning_forbidden"] is True


def test_scale_resource_rules_are_non_degenerate() -> None:
    cells = build_d3_manifest()["suites"]["scale"]["cells"]
    assert len(cells) == 7
    for values in cells.values():
        assert values["ammo_per_agent"] >= 2
        assert values["t_max"] > 0
        assert values["range_per_agent"] > values["t_max"]
        assert values["arena_half_width"] >= 6
