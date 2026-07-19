import json

import numpy as np

from experiments.run_first_batch import (
    extract_passed_test_count,
    hash_files,
    semantic_json_equal,
    write_running_manifest,
)


def test_semantic_json_comparison_ignores_runtime_metadata_only():
    first = {
        "value": 1.25,
        "nested": {"timestamp": "first", "duration_seconds": 2.0},
    }
    second = {
        "value": 1.25,
        "nested": {"timestamp": "second", "duration_seconds": 9.0},
    }
    assert semantic_json_equal(first, second)
    second["value"] = 1.26
    assert not semantic_json_equal(first, second)


def test_semantic_json_comparison_normalizes_numpy_arrays():
    first = {"oracle": np.array([1.0, 2.0])}
    second = {"oracle": np.array([1.0, 2.0])}
    assert semantic_json_equal(first, second)


def test_extract_passed_test_count_reads_pytest_summary():
    assert extract_passed_test_count("52 passed in 0.36s") == 52


def test_source_hashes_change_when_source_content_changes(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    first = hash_files((source,), root=tmp_path)
    source.write_text("value = 2\n", encoding="utf-8")
    second = hash_files((source,), root=tmp_path)
    assert first != second


def test_running_manifest_invalidates_previous_pass_immediately(tmp_path):
    write_running_manifest(
        tmp_path,
        run_id="run-123",
        started_at="2026-07-13T00:00:00+00:00",
        source_hashes={"source.py": "abc"},
    )
    manifest = json.loads(
        (tmp_path / "first_batch_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "RUNNING"
    assert manifest["run_id"] == "run-123"
