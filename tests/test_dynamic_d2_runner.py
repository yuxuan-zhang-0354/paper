import json

import pytest

from experiments.run_dynamic_d2 import build_scenarios, load_frozen_design


def test_frozen_d2_manifest_generates_exact_rectangle() -> None:
    manifest = load_frozen_design(
        __import__("pathlib").Path("results/dynamic_mainline/d2_design/d2_manifest.json"),
        __import__("pathlib").Path("results/dynamic_mainline/d2_design/execution_authorization.json"),
    )
    scenarios = build_scenarios(manifest)
    assert len(scenarios) == 4096
    assert len({scenario.scenario_id for scenario in scenarios}) == 4096


def test_d2_rejects_wrong_authorization(tmp_path) -> None:
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps({"authorized": True, "manifest_digest": "wrong"}))
    with pytest.raises(RuntimeError, match="not authorized"):
        load_frozen_design(
            __import__("pathlib").Path("results/dynamic_mainline/d2_design/d2_manifest.json"),
            authorization,
        )
