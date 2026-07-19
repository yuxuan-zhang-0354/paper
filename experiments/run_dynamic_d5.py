"""Run the registered D5 allocator factorial with fresh common-random-number scenarios."""

from __future__ import annotations

from argparse import ArgumentParser
import json
import os
from pathlib import Path

from experiments.run_dynamic_d3 import (
    Work, _execute, _health, _projection, _write, load_frozen_design,
)
from uav_lifecycle.artifacts import sha256_file, write_json_atomic
from uav_lifecycle.dynamic_d3 import (
    D3_SCALE_CELLS, generate_d5_allocation_pressure, generate_d5_scale,
)
from uav_lifecycle.dynamic_types import DynamicConfig, EnvironmentModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results/dynamic_mainline/d5_factorial_ablation"
DEFAULT_MANIFEST = DEFAULT_ROOT / "design/d5_manifest.json"
DEFAULT_AUTHORIZATION = DEFAULT_ROOT / "design/execution_authorization.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "formal"


def build_works(manifest: dict) -> tuple[Work, ...]:
    config = DynamicConfig()
    environment = EnvironmentModel.from_config(config)
    scale = {cell.cell_id: cell for cell in D3_SCALE_CELLS}
    generated = {}
    for scenario_id in manifest["scenario_ids"]:
        meta = manifest["scenario_metadata"][scenario_id]
        if meta["suite"] == "pressure":
            scenario = generate_d5_allocation_pressure(meta["condition"], int(meta["index"]), config)
        elif meta["suite"] == "scale":
            scenario = generate_d5_scale(scale[meta["cell_id"]], int(meta["index"]), config)
        else:
            raise ValueError(f"unknown D5 suite: {meta['suite']}")
        if scenario.scenario_id != scenario_id:
            raise RuntimeError("generated D5 scenario ID does not match manifest")
        generated[scenario_id] = (scenario, meta["suite"])
    works = tuple(
        Work(generated[scenario_id][1], generated[scenario_id][0], config, environment, method)
        for scenario_id, method in manifest["expected_rectangle"]
    )
    if [[work.scenario.scenario_id, work.method] for work in works] != manifest["expected_rectangle"]:
        raise RuntimeError("generated D5 rectangle does not match manifest")
    return works


def run(manifest_path: Path, authorization_path: Path, output: Path, workers: int) -> dict:
    manifest = load_frozen_design(manifest_path, authorization_path)
    works = build_works(manifest)
    payloads = _execute(works, workers)
    health = _health(payloads, manifest)
    _write(output / "canonical", payloads, manifest["methods"], "d5")
    expected = manifest["expected_record_count"]
    complete = health == {
        "record_count": expected, "missing_count": 0, "extra_count": 0,
        "duplicate_count": 0, "failure_count": 0, "nonterminal_count": 0,
        "gate_count": 0, "nan_count": 0,
    }
    replay = {"status": "NOT_RUN"}
    if complete:
        replay_ids = set(manifest["replay_audit"]["scenario_ids"])
        replay_works = tuple(work for work in works if work.scenario.scenario_id in replay_ids)
        canonical = _projection(
            [payload for payload in payloads if payload["key"][0] in replay_ids], manifest["methods"],
        )[:3]
        mismatches = {}
        replay_manifest = {
            **manifest,
            "expected_rectangle": [[work.scenario.scenario_id, work.method] for work in replay_works],
        }
        for count in manifest["replay_audit"]["workers"]:
            rerun = _execute(replay_works, int(count))
            mismatches[str(count)] = (
                _projection(rerun, manifest["methods"])[:3] != canonical
                or _health(rerun, replay_manifest)["failure_count"] > 0
            )
        replay = {"status": "PASS" if not any(mismatches.values()) else "FAILED", "mismatches": mismatches}
    complete = complete and replay["status"] == "PASS"
    summary = {
        "status": "D5_COMPLETE" if complete else "D5_FAILED_INCOMPLETE",
        "manifest_digest": manifest["manifest_digest"], "health": health,
        "replay": replay, "canonical_workers": workers,
    }
    write_json_atomic(output / "summary.json", summary)
    inventory = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_inventory.json"
    }
    write_json_atomic(output / "artifact_inventory.json", inventory)
    return summary


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(22, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()
    result = run(args.manifest, args.authorization, args.output, args.workers)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "D5_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
