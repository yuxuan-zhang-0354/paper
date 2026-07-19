"""Manifest-driven D4 sensitivity runner."""

from __future__ import annotations

from argparse import ArgumentParser
import json
import os
from pathlib import Path
from typing import Any

from experiments.build_d3_manifests import canonical_digest
from experiments.run_dynamic_d3 import Work, _execute, _health, _projection
from experiments.run_dynamic_mainline import PRIVATE_COLUMNS, PUBLIC_COLUMNS, _csv_text, _text_atomic, _write_records
from uav_lifecycle.artifacts import sha256_file, write_json_atomic
from uav_lifecycle.dynamic_d4 import generate_battlefield_structure, generate_reachability
from uav_lifecycle.dynamic_types import DynamicConfig, EnvironmentModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results/dynamic_mainline/d4_design/d4_manifest.json"
DEFAULT_AUTHORIZATION = ROOT / "results/dynamic_mainline/d4_design/execution_authorization.json"
DEFAULT_OUTPUT = ROOT / "results/dynamic_mainline/d4_sensitivity"


def load_design(manifest_path: Path, authorization_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if canonical_digest(unsigned) != manifest.get("manifest_digest"):
        raise RuntimeError("D4 manifest digest mismatch")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if not authorization.get("authorized") or authorization.get("manifest_digest") != manifest["manifest_digest"]:
        raise RuntimeError("formal D4 execution is not authorized")
    return manifest


def build_works(manifest: dict[str, Any]) -> tuple[Work, ...]:
    config = DynamicConfig()
    environment = EnvironmentModel.from_config(config)
    generated = {}
    for scenario_id in manifest["scenario_ids"]:
        meta = manifest["scenario_metadata"][scenario_id]
        scenario = (
            generate_battlefield_structure(meta["structure"], float(meta["wreck_rate"]), int(meta["index"]), config)
            if meta["suite"] == "battlefield_structure" else
            generate_reachability(float(meta["map_scale"]), float(meta["time_scale"]), int(meta["index"]), config)
        )
        generated[scenario_id] = scenario
    if [generated[item].scenario_id for item in manifest["scenario_ids"]] != manifest["scenario_ids"]:
        raise RuntimeError("generated D4 scenario IDs do not match manifest")
    works = tuple(
        Work(manifest["scenario_metadata"][scenario_id]["suite"], generated[scenario_id], config, environment, method)
        for scenario_id, method in manifest["expected_rectangle"]
    )
    return works


def _write(output: Path, payloads: list[dict[str, Any]], methods: list[str]) -> None:
    records, public, private, runtime = _projection(payloads, methods)
    _write_records(output / "d4_records.csv", records)
    _text_atomic(output / "d4_public_events.csv", _csv_text(public, PUBLIC_COLUMNS))
    _text_atomic(output / "d4_private_audit_events.csv", _csv_text(private, PRIVATE_COLUMNS))
    if runtime:
        columns = list(runtime[0])
        columns += sorted({key for row in runtime for key in row} - set(columns))
        _text_atomic(output / "d4_runtime.csv", _csv_text(runtime, tuple(columns)))


def run(manifest_path: Path, authorization_path: Path, output: Path, workers: int) -> dict[str, Any]:
    manifest = load_design(manifest_path, authorization_path)
    works = build_works(manifest)
    payloads = _execute(works, workers)
    health = _health(payloads, manifest)
    _write(output / "canonical", payloads, manifest["methods"])
    expected = manifest["expected_record_count"]
    complete = health == {
        "record_count": expected, "missing_count": 0, "extra_count": 0,
        "duplicate_count": 0, "failure_count": 0, "nonterminal_count": 0,
        "gate_count": 0, "nan_count": 0,
    }
    replay: dict[str, Any] = {"status": "NOT_RUN"}
    if complete:
        ids = set(manifest["replay_audit"]["scenario_ids"])
        replay_works = tuple(work for work in works if work.scenario.scenario_id in ids)
        canonical = _projection([item for item in payloads if item["key"][0] in ids], manifest["methods"])[:3]
        mismatches = {}
        for worker_count in manifest["replay_audit"]["workers"]:
            actual_payloads = _execute(replay_works, worker_count)
            actual = _projection(actual_payloads, manifest["methods"])[:3]
            mismatches[str(worker_count)] = actual != canonical or any(item["kind"] != "result" for item in actual_payloads)
        replay = {"status": "PASS" if not any(mismatches.values()) else "FAILED", "mismatches": mismatches}
    complete = complete and replay["status"] == "PASS"
    summary = {
        "status": "D4_COMPLETE" if complete else "D4_FAILED_INCOMPLETE",
        "manifest_digest": manifest["manifest_digest"], "health": health,
        "replay": replay, "canonical_workers": workers,
    }
    write_json_atomic(output / "summary.json", summary)
    inventory = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_inventory.json"
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
    return 0 if result["status"] == "D4_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
