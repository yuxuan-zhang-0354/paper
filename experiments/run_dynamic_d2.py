"""Run the frozen D2 confirmation matrix and deterministic replay audit."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.build_d2_manifest import canonical_digest
from experiments.run_dynamic_mainline import (
    PRIVATE_COLUMNS,
    PUBLIC_COLUMNS,
    WorkerInput,
    _csv_text,
    _digest,
    _private_rows,
    _public_rows,
    _record_row,
    _run_jobs,
    _text_atomic,
    _write_records,
)
from uav_lifecycle.artifacts import sha256_file, write_json_atomic
from uav_lifecycle.dynamic_scenarios import (
    D1_CELLS,
    dynamic_config_registry,
    generate_d2_scenario,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results/dynamic_mainline/d2_design/d2_manifest.json"
DEFAULT_AUTHORIZATION = ROOT / "results/dynamic_mainline/d2_design/execution_authorization.json"
DEFAULT_OUTPUT = ROOT / "results/dynamic_mainline/d2_confirmation"


def load_frozen_design(manifest_path: Path, authorization_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if canonical_digest(unsigned) != manifest.get("manifest_digest"):
        raise RuntimeError("D2 manifest digest mismatch")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if not authorization.get("authorized") or authorization.get("manifest_digest") != manifest["manifest_digest"]:
        raise RuntimeError("D2 execution is not authorized for this manifest")
    return manifest


def build_scenarios(manifest: dict[str, Any]) -> tuple[Any, ...]:
    cells = {cell.cell_id: cell for cell in D1_CELLS}
    start = manifest["generator"]["scenario_index_start"]
    stop = manifest["generator"]["scenario_index_stop_exclusive"]
    scenarios = tuple(
        generate_d2_scenario(cells[cell_id], index, manifest["registered_config_id"])
        for cell_id in manifest["cells"]
        for index in range(start, stop)
    )
    ids = [scenario.scenario_id for scenario in scenarios]
    if ids != manifest["scenario_ids"]:
        raise RuntimeError("generated D2 scenario IDs do not match frozen manifest")
    rectangle = [[scenario_id, method] for scenario_id in ids for method in manifest["methods"]]
    if rectangle != manifest["expected_rectangle"]:
        raise RuntimeError("generated D2 rectangle does not match frozen manifest")
    if any(scenario.crn_namespace != manifest["generator"]["rng_namespace"] for scenario in scenarios):
        raise RuntimeError("generated D2 RNG namespace does not match frozen manifest")
    return scenarios


def _execute(
    scenarios: tuple[Any, ...], methods: tuple[str, ...], workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config = dynamic_config_registry()["recon_damage_plus_010_r2_a6_b3"]
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    works = tuple(WorkerInput(scenario, config, method) for scenario in scenarios for method in methods)
    payloads = _run_jobs(works, workers)
    method_order = {method: index for index, method in enumerate(methods)}
    payloads.sort(key=lambda item: (item["key"][0], method_order[item["key"][1]]))
    records = [_record_row(payload, scenario_by_id) for payload in payloads]
    results = [payload["result"] for payload in payloads if payload["kind"] == "result"]
    public = [row for result in results for row in _public_rows(result)]
    private = [row for result in results for row in _private_rows(result)]
    key = lambda row: (row["scenario_id"], method_order[row["method"]], row["event_id"])
    public.sort(key=key)
    private.sort(key=key)
    return records, public, private


def _write_run(
    destination: Path,
    records: list[dict[str, Any]],
    public: list[dict[str, Any]],
    private: list[dict[str, Any]],
) -> None:
    _write_records(destination / "records.csv", records)
    _text_atomic(destination / "public_events.csv", _csv_text(public, PUBLIC_COLUMNS))
    _text_atomic(destination / "private_audit_events.csv", _csv_text(private, PRIVATE_COLUMNS))


def _run_health(records: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, int]:
    numeric = ("normalized_utility", "realized_utility", "gross_scenario_value")
    keys = [(row.get("scenario_id"), row.get("method")) for row in records]
    expected = {tuple(key) for key in manifest["expected_rectangle"]}
    return {
        "record_count": len(records),
        "missing_count": len(expected - set(keys)),
        "extra_count": len(set(keys) - expected),
        "duplicate_count": len(keys) - len(set(keys)),
        "nonterminal_count": sum(row.get("terminal") is not True for row in records),
        "failure_count": sum(row.get("status") != "complete" for row in records),
        "gate_count": sum(len(row.get("allocator_gates", ())) for row in records),
        "nan_count": sum(
            not math.isfinite(float(row[name]))
            for row in records for name in numeric if name in row
        ),
    }


def _bootstrap(
    differences: dict[str, list[float]], contrast: str, manifest: dict[str, Any],
) -> tuple[float, list[float]]:
    cells = sorted(differences)
    iterations = manifest["bootstrap"]["iterations"]
    samples: list[float] = []
    for replicate in range(iterations):
        means = []
        for cell in cells:
            values = differences[cell]
            draws = []
            for draw in range(len(values)):
                token = (
                    f"{manifest['bootstrap']['namespace']}|{manifest['manifest_digest']}|"
                    f"{contrast}|{cell}|{replicate}|{draw}"
                )
                index = int.from_bytes(sha256(token.encode()).digest()[:8], "big") % len(values)
                draws.append(values[index])
            means.append(float(np.mean(draws)))
        samples.append(float(np.mean(means)))
    mean = float(np.mean([np.mean(differences[cell]) for cell in cells]))
    return mean, samples


def _holm(raw_p: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw_p, key=raw_p.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * raw_p[name]))
        adjusted[name] = running
    return adjusted


def analyze(records: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    by_scenario: dict[str, dict[str, dict[str, Any]]] = {}
    for row in records:
        by_scenario.setdefault(row["scenario_id"], {})[row["method"]] = row
    contrasts: dict[str, Any] = {}
    raw_secondary: dict[str, float] = {}
    comparators = [name.removeprefix("P-") for name in (
        [manifest["primary_contrast"]]
        + manifest["secondary_holm_family"]
        + manifest["sensitivity_contrasts"]
        + [manifest["separate_exact_contrast"]]
    )]
    for comparator in comparators:
        differences: dict[str, list[float]] = {}
        for methods in by_scenario.values():
            left, right = methods["P"], methods[comparator]
            differences.setdefault(left["cell_id"], []).append(
                float(left["normalized_utility"]) - float(right["normalized_utility"])
            )
        name = f"P-{comparator}"
        mean, boot = _bootstrap(differences, name, manifest)
        interval = [float(np.quantile(boot, q)) for q in (0.025, 0.975)]
        # Shift the empirical bootstrap distribution to the null before using it
        # as a one-sided test statistic; unshifted bootstrap draws are for CIs.
        raw_p = (sum((value - mean) >= mean for value in boot) + 1) / (len(boot) + 1)
        contrasts[name] = {
            "equal_cell_mean": mean,
            "bootstrap_ci_95": interval,
            "one_sided_centered_bootstrap_p": raw_p,
            "cell_means": {cell: float(np.mean(values)) for cell, values in sorted(differences.items())},
            "win_tie_loss": {
                "win": sum(value > 1e-12 for values in differences.values() for value in values),
                "tie": sum(abs(value) <= 1e-12 for values in differences.values() for value in values),
                "loss": sum(value < -1e-12 for values in differences.values() for value in values),
            },
        }
        if name in manifest["secondary_holm_family"]:
            raw_secondary[name] = raw_p
    adjusted = _holm(raw_secondary)
    for name, value in adjusted.items():
        contrasts[name]["holm_adjusted_p"] = value
        contrasts[name]["holm_reject_0.05"] = value <= 0.05
    primary = contrasts[manifest["primary_contrast"]]
    confirmed = (
        primary["equal_cell_mean"] >= manifest["confirmation_rule"]["equal_cell_mean_at_least"]
        and primary["bootstrap_ci_95"][0] > 0.0
    )
    return {
        "status": "D2_CONFIRMATION_PASS" if confirmed else "D2_COMPLETE_PRIMARY_NOT_CONFIRMED",
        "primary_confirmed": confirmed,
        "contrasts": contrasts,
        "bootstrap_iterations": manifest["bootstrap"]["iterations"],
        "manifest_digest": manifest["manifest_digest"],
    }


def finalize_existing(manifest_path: Path, authorization_path: Path, output: Path) -> dict[str, Any]:
    """Re-audit and analyze already completed canonical artifacts without rerunning episodes."""

    manifest = load_frozen_design(manifest_path, authorization_path)
    with (output / "canonical/records.csv").open(encoding="utf-8", newline="") as source:
        records = list(csv.DictReader(source))
    for row in records:
        row["terminal"] = row["terminal"].lower() == "true"
        row["allocator_gates"] = json.loads(row.get("allocator_gates") or "[]")
    health = _run_health(records, manifest)
    replay = json.loads((output / "replay_summary.json").read_text(encoding="utf-8"))
    structural_ok = (
        health["record_count"] == manifest["expected_record_count"]
        and all(value == 0 for name, value in health.items() if name != "record_count")
        and replay.get("status") == "PASS"
    )
    if structural_ok:
        analysis = analyze(records, manifest)
        write_json_atomic(output / "d2_confirmation_analysis.json", analysis)
        summary = {**analysis, "health": health, "replay": replay}
    else:
        summary = {
            "status": "D2_FAILED_INCOMPLETE", "health": health, "replay": replay,
            "manifest_digest": manifest["manifest_digest"],
        }
    write_json_atomic(output / "summary.json", summary)
    inventory = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_inventory.json"
    }
    write_json_atomic(output / "artifact_inventory.json", inventory)
    return summary


def run(manifest_path: Path, authorization_path: Path, output: Path) -> dict[str, Any]:
    manifest = load_frozen_design(manifest_path, authorization_path)
    scenarios = build_scenarios(manifest)
    methods = tuple(manifest["methods"])
    canonical = output / "canonical"
    records, public, private = _execute(scenarios, methods, manifest["canonical_workers"])
    _write_run(canonical, records, public, private)
    health = _run_health(records, manifest)
    expected = manifest["expected_record_count"]
    structural_ok = health == {
        "record_count": expected, "missing_count": 0, "extra_count": 0,
        "duplicate_count": 0, "nonterminal_count": 0, "failure_count": 0,
        "gate_count": 0, "nan_count": 0,
    }

    replay_ok = False
    replay_summary: dict[str, Any] = {"status": "NOT_RUN"}
    if structural_ok:
        replay_ids = set(manifest["replay_audit"]["scenario_ids"])
        replay_scenarios = tuple(scenario for scenario in scenarios if scenario.scenario_id in replay_ids)
        canonical_record = {(row["scenario_id"], row["method"]): row for row in records if row["scenario_id"] in replay_ids}
        canonical_public = [row for row in public if row["scenario_id"] in replay_ids]
        canonical_private = [row for row in private if row["scenario_id"] in replay_ids]
        mismatches: dict[str, bool] = {}
        for workers in manifest["replay_audit"]["workers"]:
            replay_records, replay_public, replay_private = _execute(replay_scenarios, methods, workers)
            _write_run(output / f"replay_workers{workers}", replay_records, replay_public, replay_private)
            selected = {(row["scenario_id"], row["method"]): row for row in replay_records}
            mismatches[str(workers)] = not (
                selected == canonical_record
                and replay_public == canonical_public
                and replay_private == canonical_private
            )
        replay_ok = not any(mismatches.values())
        replay_summary = {"status": "PASS" if replay_ok else "FAILED", "mismatches": mismatches}
    write_json_atomic(output / "replay_summary.json", replay_summary)

    if not structural_ok or not replay_ok:
        summary = {
            "status": "D2_FAILED_INCOMPLETE", "health": health,
            "replay": replay_summary, "manifest_digest": manifest["manifest_digest"],
        }
    else:
        analysis = analyze(records, manifest)
        write_json_atomic(output / "d2_confirmation_analysis.json", analysis)
        summary = {**analysis, "health": health, "replay": replay_summary}
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
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    result = (
        finalize_existing(args.manifest, args.authorization, args.output)
        if args.finalize_existing else run(args.manifest, args.authorization, args.output)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "D2_CONFIRMATION_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
