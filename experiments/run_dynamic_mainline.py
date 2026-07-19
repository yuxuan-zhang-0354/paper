"""Small deterministic runner for the frozen dynamic-lifecycle D0/D1 experiment."""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import csv
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile
from typing import Any, Iterable

from uav_lifecycle.artifacts import jsonable, sha256_file, write_json_atomic
from uav_lifecycle.dynamic_d0 import write_d0_artifacts
from uav_lifecycle.dynamic_analysis import (
    MethodBlindDiagnostics,
    PublicTick0Coverage,
    paired_d1_summary,
    project_calibration_coverage,
    summarize_effect_blind,
    validate_method_matrix,
    write_dynamic_verdict,
)
from uav_lifecycle.dynamic_policies import make_policy
from uav_lifecycle.dynamic_planning import tick0_target_screening
from uav_lifecycle.dynamic_scenarios import (
    D1_CELLS,
    d1_draw_digest,
    dynamic_config_registry,
    dynamic_registry_digest,
    generate_d1_scenario,
)
from uav_lifecycle.dynamic_simulator import initialize_state, run_episode
from uav_lifecycle.dynamic_types import DynamicConfig, DynamicScenario, EpisodeResult


D1_METHODS = (
    "P", "B1m", "B2", "B3", "B4", "B5(4)", "B5(2)", "B5(8)", "B6", "CEX",
)
ALLOWED_STAGES = {"d0", "d1-calibrate", "d1-reveal", "all-d0-d1"}
SPEC_VERSION = "dynamic_lifecycle_mainline_v2"
WORKSPACE = Path(__file__).resolve().parents[1]
SPEC_PATH = WORKSPACE / "docs/superpowers/specs/2026-07-14-dynamic-lifecycle-mainline-design.md"

PUBLIC_COLUMNS = (
    "scenario_id", "cell_id", "method", "event_id", "tick", "target_id",
    "agent_id", "mode", "event_kind", "observation",
)
PRIVATE_COLUMNS = (
    "scenario_id", "cell_id", "method", "event_id", "tick", "target_id",
    "agent_id", "mode", "draw", "true_category", "damage_before", "damage_after",
    "physical_success", "realized_reward", "invalid_attack", "initial_wreck_attack",
    "counter_key",
)
@dataclass(frozen=True, slots=True)
class WorkerInput:
    scenario: DynamicScenario
    config: DynamicConfig
    method: str
def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")
def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _file_set_digest(paths: Iterable[Path]) -> str:
    digest = sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.relative_to(WORKSPACE).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _all_d1_scenarios(config_id: str) -> tuple[DynamicScenario, ...]:
    return tuple(
        generate_d1_scenario(cell, seed, config_id)
        for cell in D1_CELLS for seed in range(20)
    )
def build_d1_manifest(
    registered_config_id: str,
    workers: int,
    calibration_run_id: str,
    *,
    scenarios: tuple[DynamicScenario, ...] | None = None,
    methods: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], tuple[DynamicScenario, ...]]:
    """Build the complete expected rectangle without running an episode."""

    if not 1 <= workers <= 22:
        raise ValueError("workers must lie in [1, 22]")
    registry = dynamic_config_registry()
    if registered_config_id not in registry:
        raise KeyError(f"unregistered dynamic config ID: {registered_config_id}")
    selected_scenarios = (
        _all_d1_scenarios(registered_config_id)
        if scenarios is None else tuple(scenarios)
    )
    selected_methods = D1_METHODS if methods is None else tuple(methods)
    scenario_ids = tuple(scenario.scenario_id for scenario in selected_scenarios)
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("duplicate scenario ID in D1 manifest")
    if len(set(selected_methods)) != len(selected_methods):
        raise ValueError("duplicate method in D1 manifest")
    unknown = [method for method in selected_methods if method not in D1_METHODS]
    if unknown:
        raise ValueError(f"unknown D1 method: {unknown[0]}")
    if "CEX" in selected_methods and any(len(scenario.targets) > 5 for scenario in selected_scenarios):
        raise ValueError("CEX is applicable only when M<=5")
    rectangle = tuple(
        (scenario.scenario_id, method)
        for scenario in sorted(selected_scenarios, key=lambda item: item.scenario_id)
        for method in selected_methods
    )
    config = registry[registered_config_id]
    code_paths = (
        Path(__file__),
        WORKSPACE / "src/uav_lifecycle/dynamic_scenarios.py",
        WORKSPACE / "src/uav_lifecycle/dynamic_policies.py",
        WORKSPACE / "src/uav_lifecycle/dynamic_simulator.py",
        WORKSPACE / "src/uav_lifecycle/dynamic_rng.py",
    )
    manifest: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "spec_digest": sha256_file(SPEC_PATH),
        "spec_registry_sha256": dynamic_registry_digest(),
        "registered_config_id": registered_config_id,
        "config_digest": _digest(config),
        "generator_digest": d1_draw_digest(selected_scenarios),
        "code_digest": _file_set_digest(code_paths),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "workers": workers,
        "scenario_ids": list(scenario_ids),
        "scenario_cells": {
            scenario.scenario_id: scenario.cell_id for scenario in selected_scenarios
        },
        "methods": list(selected_methods),
        "expected_rectangle": [list(key) for key in rectangle],
        "calibration_round": str(calibration_run_id),
        "crn_contract": {
            "rng_version": "sha256-u64-v1",
            "experiment_id": "dynamic-lifecycle-mainline-v2",
            "generator_version": "d1-generator-v1",
            "cross_method_initial_truth": True,
            "manifest_key": ["scenario_id", "method"],
        },
        "d2_authorized": False,
    }
    manifest["manifest_digest"] = _digest(manifest)
    return manifest, selected_scenarios
def _failure(work: WorkerInput, error: BaseException) -> dict[str, Any]:
    return {
        "kind": "failure",
        "key": (work.scenario.scenario_id, work.method),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
def _worker(work: WorkerInput) -> dict[str, Any]:
    """Run one immutable scenario/config/method tuple and return in memory."""

    try:
        policy = make_policy(work.method, work.config)
        result = run_episode(
            work.scenario, policy, config=work.config, method=work.method,
        )
        return {
            "kind": "result",
            "key": (work.scenario.scenario_id, work.method),
            "result": result,
        }
    except Exception as error:
        return _failure(work, error)
def _run_jobs(works: tuple[WorkerInput, ...], workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        payloads = [_worker(work) for work in works]
    else:
        payloads = []
        submitted: list[tuple[WorkerInput, Any]] = []
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for work in works:
                    try:
                        submitted.append((work, executor.submit(_worker, work)))
                    except Exception as error:
                        payloads.append(_failure(work, error))
                for work, future in submitted:
                    try:
                        payloads.append(future.result())
                    except Exception as error:
                        payloads.append(_failure(work, error))
        except Exception as error:
            completed = {tuple(payload["key"]) for payload in payloads}
            payloads.extend(
                _failure(work, error)
                for work in works
                if (work.scenario.scenario_id, work.method) not in completed
            )
    by_key = {tuple(payload["key"]): payload for payload in payloads}
    return [
        by_key.get(
            (work.scenario.scenario_id, work.method),
            _failure(work, RuntimeError("worker returned no terminal result")),
        )
        for work in works
    ]
def _public_rows(result: EpisodeResult) -> list[dict[str, Any]]:
    identity = {
        "scenario_id": result.record.scenario_id,
        "cell_id": result.record.cell_id,
        "method": result.record.method,
    }
    return [
        {
            **identity,
            "event_id": event.event_id,
            "tick": event.tick,
            "target_id": event.target_id,
            "agent_id": event.agent_id,
            "mode": event.mode,
            "event_kind": "observation" if hasattr(event, "observation") else "ack",
            "observation": getattr(event, "observation", ""),
        }
        for event in result.public_events
    ]
def _private_rows(result: EpisodeResult) -> list[dict[str, Any]]:
    identity = {
        "scenario_id": result.record.scenario_id,
        "cell_id": result.record.cell_id,
        "method": result.record.method,
    }
    return [
        {**identity, **asdict(event), "counter_key": asdict(event.counter_key)}
        for event in result.private_audit_events
    ]
def _record_row(payload: dict[str, Any], scenarios: dict[str, DynamicScenario]) -> dict[str, Any]:
    scenario_id, method = payload["key"]
    if payload["kind"] == "result":
        row = asdict(payload["result"].record)
        row.update({"terminal": True, "error_type": "", "error_message": ""})
    else:
        row = {
            "scenario_id": scenario_id,
            "cell_id": scenarios[scenario_id].cell_id,
            "method": method,
            "termination": "worker_failure",
            "status": "failed",
            "terminal": True,
            "allocator_gates": [],
            "error_type": payload["error_type"],
            "error_message": payload["error_message"],
        }
    row["record_digest"] = _digest(row)
    return row
def _text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _csv_text(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            name: (
                _canonical_bytes(row.get(name)).decode("ascii")
                if isinstance(row.get(name), (dict, tuple, list))
                else row.get(name, "")
            )
            for name in columns
        })
    return stream.getvalue()


def _write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    first = (
        "scenario_id", "cell_id", "method", "termination", "status", "terminal",
        "error_type", "error_message", "record_digest",
    )
    extra = tuple(sorted(set().union(*(row for row in rows)) - set(first)))
    _text_atomic(path, _csv_text(rows, first + extra))


def _read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        for name in ("allocator_gates",):
            try:
                row[name] = json.loads(row.get(name, "[]"))
            except json.JSONDecodeError:
                pass
    return rows


def _write_manifest_once(path: Path, requested: dict[str, Any]) -> None:
    if not path.exists():
        write_json_atomic(path, requested)
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in existing.items() if key != "manifest_digest"}
    if existing.get("manifest_digest") != _digest(unsigned) or existing != requested:
        raise RuntimeError("sealed manifest cannot be reused or changed")


def _run_d1(
    output: Path,
    workers: int,
    run_id: str,
    config_id: str,
    scenarios: tuple[DynamicScenario, ...] | None,
    methods: tuple[str, ...] | None,
) -> dict[str, Any]:
    manifest, selected = build_d1_manifest(
        config_id, workers, run_id, scenarios=scenarios, methods=methods,
    )
    sealed = output / "calibration" / f"run_{run_id}" / "sealed"
    _write_manifest_once(sealed / "dynamic_manifest.json", manifest)
    scenario_by_id = {scenario.scenario_id: scenario for scenario in selected}
    config = dynamic_config_registry()[config_id]
    works = tuple(
        WorkerInput(scenario_by_id[scenario_id], config, method)
        for scenario_id, method in map(tuple, manifest["expected_rectangle"])
    )
    payloads = _run_jobs(works, workers)
    method_order = tuple(manifest["methods"])
    payloads.sort(key=lambda item: (
        item["key"][0], method_order.index(item["key"][1]),
    ))
    records = [_record_row(payload, scenario_by_id) for payload in payloads]
    results = [payload["result"] for payload in payloads if payload["kind"] == "result"]
    public = [row for result in results for row in _public_rows(result)]
    private = [row for result in results for row in _private_rows(result)]
    def event_key(row: dict[str, Any]) -> tuple[str, int, int]:
        return row["scenario_id"], method_order.index(row["method"]), row["event_id"]
    public.sort(key=event_key)
    private.sort(key=event_key)
    statistics = {
        "record_count": len(records),
        "terminal_count": sum(bool(row["terminal"]) for row in records),
        "failure_count": sum(row["status"] == "failed" for row in records),
        "gate_count": sum(len(result.gate_failures) for result in results),
    }
    algorithm_digest = _digest({
        "records": records, "public_events": public,
        "private_audit_events": private, "statistics": statistics,
    })
    summary = {
        **statistics,
        "stage": "d1-calibrate",
        "status": (
            "COMPLETE"
            if statistics["failure_count"] == statistics["gate_count"] == 0
            else "FAILED/INCOMPLETE"
        ),
        "algorithm_digest": algorithm_digest,
        "worker_count": workers,
        "d2_authorized": False,
    }
    _write_records(sealed / "records.csv", records)
    _text_atomic(sealed / "public_events.csv", _csv_text(public, PUBLIC_COLUMNS))
    _text_atomic(sealed / "private_audit_events.csv", _csv_text(private, PRIVATE_COLUMNS))
    write_json_atomic(sealed / "summary.json", summary)
    d0_path = output / "d0_witnesses" / "d0_summary.json"
    d0_passed = d0_path.is_file() and json.loads(
        d0_path.read_text(encoding="utf-8")
    ).get("status") == "passed"
    tiers = {cell.cell_id: cell.resource_tier for cell in D1_CELLS}
    public_coverage = tuple(
        PublicTick0Coverage(
            scenario.scenario_id, tiers[scenario.cell_id],
            tuple(item.region for item in screening),
            any(item.positive_single_task for item in screening),
            any(item.resource_blocked for item in screening),
        )
        for scenario in selected
        for screening in (
            tick0_target_screening(initialize_state(scenario).snapshot(), config, scenario.t_max_tick),
        )
    )
    diagnostics = {
        scenario.scenario_id: MethodBlindDiagnostics(
            sum(len(row["allocator_gates"]) for row in records if row["scenario_id"] == scenario.scenario_id),
            sum(row["status"] != "complete" for row in records if row["scenario_id"] == scenario.scenario_id),
            0.0,
        ) for scenario in selected
    }
    scenario_ids = tuple(scenario.scenario_id for scenario in selected)
    target_counts = {scenario.scenario_id: len(scenario.targets) for scenario in selected}
    coverage = project_calibration_coverage(
        public_coverage, diagnostics, scenario_ids, target_counts,
    )
    write_json_atomic(sealed / "public_coverage.json", {
        "calibration_run_id": run_id, "registered_config_id": config_id,
        "d0_passed": d0_passed, "records": coverage,
        "summary": summarize_effect_blind(coverage, scenario_ids, target_counts),
    })
    _text_atomic(
        sealed / "dynamic_verdict.md",
        "# D1 calibration execution\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Terminal records: {summary['terminal_count']} / {len(works)}\n"
        f"- Failures / Gates: {summary['failure_count']} / {summary['gate_count']}\n"
        "- Effects revealed: `false`\n- D2 authorized: `false`\n",
    )
    return summary


def _reveal(output: Path, run_id: str) -> dict[str, Any]:
    root = output / "calibration" / f"run_{run_id}"
    lock_path = root / "calibration_lock.json"
    if not lock_path.is_file():
        raise RuntimeError("reveal requires an existing locked calibration")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("locked") is not True or lock.get("status") != "passed":
        raise RuntimeError("reveal requires a passing locked calibration")
    manifest_path = root / "sealed" / "dynamic_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("locked calibration is missing its sealed manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if lock.get("manifest_digest") != manifest.get("manifest_digest"):
        raise RuntimeError("locked calibration manifest digest mismatch")
    result = {
        "stage": "d1-reveal", "status": "LOCK_VERIFIED",
        "analysis_performed": False, "manifest_digest": manifest["manifest_digest"],
        "d2_authorized": False,
    }
    sealed = root / "sealed"
    records = _read_records(sealed / "records.csv")
    expected = tuple(map(tuple, manifest["expected_rectangle"]))
    expected_cells = manifest["scenario_cells"]
    replay_verified = lock.get("replay_verified")
    validation = validate_method_matrix(
        records, expected, expected_cells=expected_cells,
        replay_verified=replay_verified,
    )
    methods = {row.get("method") for row in records}
    if validation.status == "COMPLETE" and {"P", "B1m"} <= methods:
        summary = paired_d1_summary(
            records, expected, expected_cells=expected_cells,
            manifest_digest=manifest["manifest_digest"],
            replay_verified=replay_verified,
        )
        pilot = output / "d1_pilot"
        write_json_atomic(pilot / "d1_exploratory_summary.json", summary)
        write_dynamic_verdict(pilot / "dynamic_verdict.md", summary)
        result["analysis_performed"] = True
        result["analysis_status"] = summary["status"]
    return result


def run_dynamic_mainline(
    stage: str,
    output: str | Path,
    workers: int,
    calibration_run_id: str,
    registered_config_id: str,
    *,
    scenarios: tuple[DynamicScenario, ...] | None = None,
    methods: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"stage must be one of {sorted(ALLOWED_STAGES)}; d2 is not authorized")
    if not 1 <= workers <= 22:
        raise ValueError("workers must lie in [1, 22]")
    destination = Path(output)
    if stage == "d0":
        return {"stage": "d0", **write_d0_artifacts(destination), "d2_authorized": False}
    if stage == "d1-reveal":
        return _reveal(destination, calibration_run_id)
    d0 = write_d0_artifacts(destination) if stage == "all-d0-d1" else None
    summary = _run_d1(
        destination, workers, calibration_run_id, registered_config_id,
        scenarios, methods,
    )
    if d0 is not None:
        summary["d0"] = d0
    return summary


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/dynamic_mainline"))
    parser.add_argument("--workers", type=int, default=max(1, min(22, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--calibration-run-id", default="0")
    parser.add_argument("--registered-config-id", default="recon_damage_plus_010_r2_a6_b3")
    args = parser.parse_args()
    result = run_dynamic_mainline(
        args.stage, args.output, args.workers,
        args.calibration_run_id, args.registered_config_id,
    )
    return 0 if result.get("status") != "FAILED/INCOMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
