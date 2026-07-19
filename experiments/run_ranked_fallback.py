"""Run the pre-registered ranked-fallback F0/F1 experiment stages."""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import json
import os
from pathlib import Path
from statistics import mean, median
from tempfile import NamedTemporaryFile
from typing import Iterable

from experiments.run_third_batch import _summarize
from uav_lifecycle.artifacts import write_json_atomic
from uav_lifecycle.mode_fallback import (
    EPSILON_SCORE,
    rank_mode_candidates,
    run_ranked_fallback,
)
from uav_lifecycle.third_batch import (
    evaluate_fallback_instance,
    fallback_gate_telemetry,
    random_mode_instance,
    tier0_mode_instances,
)


F1_CELLS = ((2, 4), (3, 4), (3, 5))
F1_SEEDS = tuple(range(1000, 1020))
LEVELS = ("tight", "medium", "loose")
CONTINUATIONS = ("optimistic", "no_continuation", "ammo_reachability_gate")
F1_PROFILE = "stratified"
MINIMUM_GAIN = -1e-9
MAX_MEAN_FALLBACK_SECONDS = 1.0
SPEC_PATH = Path("docs/superpowers/specs/2026-07-14-ranked-mode-fallback-design.md")
SOURCE_HOLDOUT_PATH = Path("results/third_batch/holdout_screen")
DEFAULT_F0_MANIFEST = Path("results/third_batch/fallback/tier_f0/fallback_manifest.json")


def build_ranked_fallback_jobs(stage: str) -> list[tuple[object, dict[str, object]]]:
    """Build frozen F0/F1 jobs without evaluating or writing artifacts."""

    if stage not in {"tier_f0", "tier_f1", "all"}:
        raise ValueError("stage must be tier_f0, tier_f1, or all")
    jobs: list[tuple[object, dict[str, object]]] = []
    if stage in {"tier_f0", "all"}:
        jobs.extend(
            (
                instance,
                {
                    "stage": "tier_f0",
                    "cell": [len(instance.agents), len(instance.tasks_by_target)],
                    "ammo_tightness": "directed",
                    "horizon_tightness": "directed",
                    "continuation_profile": instance.continuation,
                    "belief_profile": "directed",
                    "seed": -1,
                },
            )
            for instance in tier0_mode_instances()
        )
    if stage in {"tier_f1", "all"}:
        for n_agents, n_targets in F1_CELLS:
            for ammo in LEVELS:
                for horizon in LEVELS:
                    for continuation in CONTINUATIONS:
                        for seed in F1_SEEDS:
                            jobs.append(
                                (
                                    random_mode_instance(
                                        n_agents,
                                        n_targets,
                                        seed,
                                        ammo,
                                        horizon,
                                        continuation,
                                        F1_PROFILE,
                                    ),
                                    {
                                        "stage": "tier_f1",
                                        "cell": [n_agents, n_targets],
                                        "ammo_tightness": ammo,
                                        "horizon_tightness": horizon,
                                        "continuation_profile": continuation,
                                        "belief_profile": F1_PROFILE,
                                        "seed": seed,
                                    },
                                )
                            )
    return jobs


def _evaluate_indexed_job(indexed_job):
    index, (instance, metadata) = indexed_job
    candidate_count = sum(len(row) for row in rank_mode_candidates(instance))
    return index, {
        **evaluate_fallback_instance(instance),
        "candidate_count": candidate_count,
        "theoretical_call_bound": 1 + candidate_count,
        **metadata,
    }


def _evaluate_jobs(
    jobs: list[tuple[object, dict[str, object]]], workers: int
) -> tuple[list[dict[str, object]], bool]:
    """Evaluate jobs while bounding in-flight work and stopping on a Gate failure."""

    if workers == 1:
        records = []
        for indexed_job in enumerate(jobs):
            _, record = _evaluate_indexed_job(indexed_job)
            records.append(record)
            if int(record["fallback_gate_failures"]):
                return records, True
        return records, False

    records_by_index: dict[int, dict[str, object]] = {}
    stopped = False
    executor = ProcessPoolExecutor(max_workers=workers)
    pending = {}
    jobs_iter = iter(enumerate(jobs))
    try:
        for indexed_job in jobs_iter:
            future = executor.submit(_evaluate_indexed_job, indexed_job)
            pending[future] = indexed_job[0]
            if len(pending) == workers:
                break
        while pending and not stopped:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                index, record = future.result()
                records_by_index[index] = record
                if int(record["fallback_gate_failures"]):
                    stopped = True
                    break
                try:
                    indexed_job = next(jobs_iter)
                except StopIteration:
                    continue
                new_future = executor.submit(_evaluate_indexed_job, indexed_job)
                pending[new_future] = indexed_job[0]
        if stopped:
            for future in pending:
                future.cancel()
    finally:
        executor.shutdown(wait=True, cancel_futures=stopped)
    return [records_by_index[index] for index in sorted(records_by_index)], stopped


def _write_records_atomic(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(records[0]) if records else ()
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            writer = csv.DictWriter(temporary, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        key: json.dumps(value, sort_keys=True)
                        if isinstance(value, (list, dict, tuple))
                        else value
                        for key, value in record.items()
                    }
                )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stable_record(record: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in record.items() if key != "fallback_wall_clock_seconds"}


def _compare_f0_replay(
    first: list[dict[str, object]],
    replay: list[dict[str, object]],
    replay_gate_stopped: bool,
) -> dict[str, object]:
    """Compare possibly partial replay output without assuming equal lengths."""

    paired_failures = sum(
        _stable_record(original) != _stable_record(repeated)
        for original, repeated in zip(first, replay)
    )
    return {
        "replay_failures": paired_failures + abs(len(first) - len(replay)),
        "replay_gate_failure": bool(replay_gate_stopped),
    }


def _augment_summary(
    records: list[dict[str, object]],
    replay_failures: int | None,
    replay_gate_failures: int | None = 0,
) -> dict[str, object]:
    summary = _summarize(records)
    runtimes = [float(record["fallback_wall_clock_seconds"]) for record in records]
    summary.update(
        {
            "deterministic_replay_failures": replay_failures,
            "replay_gate_failures": replay_gate_failures,
            "finite_iteration_bound_failures": sum(
                int(record["fallback_total_johnson_calls"])
                > int(record["theoretical_call_bound"])
                or int(record["theoretical_call_bound"])
                != 1 + int(record["candidate_count"])
                for record in records
            ),
            "runtime_distribution_seconds": {
                "minimum": min(runtimes) if runtimes else None,
                "median": median(runtimes) if runtimes else None,
                "p95": _percentile(runtimes, 0.95),
                "maximum": max(runtimes) if runtimes else None,
                "mean": mean(runtimes) if runtimes else None,
            },
        }
    )
    return summary


def _f2_summary(stage: str, records: list[dict[str, object]]) -> dict[str, object]:
    """Return the independent F1-only evidence used for F2 decisions."""

    selected = (
        [record for record in records if record.get("stage") == "tier_f1"]
        if stage in {"tier_f1", "all"}
        else records
    )
    full = _augment_summary(selected, None, None)
    fields = (
        "record_count",
        "fallback_record_count",
        "fallback_base_target_count",
        "base_orphan_count",
        "base_orphan_rate",
        "fallback_unresolved_count",
        "fallback_unresolved_rate",
        "resolved_count",
        "resolution_rate",
        "newly_unassigned_count",
        "selected_defer_count",
        "fallback_gate_failures",
        "finite_iteration_bound_failures",
        "mean_fallback_gain",
        "minimum_fallback_gain",
        "fallback_win_count",
        "fallback_tie_count",
        "fallback_loss_count",
        "runtime_distribution_seconds",
    )
    return {field: full[field] for field in fields}


def _runner_exit_code(status: str) -> int:
    """Map only a complete runner status to a successful CLI exit."""

    return 0 if status == "COMPLETE" else 1


def _replay_manifest_value(stage: str, replay_failures: int) -> int | None:
    """Return null when deterministic replay was not run for the stage."""

    return None if stage == "tier_f1" else replay_failures


def _f0_hard_gate_status(summary: dict[str, object]) -> str:
    """Classify the first failed F0 hard-gate condition."""

    if int(summary["fallback_gate_failures"]) or int(
        summary.get("replay_gate_failures", 0)
    ):
        return "STOPPED_GATE_FAILURE"
    if int(summary["deterministic_replay_failures"]):
        return "STOPPED_REPLAY_MISMATCH"
    if int(summary["finite_iteration_bound_failures"]):
        return "STOPPED_FINITE_BOUND_FAILURE"
    return "COMPLETE"


def _validate_f0_prerequisite(path: str | Path) -> dict[str, object]:
    """Load and validate the successful F0 manifest required by standalone F1."""

    prerequisite = Path(path)
    if not prerequisite.is_file():
        raise FileNotFoundError(f"Tier F0 prerequisite manifest not found: {prerequisite}")
    manifest = json.loads(prerequisite.read_text(encoding="utf-8"))
    summary = manifest.get("summary", {})
    valid = (
        manifest.get("stage") == "tier_f0"
        and manifest.get("status") == "COMPLETE"
        and summary.get("fallback_gate_failures") == 0
        and summary.get("deterministic_replay_failures") == 0
        and summary.get("finite_iteration_bound_failures") == 0
    )
    if not valid:
        raise RuntimeError(f"standalone Tier F1 requires a passing Tier F0 manifest: {prerequisite}")
    return manifest


def _f2_decision(
    stage: str, completed: bool, summary: dict[str, object]
) -> dict[str, object]:
    """Apply only pre-registered decision criteria; do not invent effect size."""

    if stage not in {"tier_f1", "all"}:
        return {
            "proceed_to_locked_confirmation": False,
            "f2_decision": "not_applicable_before_tier_f1",
            "effect_evidence": "tier_f1 not evaluated",
        }
    runtime_mean = summary["runtime_distribution_seconds"]["mean"]
    prerequisites_met = bool(
        completed
        and summary["fallback_gate_failures"] == 0
        and summary["minimum_fallback_gain"] is not None
        and float(summary["minimum_fallback_gain"]) >= MINIMUM_GAIN
        and runtime_mean is not None
        and float(runtime_mean) <= MAX_MEAN_FALLBACK_SECONDS
    )
    if not prerequisites_met:
        return {
            "proceed_to_locked_confirmation": False,
            "f2_decision": "prerequisites_not_met",
            "effect_evidence": (
                f"resolved {summary['resolved_count']}/{summary['base_orphan_count']}"
            ),
        }
    return {
        "proceed_to_locked_confirmation": False,
        "f2_decision": "effect_criterion_not_preregistered",
        "effect_evidence": (
            f"strict decrease only: {summary['resolved_count']}/{summary['base_orphan_count']}"
        ),
    }


def _verdict_text(
    stage: str,
    summary: dict[str, object],
    decision: dict[str, object],
    decision_summary: dict[str, object] | None = None,
) -> str:
    evidence = decision_summary or summary
    runtime = evidence["runtime_distribution_seconds"]
    return (
        "# Ranked fallback staged verdict\n\n"
        f"- Stage: `{stage}`\n"
        f"- Records: {summary['record_count']}\n"
        f"- Fallback Gate failures: {summary['fallback_gate_failures']}\n"
        f"- F2 evidence scope: `{'tier_f1_only' if stage in {'tier_f1', 'all'} else stage}`\n"
        f"- Base orphan rate: {evidence['base_orphan_rate']}\n"
        f"- Fallback unresolved rate: {evidence['fallback_unresolved_rate']}\n"
        f"- Resolved / newly unassigned: {evidence['resolved_count']} / {evidence['newly_unassigned_count']}\n"
        f"- Selected assigned-to-Defer changes: {evidence['selected_defer_count']}\n"
        f"- Paired score win / tie / loss: {evidence['fallback_win_count']} / {evidence['fallback_tie_count']} / {evidence['fallback_loss_count']}\n"
        f"- Minimum paired gain: {evidence['minimum_fallback_gain']}\n"
        f"- Runtime seconds (min / median / p95 / max): {runtime['minimum']} / {runtime['median']} / {runtime['p95']} / {runtime['maximum']}\n"
        f"- CPU runtime acceptable (mean <= {MAX_MEAN_FALLBACK_SECONDS}s): "
        f"{runtime['mean'] is not None and runtime['mean'] <= MAX_MEAN_FALLBACK_SECONDS}\n"
        f"- F2 decision: `{decision['f2_decision']}`\n"
        f"- Effect evidence: {decision['effect_evidence']}\n"
        f"- `proceed_to_locked_confirmation`: "
        f"`{str(decision['proceed_to_locked_confirmation']).lower()}`\n\n"
        "The observed arithmetic strict decrease is not called a pre-registered "
        "measurable effect: no effect-size or uncertainty criterion was registered.\n\n"
        "This runner does not implement or launch the 100-seed locked confirmation.\n"
    )


def run_ranked_fallback_stage(
    stage: str,
    output: str | Path,
    workers: int | None = None,
    prerequisite_manifest: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate one frozen fallback stage and write parent-owned artifacts."""

    if stage not in {"tier_f0", "tier_f1", "all"}:
        raise ValueError("stage must be tier_f0, tier_f1, or all")
    worker_count = (
        min(22, max(1, (os.cpu_count() or 1) - 2))
        if workers is None
        else int(workers)
    )
    if not 1 <= worker_count <= 22:
        raise ValueError("workers must lie in [1, 22]")

    root = Path(__file__).resolve().parents[1]
    prerequisite_path: Path | None = None
    if stage == "tier_f1":
        prerequisite_path = Path(prerequisite_manifest or (root / DEFAULT_F0_MANIFEST))
        _validate_f0_prerequisite(prerequisite_path)

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    f0_jobs = build_ranked_fallback_jobs("tier_f0") if stage in {"tier_f0", "all"} else []
    f1_jobs = build_ranked_fallback_jobs("tier_f1") if stage in {"tier_f1", "all"} else []
    records, stopped = _evaluate_jobs(f0_jobs, worker_count) if f0_jobs else ([], False)
    replay_failures = 0
    replay_gate_failures = 0
    if f0_jobs and not stopped:
        replay_records, replay_stopped = _evaluate_jobs(f0_jobs, 1)
        replay_audit = _compare_f0_replay(
            records, replay_records, replay_gate_stopped=replay_stopped
        )
        replay_failures = int(replay_audit["replay_failures"])
        replay_gate_failures = int(bool(replay_audit["replay_gate_failure"]))
        stopped = replay_gate_failures > 0 or replay_failures > 0
    if f0_jobs:
        f0_summary = _augment_summary(
            records, replay_failures, replay_gate_failures
        )
        f0_status = _f0_hard_gate_status(f0_summary)
        stopped = f0_status != "COMPLETE"
    else:
        f0_status = "NOT_RUN"
    if f1_jobs and not stopped:
        f1_records, stopped = _evaluate_jobs(f1_jobs, worker_count)
        records.extend(f1_records)

    replay_audit = _replay_manifest_value(stage, replay_failures)
    replay_gate_audit = None if stage == "tier_f1" else replay_gate_failures
    summary = _augment_summary(records, replay_audit, replay_gate_audit)
    includes_f1 = stage in {"tier_f1", "all"}
    completed = len(records) == len(f0_jobs) + len(f1_jobs)
    if f0_status != "NOT_RUN" and f0_status != "COMPLETE":
        status = f0_status
    elif stopped and int(summary["fallback_gate_failures"]):
        status = "STOPPED_GATE_FAILURE"
    elif int(summary["finite_iteration_bound_failures"]):
        status = "STOPPED_FINITE_BOUND_FAILURE"
    else:
        status = "COMPLETE"
    summary["runner_status"] = status
    decision_summary = _f2_summary(stage, records)
    decision = _f2_decision(
        stage, completed and status == "COMPLETE", decision_summary
    )

    parameters = {
        "stage": stage,
        "workers": worker_count,
        "cells": [list(cell) for cell in F1_CELLS] if includes_f1 else [],
        "ammo_tightness": list(LEVELS) if includes_f1 else ["directed"],
        "horizon_tightness": list(LEVELS) if includes_f1 else ["directed"],
        "continuations": list(CONTINUATIONS) if includes_f1 else ["directed"],
        "belief_profile": F1_PROFILE if includes_f1 else "directed",
        "seeds": list(F1_SEEDS) if includes_f1 else [],
    }
    manifest = {
        "experiment": "ranked_mode_fallback",
        "stage": stage,
        "status": status,
        "fallback_trigger": "base task orphan rate > 1%",
        "ranked_fallback_implemented": True,
        "spec_path": str((root / SPEC_PATH).resolve()),
        "source_holdout_path": str((root / SOURCE_HOLDOUT_PATH).resolve()),
        "source_holdout_manifest": str(
            (root / SOURCE_HOLDOUT_PATH / "third_batch_manifest.json").resolve()
        ),
        "tolerances": {
            "score_epsilon": EPSILON_SCORE,
            "minimum_gain": MINIMUM_GAIN,
            "maximum_mean_fallback_seconds": MAX_MEAN_FALLBACK_SECONDS,
        },
        "parameters": parameters,
        "record_count": len(records),
        "expected_record_count": len(f0_jobs) + len(f1_jobs),
        "deterministic_replay_failures": replay_audit,
        "replay_gate_failures": replay_gate_audit,
        "f0_prerequisite_manifest": (
            str(prerequisite_path.resolve()) if prerequisite_path is not None else None
        ),
        **decision,
        "locked_confirmation_launched": False,
        "summary": summary,
        "f2_summary": decision_summary,
    }
    _write_records_atomic(destination / "records.csv", records)
    write_json_atomic(destination / "summary.json", summary)
    write_json_atomic(destination / "fallback_manifest.json", manifest)
    _write_text_atomic(
        destination / "fallback_verdict.md",
        _verdict_text(stage, summary, decision, decision_summary),
    )
    return summary


def rewrite_existing_fallback_artifacts(
    stage: str,
    output: str | Path,
    prerequisite_manifest: str | Path | None = None,
) -> dict[str, object]:
    """Upgrade audit metadata without re-running measured fallback evaluations."""

    if stage not in {"tier_f0", "tier_f1"}:
        raise ValueError("existing artifact rewrite supports tier_f0 or tier_f1")
    destination = Path(output)
    records_path = destination / "records.csv"
    summary_path = destination / "summary.json"
    manifest_path = destination / "fallback_manifest.json"
    with records_path.open(encoding="utf-8", newline="") as source:
        records: list[dict[str, object]] = list(csv.DictReader(source))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = build_ranked_fallback_jobs(stage)
    if len(records) != len(jobs):
        raise RuntimeError(
            f"cannot rewrite {stage}: found {len(records)} records, expected {len(jobs)}"
        )

    for record, (instance, metadata) in zip(records, jobs, strict=True):
        expected_keys = {
            "instance_id": instance.instance_id,
            "stage": metadata["stage"],
            "ammo_tightness": metadata["ammo_tightness"],
            "horizon_tightness": metadata["horizon_tightness"],
            "continuation_profile": metadata["continuation_profile"],
            "belief_profile": metadata["belief_profile"],
        }
        if any(record[key] != str(value) for key, value in expected_keys.items()):
            raise RuntimeError(f"cannot rewrite {stage}: frozen job key mismatch")
        if int(record["seed"]) != int(metadata["seed"]):
            raise RuntimeError(f"cannot rewrite {stage}: frozen seed mismatch")
        if json.loads(record["cell"]) != metadata["cell"]:
            raise RuntimeError(f"cannot rewrite {stage}: frozen cell mismatch")

        candidate_count = sum(len(row) for row in rank_mode_candidates(instance))
        record["candidate_count"] = candidate_count
        record["theoretical_call_bound"] = 1 + candidate_count
        fallback = run_ranked_fallback(instance)
        telemetry = fallback_gate_telemetry(fallback)
        for aggregate_key in (
            "fallback_gate_failures",
            "fallback_base_gate_failures",
            "fallback_late_gate_failures",
        ):
            if int(record[aggregate_key]) != telemetry[aggregate_key]:
                raise RuntimeError(
                    f"cannot rewrite {stage}: measured {aggregate_key} changed"
                )
        record.update(telemetry)
        record["selected_defer_count"] = len(fallback.selected_defers)
        record["selected_defer_targets"] = list(fallback.selected_defers)
    finite_failures = sum(
        int(record["fallback_total_johnson_calls"])
        > int(record["theoretical_call_bound"])
        for record in records
    )
    replay_audit = (
        int(summary.get("deterministic_replay_failures", 0))
        if stage == "tier_f0"
        else None
    )
    summary["deterministic_replay_failures"] = replay_audit
    replay_gate_audit = 0 if stage == "tier_f0" else None
    summary["replay_gate_failures"] = replay_gate_audit
    summary["finite_iteration_bound_failures"] = finite_failures
    summary["selected_defer_count"] = sum(
        int(record["selected_defer_count"]) for record in records
    )
    if stage == "tier_f0":
        status = _f0_hard_gate_status(summary)
        prerequisite_path = None
    else:
        root = Path(__file__).resolve().parents[1]
        prerequisite_path = Path(prerequisite_manifest or (root / DEFAULT_F0_MANIFEST))
        _validate_f0_prerequisite(prerequisite_path)
        if int(summary["fallback_gate_failures"]):
            status = "STOPPED_GATE_FAILURE"
        elif finite_failures:
            status = "STOPPED_FINITE_BOUND_FAILURE"
        else:
            status = "COMPLETE"
    summary["runner_status"] = status
    decision_summary = _f2_summary(stage, records)
    decision = _f2_decision(stage, status == "COMPLETE", decision_summary)

    manifest.update(
        {
            "status": status,
            "deterministic_replay_failures": replay_audit,
            "replay_gate_failures": replay_gate_audit,
            "f0_prerequisite_manifest": (
                str(prerequisite_path.resolve())
                if prerequisite_path is not None
                else None
            ),
            **decision,
            "locked_confirmation_launched": False,
            "summary": summary,
            "f2_summary": decision_summary,
        }
    )
    _write_records_atomic(records_path, records)
    write_json_atomic(summary_path, summary)
    write_json_atomic(manifest_path, manifest)
    _write_text_atomic(
        destination / "fallback_verdict.md",
        _verdict_text(stage, summary, decision, decision_summary),
    )
    return summary


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("tier_f0", "tier_f1", "all"), default="tier_f0")
    parser.add_argument("--output", type=Path, default=Path("results/third_batch/fallback/tier_f0"))
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--f0-manifest",
        type=Path,
        default=DEFAULT_F0_MANIFEST,
        help="passing Tier F0 manifest required by standalone tier_f1",
    )
    args = parser.parse_args()
    summary = run_ranked_fallback_stage(
        args.stage,
        args.output,
        args.workers,
        prerequisite_manifest=args.f0_manifest,
    )
    return _runner_exit_code(str(summary["runner_status"]))


if __name__ == "__main__":
    raise SystemExit(main())
