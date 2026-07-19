"""Run staged all-mode screening validation with parent-only artifact writes."""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
from statistics import mean

from uav_lifecycle.artifacts import write_json_atomic
from uav_lifecycle.mode_fallback import EPSILON_SCORE
from uav_lifecycle.third_batch import evaluate_mode_instance, random_mode_instance, tier0_mode_instances


def _parse_cells(value: str) -> tuple[tuple[int, int], ...]:
    cells = tuple(tuple(int(part) for part in token.lower().split("x")) for token in value.split(","))
    if not cells or any(len(cell) != 2 or cell[0] not in (2, 3, 4) or cell[1] not in (3, 4, 5) for cell in cells):
        raise ValueError("cells must be comma-separated NxM pairs with N in 2..4 and M in 3..5")
    return cells


def _evaluate_job(job):
    instance, metadata = job
    return {**evaluate_mode_instance(instance), **metadata}


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(records[0]) if records else ()
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict, tuple)) else value
                    for key, value in record.items()
                }
            )
    os.replace(temporary, path)


def _summarize(
    records: list[dict[str, object]], epsilon_score: float = EPSILON_SCORE
) -> dict[str, object]:
    if epsilon_score < 0:
        raise ValueError("epsilon_score must be nonnegative")
    g1_failures = sum(
        float(record["all_mode_score"]) + 1e-9 < float(record["fixed_mode_score"])
        or abs(
            float(record["screening_loss"])
            - (float(record["all_mode_score"]) - float(record["fixed_mode_score"]))
        ) > 1e-9
        for record in records
    )
    # G2 is the Johnson theory/interface gate. Full-Rebuild-Raw is a
    # deliberately non-DMG ablation and its cycles are reported separately.
    g2_failures = sum(int(record["johnson_gate_failures"]) for record in records)
    full_raw_nonconverged = sum(record["full_raw_status"] != "converged" for record in records)
    screened = sum(int(record["screened_task_count"]) for record in records)
    orphans = sum(int(record["orphan_count"]) for record in records)
    compared_modes = sum(
        sum(mode is not None for mode in record["central_modes"])
        for record in records
    )
    substitutions = sum(int(record["mode_substitutions"]) for record in records)
    ratios = [float(record["johnson_ratio"]) for record in records]
    decomposition_records = [record for record in records if bool(record.get("decomposition_valid", True))]
    summary = {
        "record_count": len(records),
        "gate_g1_pass": g1_failures == 0,
        "gate_g1_failures": g1_failures,
        "gate_g2_pass": g2_failures == 0,
        "gate_g2_failures": g2_failures,
        "full_raw_nonconvergence_rate": full_raw_nonconverged / len(records) if records else 0.0,
        "decomposition_valid_rate": len(decomposition_records) / len(records) if records else 1.0,
        "mean_johnson_ratio": mean(ratios) if ratios else 1.0,
        "task_orphan_rate": orphans / screened if screened else 0.0,
        "mode_substitution_rate": substitutions / compared_modes if compared_modes else 0.0,
        "mean_screening_loss": mean(float(record["screening_loss"]) for record in records) if records else 0.0,
        "mean_allocation_loss": mean(float(record["allocation_loss"]) for record in decomposition_records) if decomposition_records else None,
        "mean_warping_loss": mean(float(record["warping_loss"]) for record in decomposition_records) if decomposition_records else None,
    }
    fallback_records = [
        record for record in records if "fallback_base_target_count" in record
    ]
    total_base_targets = sum(
        int(record["fallback_base_target_count"]) for record in fallback_records
    )
    total_base_orphans = sum(
        int(record["base_orphan_count"]) for record in fallback_records
    )
    total_fallback_unresolved = sum(
        int(record["fallback_unresolved_count"]) for record in fallback_records
    )
    total_resolved = sum(int(record["resolved_count"]) for record in fallback_records)
    gains = [
        float(record["fallback_gain"])
        for record in fallback_records
        if record.get("fallback_gain") is not None
    ]
    wall_times = [
        float(record["fallback_wall_clock_seconds"])
        for record in fallback_records
    ]
    summary.update(
        {
            "fallback_record_count": len(fallback_records),
            "fallback_base_target_count": total_base_targets,
            "base_orphan_count": total_base_orphans,
            "base_orphan_rate": (
                total_base_orphans / total_base_targets
                if total_base_targets
                else None
            ),
            "fallback_unresolved_count": total_fallback_unresolved,
            "fallback_unresolved_rate": (
                total_fallback_unresolved / total_base_targets
                if total_base_targets
                else None
            ),
            "resolved_count": total_resolved,
            "resolution_rate": (
                total_resolved / total_base_orphans
                if total_base_orphans
                else None
            ),
            "newly_unassigned_count": sum(
                int(record.get("newly_unassigned_count", 0))
                for record in fallback_records
            ),
            "fallback_gate_failures": sum(
                int(record["fallback_gate_failures"]) for record in fallback_records
            ),
            "mean_fallback_gain": mean(gains) if gains else None,
            "minimum_fallback_gain": min(gains) if gains else None,
            "fallback_win_count": sum(gain > epsilon_score for gain in gains),
            "fallback_tie_count": sum(abs(gain) <= epsilon_score for gain in gains),
            "fallback_loss_count": sum(gain < -epsilon_score for gain in gains),
            "selected_mode_switches": sum(
                int(record["selected_mode_switches"])
                for record in fallback_records
            ),
            "selected_defer_count": sum(
                int(record.get("selected_defer_count", 0))
                for record in fallback_records
            ),
            "search_advances": sum(
                int(record["search_advances"]) for record in fallback_records
            ),
            "fallback_wall_clock_seconds": sum(wall_times),
            "mean_fallback_wall_clock_seconds": (
                mean(wall_times) if wall_times else None
            ),
        }
    )
    return summary


def run_third_batch(
    stage: str,
    output: str | Path,
    workers: int | None = None,
    per_cell: int = 20,
    cells: tuple[tuple[int, int], ...] | None = None,
    belief_profile: str = "uniform",
    seed_start: int = 0,
) -> dict[str, object]:
    if stage not in {"tier0", "pilot", "all"}:
        raise ValueError("stage must be tier0, pilot, or all")
    if per_cell < 1:
        raise ValueError("per_cell must be positive")
    if seed_start < 0:
        raise ValueError("seed_start must be nonnegative")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    worker_count = min(22, max(1, (os.cpu_count() or 1) - 2)) if workers is None else int(workers)
    if worker_count < 1 or worker_count > 22:
        raise ValueError("workers must lie in [1, 22]")

    jobs = [
        (instance, {"stage": "tier0", "ammo_tightness": "directed", "horizon_tightness": "directed", "belief_profile": "directed", "seed": -1})
        for instance in tier0_mode_instances()
    ]
    if stage in {"pilot", "all"}:
        selected_cells = cells or ((2, 3),)
        for n_agents, n_targets in selected_cells:
            for ammo in ("tight", "medium", "loose"):
                for horizon in ("tight", "medium", "loose"):
                    for continuation in ("optimistic", "no_continuation", "ammo_reachability_gate"):
                        for seed in range(seed_start, seed_start + per_cell):
                            jobs.append(
                                (
                                    random_mode_instance(
                                        n_agents,
                                        n_targets,
                                        seed,
                                        ammo,
                                        horizon,
                                        continuation,
                                        belief_profile,
                                    ),
                                    {
                                        "stage": "pilot",
                                        "n_agents": n_agents,
                                        "n_targets": n_targets,
                                        "ammo_tightness": ammo,
                                        "horizon_tightness": horizon,
                                        "belief_profile": belief_profile,
                                        "seed": seed,
                                    },
                                )
                            )
    if worker_count == 1:
        records = [_evaluate_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            records = list(executor.map(_evaluate_job, jobs, chunksize=1))

    tier0_records = [record for record in records if record["stage"] == "tier0"]
    _write_records(destination / "iteration0" / "records.csv", tier0_records)
    if stage in {"pilot", "all"}:
        _write_records(destination / "pilot" / "records.csv", [record for record in records if record["stage"] == "pilot"])
    summary = _summarize(records)
    write_json_atomic(destination / "summary.json", summary)
    manifest = {
        "experiment": "third_batch_all_mode_screening",
        "stage": stage,
        "record_count": len(records),
        "workers": worker_count,
        "per_cell": per_cell,
        "belief_profile": belief_profile,
        "seed_start": seed_start,
        "cells": list(cells or ((2, 3),)) if stage in {"pilot", "all"} else [],
        "calibration_confirmation_isolation": True,
        "ranked_fallback_implemented": False,
        "summary": summary,
    }
    write_json_atomic(destination / "third_batch_manifest.json", manifest)
    verdict = (
        "# Third-batch staged verdict\n\n"
        f"- Stage: `{stage}`\n"
        f"- Records: {len(records)}\n"
        f"- Gate G1: {'PASS' if summary['gate_g1_pass'] else 'FAIL'}\n"
        f"- Gate G2: {'PASS' if summary['gate_g2_pass'] else 'FAIL'}\n"
        f"- Mean Johnson/all-mode ratio: {summary['mean_johnson_ratio']:.6f}\n"
        f"- Task orphan rate: {summary['task_orphan_rate']:.6f}\n"
        f"- Mode substitution rate: {summary['mode_substitution_rate']:.6f}\n"
        "\nRanked fallback remains unimplemented pending the pre-registered locked-confirmation trigger.\n"
    )
    (destination / "third_batch_verdict.md").write_text(verdict, encoding="utf-8")
    return summary


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("tier0", "pilot", "all"), default="tier0")
    parser.add_argument("--output", type=Path, default=Path("results/third_batch"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--per-cell", type=int, default=20)
    parser.add_argument("--belief-profile", choices=("uniform", "stratified"), default="uniform")
    parser.add_argument("--cells", default="2x3")
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()
    summary = run_third_batch(
        args.stage,
        args.output,
        args.workers,
        args.per_cell,
        cells=_parse_cells(args.cells),
        belief_profile=args.belief_profile,
        seed_start=args.seed_start,
    )
    return 0 if summary["gate_g1_pass"] and summary["gate_g2_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
