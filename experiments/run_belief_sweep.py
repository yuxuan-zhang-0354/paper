"""Run the deterministic parallel belief-simplex decision-region sweep."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from math import isfinite
import os
from pathlib import Path
from typing import Iterable, Sequence

from numpy.typing import NDArray
import numpy as np

from uav_lifecycle.artifacts import write_json_atomic
from uav_lifecycle.attack import marginals
from uav_lifecycle.belief import bda_kernel, recon_kernel
from uav_lifecycle.rollout import action_values
from uav_lifecycle.scenarios import ValidationConfig, validation_parameter_sets
from uav_lifecycle.simplex import ACTION_ORDER, rank_actions, simplex_grid


CSV_COLUMNS = (
    "config_id",
    "b_ha",
    "b_hd",
    "b_la",
    "b_ld",
    "q_recon",
    "q_attack",
    "q_bda",
    "q_defer",
    "best_action",
    "second_action",
    "margin",
    "p_h",
    "p_alive",
)


def evaluate_config(
    config: ValidationConfig, grid: NDArray[np.float64]
) -> list[tuple[object, ...]]:
    """Evaluate one registered configuration on a shared simplex grid."""

    zr = recon_kernel(config.recon_class_matrix, config.recon_damage_matrix)
    zb = bda_kernel(config.bda_damage_matrix)
    rows: list[tuple[object, ...]] = []
    for belief in grid:
        values = action_values(belief, zr, zb, config.params)
        best, second, margin = rank_actions(values)
        p_h, p_alive = marginals(belief)
        rows.append(
            (
                config.config_id,
                float(belief[0]),
                float(belief[1]),
                float(belief[2]),
                float(belief[3]),
                values["recon"],
                values["attack"],
                values["bda"],
                values["defer"],
                best,
                second,
                margin,
                p_h,
                p_alive,
            )
        )
    return rows


def audit_decision_csv(
    path: str | Path,
    configs: Sequence[ValidationConfig],
    grid: NDArray[np.float64],
    tolerance: float = 1e-12,
) -> dict[str, object]:
    """Stream-read the persisted CSV and verify exact ordered grid coverage."""

    expected_record_count = len(configs) * len(grid)
    row_count = 0
    header_match = False
    bad_width = 0
    parse_errors = 0
    sequence_mismatches = 0
    nonfinite_values = 0
    ranking_mismatches = 0
    margin_mismatches = 0
    marginal_mismatches = 0
    missing_rows = 0
    extra_rows = 0
    action_counts: Counter[str] = Counter()

    with Path(path).open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        header = next(reader, None)
        header_match = header == list(CSV_COLUMNS)
        exhausted = False
        for config in configs:
            for expected_belief in grid:
                row = next(reader, None)
                if row is None:
                    exhausted = True
                    missing_rows = expected_record_count - row_count
                    break
                row_count += 1
                if len(row) != len(CSV_COLUMNS):
                    bad_width += 1
                    continue
                try:
                    belief = tuple(float(value) for value in row[1:5])
                    values = {
                        "recon": float(row[5]),
                        "attack": float(row[6]),
                        "bda": float(row[7]),
                        "defer": float(row[8]),
                    }
                    stored_margin = float(row[11])
                    stored_p_h = float(row[12])
                    stored_p_alive = float(row[13])
                except (TypeError, ValueError):
                    parse_errors += 1
                    continue
                if row[0] != config.config_id or any(
                    abs(actual - float(expected)) > tolerance
                    for actual, expected in zip(
                        belief, expected_belief, strict=True
                    )
                ):
                    sequence_mismatches += 1
                numeric_values = (*belief, *values.values(), stored_margin)
                if not all(isfinite(value) for value in numeric_values):
                    nonfinite_values += 1
                    continue
                best, second, expected_margin = rank_actions(values)
                if row[9] != best or row[10] != second:
                    ranking_mismatches += 1
                if abs(stored_margin - expected_margin) > tolerance:
                    margin_mismatches += 1
                expected_p_h = float(belief[0] + belief[1])
                expected_p_alive = float(belief[0] + belief[2])
                if not (
                    abs(stored_p_h - expected_p_h) <= tolerance
                    and abs(stored_p_alive - expected_p_alive) <= tolerance
                ):
                    marginal_mismatches += 1
                action_counts[row[9]] += 1
            if exhausted:
                break
        for _ in reader:
            extra_rows += 1
            row_count += 1

    failures = (
        int(not header_match)
        + bad_width
        + parse_errors
        + sequence_mismatches
        + nonfinite_values
        + ranking_mismatches
        + margin_mismatches
        + marginal_mismatches
        + missing_rows
        + extra_rows
    )
    return {
        "integrity_pass": bool(
            failures == 0 and row_count == expected_record_count
        ),
        "header_match": header_match,
        "expected_record_count": expected_record_count,
        "row_count": row_count,
        "bad_width": bad_width,
        "parse_errors": parse_errors,
        "sequence_mismatches": sequence_mismatches,
        "nonfinite_values": nonfinite_values,
        "ranking_mismatches": ranking_mismatches,
        "margin_mismatches": margin_mismatches,
        "marginal_mismatches": marginal_mismatches,
        "missing_rows": missing_rows,
        "extra_rows": extra_rows,
        "action_counts": {
            action: int(action_counts[action]) for action in ACTION_ORDER
        },
    }


def _evaluate_job(
    job: tuple[ValidationConfig, NDArray[np.float64]],
) -> list[tuple[object, ...]]:
    return evaluate_config(*job)


def _result_stream(
    configs: Sequence[ValidationConfig],
    grid: NDArray[np.float64],
    workers: int,
) -> Iterable[list[tuple[object, ...]]]:
    jobs = ((config, grid) for config in configs)
    if workers == 1:
        return map(_evaluate_job, jobs)
    executor = ProcessPoolExecutor(max_workers=workers)
    mapped = executor.map(_evaluate_job, jobs, chunksize=1)

    def consume() -> Iterable[list[tuple[object, ...]]]:
        try:
            yield from mapped
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    return consume()


def run_belief_sweep(
    step: float,
    output: str | Path,
    workers: int | None = None,
    configs: Sequence[ValidationConfig] | None = None,
) -> dict[str, object]:
    """Evaluate and persist a deterministic, parent-written decision sweep."""

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    worker_count = (
        min(22, max(1, (os.cpu_count() or 1) - 2))
        if workers is None
        else int(workers)
    )
    if worker_count < 1:
        raise ValueError("workers must be positive")
    selected_configs = tuple(
        validation_parameter_sets() if configs is None else configs
    )
    if not selected_configs:
        raise ValueError("at least one configuration is required")
    config_ids = [config.config_id for config in selected_configs]
    if len(set(config_ids)) != len(config_ids):
        raise ValueError("configuration IDs must be unique")

    grid = simplex_grid(step)
    expected_record_count = len(selected_configs) * len(grid)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        destination / "config.json",
        {
            "experiment": "belief_simplex_sweep",
            "step": float(step),
            "simplex_point_count": len(grid),
            "configuration_count": len(selected_configs),
            "expected_record_count": expected_record_count,
            "workers": worker_count,
            "action_tie_order": ACTION_ORDER,
            "state_order": ("HA", "HD", "LA", "LD"),
            "configurations": selected_configs,
        },
    )

    temporary_csv = destination / ".decision_regions.csv.tmp"
    final_csv = destination / "decision_regions.csv"
    by_config: dict[str, dict[str, int]] = {}
    total_counts: Counter[str] = Counter()
    actual_record_count = 0
    try:
        with temporary_csv.open("w", encoding="utf-8", newline="") as sink:
            writer = csv.writer(sink, lineterminator="\n")
            writer.writerow(CSV_COLUMNS)
            for config, rows in zip(
                selected_configs,
                _result_stream(selected_configs, grid, worker_count),
                strict=True,
            ):
                writer.writerows(rows)
                counts = Counter(str(row[9]) for row in rows)
                by_config[config.config_id] = {
                    action: int(counts[action]) for action in ACTION_ORDER
                }
                total_counts.update(counts)
                actual_record_count += len(rows)
        os.replace(temporary_csv, final_csv)
    finally:
        if temporary_csv.exists():
            temporary_csv.unlink()

    in_memory_total = {
        action: int(total_counts[action]) for action in ACTION_ORDER
    }
    persisted_audit = audit_decision_csv(
        final_csv, selected_configs, grid
    )
    total = dict(persisted_audit["action_counts"])
    audit_counts_match = total == in_memory_total
    integrity_pass = bool(
        actual_record_count == expected_record_count
        and len(by_config) == len(selected_configs)
        and sum(total.values()) == expected_record_count
        and persisted_audit["integrity_pass"]
        and audit_counts_match
    )
    all_actions_present = all(total[action] > 0 for action in ACTION_ORDER)
    counts_payload = {
        "by_config": by_config,
        "total": total,
        "all_actions_present": all_actions_present,
        "expected_record_count": expected_record_count,
        "actual_record_count": actual_record_count,
        "in_memory_action_counts": in_memory_total,
        "persisted_audit": persisted_audit,
        "audit_counts_match": audit_counts_match,
    }
    write_json_atomic(destination / "action_counts.json", counts_payload)
    summary: dict[str, object] = {
        "simplex_point_count": len(grid),
        "configuration_count": len(selected_configs),
        "expected_record_count": expected_record_count,
        "actual_record_count": actual_record_count,
        "integrity_pass": integrity_pass,
        "all_actions_present": all_actions_present,
        "gate_c_pass": integrity_pass and all_actions_present,
        "total_action_counts": total,
        "persisted_audit": persisted_audit,
    }
    write_json_atomic(destination / "summary.json", summary)
    (destination / "run.log").write_text(
        (
            f"belief sweep: step={step}, configs={len(selected_configs)}, "
            f"records={actual_record_count}, workers={worker_count}, "
            f"integrity_pass={integrity_pass}, "
            f"all_actions_present={all_actions_present}\n"
        ),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--output", type=Path, default=Path("results/belief_sweep")
    )
    args = parser.parse_args()
    summary = run_belief_sweep(args.step, args.output, args.workers)
    return 0 if summary["gate_c_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
