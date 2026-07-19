from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from random import Random
from statistics import mean, median

from uav_lifecycle.warp_stress import (
    SelectedStratum,
    calibration_cases,
    confirmation_cases,
    confirmation_coverage,
    evaluate_stress_case,
    select_strata,
)


def _evaluate(case):
    return evaluate_stress_case(case)


def _run_cases(cases, workers: int) -> list[dict]:
    records: list[dict] = []
    if workers == 1:
        records.extend(map(_evaluate, cases))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records.extend(pool.map(_evaluate, cases, chunksize=8))
    records.sort(key=lambda record: record["case_id"])
    return records


def _bootstrap_ci(values: list[float], repetitions: int = 2000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = Random(7132026)
    samples = sorted(mean(rng.choice(values) for _ in values) for _ in range(repetitions))
    return [samples[int(0.025 * repetitions)], samples[min(repetitions - 1, int(0.975 * repetitions))]]


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round(probability * (len(ordered) - 1))]


def summarize(records: list[dict]) -> dict:
    deltas = [float(record["delta_j"]) for record in records]
    residuals = [float(record.get("delta_identity_error", 0.0)) for record in records]
    return {
        "instance_count": len(records),
        "activation_rate": sum(record["gap"] > 1e-9 for record in records) / len(records) if records else 0.0,
        "allocation_change_rate": sum(bool(record["allocation_changed"]) for record in records) / len(records) if records else 0.0,
        "target_winner_change_rate": sum(bool(record["target_winner_changed"]) for record in records) / len(records) if records else 0.0,
        "warping_decisive_rate": sum(bool(record["warping_decisive"]) for record in records) / len(records) if records else 0.0,
        "johnson_gate_e_failures": sum(int(record["johnson_gate_e_failures"]) for record in records),
        "delta_j_mean": mean(deltas) if deltas else 0.0,
        "delta_j_median": median(deltas) if deltas else 0.0,
        "delta_j_p05": _quantile(deltas, 0.05),
        "delta_j_p95": _quantile(deltas, 0.95),
        "delta_j_bootstrap_95ci": _bootstrap_ci(deltas),
        "win_tie_loss": {
            "win": sum(delta > 1e-9 for delta in deltas),
            "tie": sum(abs(delta) <= 1e-9 for delta in deltas),
            "loss": sum(delta < -1e-9 for delta in deltas),
        },
        "delta_identity_exact_rate": sum(abs(value) <= 1e-9 for value in residuals) / len(residuals) if residuals else 0.0,
        "interaction_residual_negative_count": sum(value < -1e-9 for value in residuals),
        "interaction_residual_positive_count": sum(value > 1e-9 for value in residuals),
        "interaction_residual_mean": mean(residuals) if residuals else 0.0,
        "interaction_residual_min": min(residuals) if residuals else 0.0,
        "interaction_residual_max": max(residuals) if residuals else 0.0,
    }


def _write_records(directory: Path, records: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if not records:
        (directory / "records.csv").write_text("", encoding="utf-8")
        return
    fields = list(records[0])
    with (directory / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key, value in row.items():
                if isinstance(value, (tuple, list, dict)):
                    row[key] = json.dumps(value, separators=(",", ":"))
            writer.writerow(row)


def _write_stage(output: Path, stage: str, records: list[dict]) -> dict:
    directory = output / stage
    _write_records(directory, records)
    summary = summarize(records)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _save_selected(output: Path, selected: tuple[SelectedStratum, ...]) -> None:
    payload = [{**asdict(item), "key": list(item.key)} for item in selected]
    (output / "tuned_ranges.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_selected(output: Path) -> tuple[SelectedStratum, ...]:
    payload = json.loads((output / "tuned_ranges.json").read_text(encoding="utf-8"))
    return tuple(SelectedStratum(tuple(item["key"]), item["count"], item["allocation_change_rate"], item["decisive_rate"], item["base_count"]) for item in payload)


def _existing_summaries(output: Path) -> dict[str, dict]:
    result = {}
    for stage in ("smoke", "calibration", "confirmation"):
        path = output / stage / "summary.json"
        if path.exists():
            result[stage] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _write_verdict(output: Path, summaries: dict[str, dict], selected_count: int) -> None:
    confirmation = summaries.get("confirmation")
    if confirmation is None:
        narrative = "确认集尚未运行。"
    elif confirmation["allocation_change_rate"] >= 0.05:
        narrative = "确认集分配改变率不少于 5%，stress corpus 具有消融辨识度。"
    elif confirmation["allocation_change_rate"] >= 0.01:
        narrative = "确认集分配改变率为 1%–5%，warping 机制存在但适用区域较窄。"
    else:
        narrative = "确认集分配改变率低于 1%，只应保留定向机制证人。"
    text = "\n".join([
        "# Warped-Bid Stress Tuning 判决",
        "",
        f"冻结 strata 数：{selected_count}",
        "",
        narrative,
        "",
        "本轮只调整场景参数，未修改 Johnson 算法规则；选区未使用 delta-J 正负。",
    ])
    (output / "stress_verdict.md").write_text(text + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "calibration", "confirmation", "all"), default="all")
    parser.add_argument("--workers", type=int, default=min(22, max(1, (os.cpu_count() or 1) - 2)))
    parser.add_argument("--smoke-count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/second_batch/warp_stress"))
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 22 or args.smoke_count < 1:
        parser.error("workers must be 1..22 and smoke-count must be positive")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = _existing_summaries(args.output)
    selected: tuple[SelectedStratum, ...] = ()
    verdict = "PASS"

    if args.stage in ("smoke", "all"):
        records = _run_cases(calibration_cases(limit=args.smoke_count), args.workers)
        summaries["smoke"] = _write_stage(args.output, "smoke", records)
        if summaries["smoke"]["johnson_gate_e_failures"]:
            verdict = "FAIL"

    if verdict == "PASS" and args.stage in ("calibration", "all"):
        records = _run_cases(calibration_cases(), args.workers)
        summaries["calibration"] = _write_stage(args.output, "calibration", records)
        if summaries["calibration"]["johnson_gate_e_failures"]:
            verdict = "FAIL"
        else:
            selected = select_strata(records)
            _save_selected(args.output, selected)

    if verdict == "PASS" and args.stage in ("confirmation", "all"):
        if not selected:
            selected = _load_selected(args.output)
        cases = confirmation_cases(selected)
        records = _run_cases(cases, args.workers)
        summaries["confirmation"] = _write_stage(args.output, "confirmation", records)
        summaries["confirmation"]["coverage"] = confirmation_coverage(selected, cases)
        (args.output / "confirmation" / "summary.json").write_text(
            json.dumps(summaries["confirmation"], indent=2), encoding="utf-8"
        )
        if summaries["confirmation"]["johnson_gate_e_failures"]:
            verdict = "FAIL"

    if not selected and (args.output / "tuned_ranges.json").exists():
        selected = _load_selected(args.output)
    if any(summary.get("johnson_gate_e_failures", 0) for summary in summaries.values()):
        verdict = "FAIL"
    manifest = {"verdict": verdict, "workers": args.workers, "selected_strata": len(selected), "stages": summaries}
    (args.output / "stress_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_verdict(args.output, summaries, len(selected))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
