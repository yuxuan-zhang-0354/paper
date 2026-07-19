from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from random import Random
from statistics import mean, median

from uav_lifecycle.cbba_static import Method
from uav_lifecycle.second_batch import evaluate_instance, exhaustive_instances, random_pilot_instances, tier0_instances


def _evaluate(instance):
    return evaluate_instance(instance)


def _bootstrap_ci(values: list[float], seed: int = 7132026, repetitions: int = 1000):
    if not values:
        return [0.0, 0.0]
    rng = Random(seed)
    samples = sorted(mean(rng.choice(values) for _ in values) for _ in range(repetitions))
    return [samples[int(0.025 * repetitions)], samples[min(repetitions - 1, int(0.975 * repetitions))]]


def _stage_summary(records: list[dict]) -> dict:
    johnson = [r for r in records if r["method"] == Method.JOHNSON_WARPED.value]
    surrogate = {r["instance_id"]: r for r in records if r["method"] == Method.SURROGATE.value}
    paired = [r["true_score"] - surrogate[r["instance_id"]]["true_score"] for r in johnson]
    return {
        "instance_count": len(johnson),
        "record_count": len(records),
        "johnson_gate_e_failures": sum(r["gate_e_failures"] for r in johnson),
        "johnson_status_counts": {status: sum(r["status"] == status for r in johnson) for status in ("converged", "cycle", "timeout")},
        "johnson_ratio_mean": mean(r["ratio"] for r in johnson) if johnson else 0.0,
        "johnson_rounds_median": median(r["rounds"] for r in johnson) if johnson else 0.0,
        "johnson_minus_surrogate_mean": mean(paired) if paired else 0.0,
        "johnson_minus_surrogate_bootstrap_95ci": _bootstrap_ci(paired),
    }


def _write_stage(output: Path, name: str, records: list[dict]) -> dict:
    stage_dir = output / name
    stage_dir.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) if records else []
    with (stage_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["assigned"] = json.dumps(row["assigned"], separators=(",", ":"))
            row["paths"] = json.dumps(row["paths"], separators=(",", ":"))
            writer.writerow(row)
    summary = _stage_summary(records)
    (stage_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def collect_stage_summaries(output: Path) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for name in ("tier0", "smoke", "exhaustive", "random"):
        path = output / name / "summary.json"
        if path.exists():
            summaries[name] = json.loads(path.read_text(encoding="utf-8"))
    return summaries


def _run_instances(instances, workers: int) -> list[dict]:
    records: list[dict] = []
    if workers == 1:
        groups = map(_evaluate, instances)
        for group in groups:
            records.extend(group)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for group in pool.map(_evaluate, instances, chunksize=1):
                records.extend(group)
    records.sort(key=lambda r: (r["instance_id"], r["method"]))
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("tier0", "smoke", "exhaustive", "random", "all"), default="smoke")
    parser.add_argument("--workers", type=int, default=min(22, max(1, (os.cpu_count() or 1) - 2)))
    parser.add_argument("--smoke-count", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/second_batch"))
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 22 or args.smoke_count < 1:
        parser.error("workers must be 1..22 and smoke-count must be positive")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = collect_stage_summaries(args.output)

    tier0 = _run_instances(tier0_instances(), args.workers)
    summaries["tier0"] = _write_stage(args.output, "tier0", tier0)
    if summaries["tier0"]["johnson_gate_e_failures"]:
        verdict = "FAIL"
    else:
        verdict = "PASS"
        requested = [args.stage] if args.stage != "all" else ["smoke", "exhaustive", "random"]
        for stage in requested:
            if stage == "tier0":
                continue
            if stage == "smoke":
                instances = list(islice(exhaustive_instances(), args.smoke_count))
            elif stage == "exhaustive":
                instances = exhaustive_instances()
            else:
                instances = random_pilot_instances()
            records = _run_instances(instances, args.workers)
            summaries[stage] = _write_stage(args.output, stage, records)
            if summaries[stage]["johnson_gate_e_failures"]:
                verdict = "FAIL"
                break
    if any(summary.get("johnson_gate_e_failures", 0) for summary in summaries.values()):
        verdict = "FAIL"
    manifest = {"verdict": verdict, "stages": summaries, "workers": args.workers}
    (args.output / "second_batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
