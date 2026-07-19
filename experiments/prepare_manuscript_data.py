"""Prepare paper-ready tables from frozen D2-D5 artifacts without rerunning policies."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "dynamic_mainline"
OUT = ROOT / "results" / "manuscript_data"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    path = OUT / name
    if not rows:
        raise ValueError(f"no rows for {name}")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"inconsistent columns for {name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else math.nan


def i(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if value not in (None, "") else 0


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
    }


def bootstrap_ci(values: list[float], seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(10_000)
    for start in range(0, 10_000, 500):
        size = min(500, 10_000 - start)
        indices = rng.integers(0, len(array), size=(size, len(array)))
        means[start : start + size] = array[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def ratio_summary(
    strata: dict[str, list[tuple[float, float]]], token: str,
) -> tuple[float, float, float, float, float, float]:
    """Ratio of equally weighted stratum means with paired bootstrap CI."""
    boot_p = np.zeros(10_000)
    boot_b = np.zeros(10_000)
    p_means, b_means = [], []
    for name, pairs in sorted(strata.items()):
        values = np.asarray(pairs, dtype=float)
        p_means.append(float(values[:, 0].mean()))
        b_means.append(float(values[:, 1].mean()))
        seed = int.from_bytes(hashlib.sha256(f"{token}|{name}".encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        for start in range(0, 10_000, 500):
            size = min(500, 10_000 - start)
            indices = rng.integers(0, len(values), size=(size, len(values)))
            boot_p[start : start + size] += values[indices, 0].mean(axis=1) / len(strata)
            boot_b[start : start + size] += values[indices, 1].mean(axis=1) / len(strata)
    p_mean, b_mean = float(np.mean(p_means)), float(np.mean(b_means))
    if b_mean <= 0 or np.any(boot_b <= 0):
        raise ValueError(f"non-positive baseline mean in {token}")
    ratios = 100.0 * (boot_p - boot_b) / boot_b
    low, high = np.quantile(ratios, (0.025, 0.975))
    return p_mean, b_mean, p_mean - b_mean, 100.0 * (p_mean - b_mean) / b_mean, float(low), float(high)


def parse_nm(text: str) -> tuple[int, int]:
    parts = text.split("-")
    n = next(int(part[1:]) for part in parts if part.startswith("N") and part[1:].isdigit())
    m = next(int(part[1:]) for part in parts if part.startswith("M") and part[1:].isdigit())
    return n, m


def scenario_suites(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        scenario_id: metadata["suite"]
        for scenario_id, metadata in manifest["scenario_metadata"].items()
    }


def prepare_core() -> None:
    analysis_path = RESULTS / "d2_confirmation" / "d2_confirmation_analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    records_path = RESULTS / "d2_confirmation" / "canonical" / "records.csv"
    records = read_csv(records_path)
    by_scenario: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in records:
        by_scenario[row["scenario_id"]][row["method"]] = row

    forest: list[dict[str, object]] = []
    contrast_order = ["P-B1m", "P-B2", "P-B3", "P-B4", "P-B5(2)", "P-B5(4)", "P-B5(8)", "P-B6", "P-CEX"]
    for contrast_index, contrast in enumerate(contrast_order):
        item = analysis["contrasts"][contrast]
        forest.append({
            "panel": "pooled_contrast",
            "label": contrast,
            "contrast": contrast,
            "scenario_count": 4096,
            "mean": item["equal_cell_mean"],
            "ci_low": item["bootstrap_ci_95"][0],
            "ci_high": item["bootstrap_ci_95"][1],
            "win": item["win_tie_loss"]["win"],
            "tie": item["win_tie_loss"]["tie"],
            "loss": item["win_tie_loss"]["loss"],
            "inference_label": "confirmatory_primary" if contrast == "P-B1m" else "registered_secondary_or_sensitivity",
        })
        if contrast == "P-B1m":
            for cell_index, (cell, mean) in enumerate(item["cell_means"].items()):
                values = [
                    float(methods["P"]["normalized_utility"]) - float(methods["B1m"]["normalized_utility"])
                    for methods in by_scenario.values()
                    if methods["P"]["cell_id"] == cell
                ]
                low, high = bootstrap_ci(values, 2026071500 + cell_index)
                wins = sum(value > 1e-12 for value in values)
                ties = sum(abs(value) <= 1e-12 for value in values)
                forest.append({
                    "panel": "primary_by_cell",
                    "label": cell,
                    "contrast": contrast,
                    "scenario_count": len(values),
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "win": wins,
                    "tie": ties,
                    "loss": len(values) - wins - ties,
                    "inference_label": "descriptive_cell_bootstrap",
                })
    write_csv("fig5_confirmatory_forest.csv", forest)

    table5 = [row.copy() for row in forest if row["panel"] == "pooled_contrast"]
    write_csv("table5_main_lifecycle_results.csv", table5)

    bda_distribution: list[dict[str, object]] = []
    bda_bins: dict[str, list[dict[str, float]]] = defaultdict(list)
    for scenario_id, methods in by_scenario.items():
        p = methods["P"]
        baseline = methods["B4"]
        delta_utility = f(p, "normalized_utility") - f(baseline, "normalized_utility")
        bda_count = i(p, "bda_count")
        group = "0" if bda_count == 0 else "1" if bda_count == 1 else "2+"
        row = {
            "scenario_id": scenario_id,
            "cell_id": p["cell_id"],
            "p_bda_count": bda_count,
            "bda_group": group,
            "delta_normalized_utility": delta_utility,
            "delta_destroyed_value": f(p, "destroyed_value") - f(baseline, "destroyed_value"),
            "delta_service_cost": f(p, "service_cost") - f(baseline, "service_cost"),
            "delta_distance_cost": f(p, "distance_cost") - f(baseline, "distance_cost"),
            "delta_ammo_cost": f(p, "ammo_cost") - f(baseline, "ammo_cost"),
            "delta_realized_utility": f(p, "realized_utility") - f(baseline, "realized_utility"),
            "delta_invalid_attack": i(p, "invalid_attack_count") - i(baseline, "invalid_attack_count"),
            "delta_initial_wreck_attack": i(p, "initial_wreck_attack_count") - i(baseline, "initial_wreck_attack_count"),
        }
        row["utility_identity_residual"] = (
            float(row["delta_destroyed_value"])
            - float(row["delta_service_cost"])
            - float(row["delta_distance_cost"])
            - float(row["delta_ammo_cost"])
            - float(row["delta_realized_utility"])
        )
        bda_distribution.append(row)
        bda_bins[group].append({key: float(value) for key, value in row.items() if key.startswith("delta_")})
    bda_distribution.sort(key=lambda row: float(row["delta_normalized_utility"]))
    for rank, row in enumerate(bda_distribution, start=1):
        row["ecdf"] = rank / len(bda_distribution)
    write_csv("fig6_bda_ecdf.csv", bda_distribution)

    bda_summary: list[dict[str, object]] = []
    for group in ("all", "0", "1", "2+"):
        rows = bda_distribution if group == "all" else [row for row in bda_distribution if row["bda_group"] == group]
        values = [float(row["delta_normalized_utility"]) for row in rows]
        stats = quantiles(values)
        low, high = bootstrap_ci(values, 2026071600 + len(bda_summary))
        bda_summary.append({
            "bda_group": group,
            "scenario_count": len(rows),
            **stats,
            "ci_low": low,
            "ci_high": high,
            "win": sum(value > 1e-12 for value in values),
            "tie": sum(abs(value) <= 1e-12 for value in values),
            "loss": sum(value < -1e-12 for value in values),
            "mean_delta_destroyed_value": float(np.mean([float(row["delta_destroyed_value"]) for row in rows])),
            "mean_delta_service_cost": float(np.mean([float(row["delta_service_cost"]) for row in rows])),
            "mean_delta_distance_cost": float(np.mean([float(row["delta_distance_cost"]) for row in rows])),
            "mean_delta_ammo_cost": float(np.mean([float(row["delta_ammo_cost"]) for row in rows])),
            "mean_delta_invalid_attack": float(np.mean([float(row["delta_invalid_attack"]) for row in rows])),
            "max_abs_utility_identity_residual": max(abs(float(row["utility_identity_residual"])) for row in rows),
            "analysis_label": "posthoc_explanatory",
        })
    write_csv("fig6_bda_summary.csv", bda_summary)


def prepare_allocator_and_sensitivity() -> None:
    d3 = RESULTS / "d3_analysis"
    d4 = RESULTS / "d4_analysis"
    suite = read_csv(d3 / "suite_contrasts.csv")
    cycles = read_csv(d3 / "cycle_stratified_contrasts.csv")
    workload = read_csv(d3 / "normalized_allocator_workload.csv")
    exact = read_csv(d3 / "exact_gap_summary.csv")

    stability: list[dict[str, object]] = []
    for row in suite:
        if row["contrast"] == "P-DVCBBA":
            stability.append({
                "record_type": "overall_contrast",
                "suite": row["suite"],
                "condition": "all",
                "status_group": "all",
                "scenario_count": i(row, "scenario_count"),
                "mean": f(row, "equal_stratum_mean"),
                "median": f(row, "paired_median"),
                "ci_low": f(row, "bootstrap_ci_low"),
                "ci_high": f(row, "bootstrap_ci_high"),
                "win": i(row, "win"), "tie": i(row, "tie"), "loss": i(row, "loss"),
                "analysis_label": "registered_external_validation",
            })
    for row in cycles:
        if row["suite"] in {"scale", "allocation_pressure"}:
            stability.append({
                "record_type": "status_stratum",
                "suite": row["suite"],
                "condition": row["condition"],
                "status_group": row["dvcbba_status_group"],
                "scenario_count": i(row, "scenario_count"),
                "mean": f(row, "mean_p_minus_dvcbba"),
                "median": f(row, "median_p_minus_dvcbba"),
                "ci_low": f(row, "bootstrap_ci_low"),
                "ci_high": f(row, "bootstrap_ci_high"),
                "win": i(row, "win"), "tie": i(row, "tie"), "loss": i(row, "loss"),
                "analysis_label": "posthoc_descriptive_status_stratum",
            })
    write_csv("fig7_allocator_stability.csv", stability)

    workload_rows = [row for row in workload if row["suite"] in {"scale", "allocation_pressure"} and row["method"] in {"P", "DVCBBA", "SCBBA", "B6"}]
    write_csv("fig7_allocator_workload.csv", [{key: value for key, value in row.items()} for row in workload_rows])
    write_csv("table6_exact_reference.csv", [{key: value for key, value in row.items()} for row in exact])

    runtime = read_csv(RESULTS / "d3_external_validation" / "canonical" / "d3_runtime.csv")
    records = read_csv(RESULTS / "d3_external_validation" / "canonical" / "d3_records.csv")
    d3_suites = scenario_suites(RESULTS / "d3_design" / "d3_manifest.json")
    utility: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        if d3_suites[row["scenario_id"]] == "scale":
            utility[(row["cell_id"], row["method"])].append(f(row, "normalized_utility"))
    calls: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in runtime:
        if row["suite"] == "scale":
            calls[(row["cell_id"], row["method"])].append(row)
    scale_rows: list[dict[str, object]] = []
    for key in sorted(utility):
        cell, method = key
        if method not in {"P", "DVCBBA", "SCBBA", "B6", "B1m", "B4", "B5(4)"}:
            continue
        entries = calls[key]
        converged = [row for row in entries if row["allocator_status"] == "converged"]
        nonconverged = [row for row in entries if row["allocator_status"] != "converged"]
        n_agents, n_targets = parse_nm(cell)
        scale_rows.append({
            "cell_id": cell, "agents": n_agents, "targets": n_targets, "method": method,
            "episodes": len(utility[key]), "mean_normalized_utility": float(np.mean(utility[key])),
            "allocator_calls": len(entries), "converged_calls": len(converged),
            "nonconverged_calls": len(nonconverged),
            "convergence_rate": len(converged) / len(entries) if entries else math.nan,
            "rounds_per_converged_epoch": float(np.mean([i(row, "rounds") for row in converged])) if converged else math.nan,
            "message_packets_per_converged_epoch": float(np.mean([i(row, "message_packets") for row in converged])) if converged else math.nan,
            "planning_ms_p50": float(np.quantile([f(row, "planning_process_time_ns") / 1e6 for row in entries], 0.5)) if entries else math.nan,
            "planning_ms_p95": float(np.quantile([f(row, "planning_process_time_ns") / 1e6 for row in entries], 0.95)) if entries else math.nan,
            "nonconverged_wasted_rounds": sum(i(row, "rounds") for row in nonconverged),
            "nonconverged_wasted_message_packets": sum(i(row, "message_packets") for row in nonconverged),
        })
    write_csv("fig8_scalability_tradeoff.csv", scale_rows)

    battlefield = read_csv(d4 / "condition_contrasts.csv")
    write_csv("fig9_battlefield_heatmap.csv", [
        {key: value for key, value in row.items()}
        for row in battlefield
        if row["suite"] == "battlefield_structure" and row["scope"] == "condition"
    ])
    write_csv("fig9_reachability_heatmap.csv", [
        {key: value for key, value in row.items()}
        for row in battlefield
        if row["suite"] == "reachability" and row["scope"] == "condition"
    ])
    write_csv("fig9_reachability_regimes.csv", [
        {key: value for key, value in row.items()}
        for row in read_csv(d4 / "reachability_regimes.csv")
    ])
    robustness: list[dict[str, object]] = []
    for row in read_csv(d4 / "suite_contrasts.csv"):
        robustness.append({
            "suite": row["suite"], "contrast": row["contrast"],
            "scenario_count": i(row, "scenario_count"), "stratum_or_seed_blocks": i(row, "seed_block_count"),
            "mean": f(row, "seed_block_mean"), "median": f(row, "episode_median"),
            "ci_low": f(row, "bootstrap_ci_low"), "ci_high": f(row, "bootstrap_ci_high"),
            "win": i(row, "win"), "tie": i(row, "tie"), "loss": i(row, "loss"),
            "bootstrap_unit": row["bootstrap_unit"],
        })
    for row in suite:
        if row["contrast"] not in {"P-B1m", "P-B4", "P-B6", "P-DVCBBA", "P-SCBBA"}:
            continue
        robustness.append({
            "suite": row["suite"], "contrast": row["contrast"],
            "scenario_count": i(row, "scenario_count"), "stratum_or_seed_blocks": i(row, "stratum_count"),
            "mean": f(row, "equal_stratum_mean"), "median": f(row, "paired_median"),
            "ci_low": f(row, "bootstrap_ci_low"), "ci_high": f(row, "bootstrap_ci_high"),
            "win": i(row, "win"), "tie": i(row, "tie"), "loss": i(row, "loss"),
            "bootstrap_unit": "paired scenario within registered stratum",
        })
    write_csv("table6_robustness_summary.csv", robustness)


def prepare_relative_gains() -> None:
    rows: list[dict[str, object]] = []

    core_records = read_csv(RESULTS / "d2_confirmation" / "canonical" / "records.csv")
    core_nested: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in core_records:
        core_nested[row["scenario_id"]][row["method"]] = row
    for baseline in ("B1m", "B2", "B3", "B4", "B5(2)", "B5(4)", "B5(8)", "B6", "CEX"):
        strata: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for methods in core_nested.values():
            strata[methods["P"]["cell_id"]].append((f(methods["P"], "normalized_utility"), f(methods[baseline], "normalized_utility")))
        p_mean, b_mean, difference, gain, low, high = ratio_summary(strata, f"relative|core|P-{baseline}")
        rows.append({
            "suite": "core_comparison", "contrast": f"P-{baseline}", "comparison_role": "core",
            "scenario_count": sum(map(len, strata.values())), "stratum_or_block_count": len(strata),
            "p_mean": p_mean, "baseline_mean": b_mean, "absolute_difference": difference,
            "relative_improvement_percent": gain, "relative_ci_low": low, "relative_ci_high": high,
            "bootstrap_unit": "paired scenario within cell; equal-cell aggregation",
        })

    d3_manifest = json.loads((RESULTS / "d3_design" / "d3_manifest.json").read_text(encoding="utf-8"))
    d3_meta = d3_manifest["scenario_metadata"]
    d3_nested: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_csv(RESULTS / "d3_external_validation" / "canonical" / "d3_records.csv"):
        d3_nested[row["scenario_id"]][row["method"]] = row
    d3_baselines = {
        "scale": ("SCBBA", "B1m", "DVCBBA", "B6"),
        "allocation_pressure": ("SCBBA", "DVCBBA", "B6"),
        "continuous_belief": ("B1m",), "model_mismatch": ("B1m",), "utility_profile": ("B1m",),
    }
    for suite, baselines in d3_baselines.items():
        scenario_ids = [item for item, meta in d3_meta.items() if meta["suite"] == suite]
        for baseline in baselines:
            strata: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for scenario_id in scenario_ids:
                meta, methods = d3_meta[scenario_id], d3_nested[scenario_id]
                stratum = (
                    f"{meta['condition']}|{meta['cell_id']}" if suite in {"model_mismatch", "utility_profile"}
                    else meta["condition"] if suite == "allocation_pressure" else meta["cell_id"]
                )
                strata[stratum].append((f(methods["P"], "normalized_utility"), f(methods[baseline], "normalized_utility")))
            p_mean, b_mean, difference, gain, low, high = ratio_summary(strata, f"relative|{suite}|P-{baseline}")
            rows.append({
                "suite": suite, "contrast": f"P-{baseline}", "comparison_role": "external_validation",
                "scenario_count": len(scenario_ids), "stratum_or_block_count": len(strata),
                "p_mean": p_mean, "baseline_mean": b_mean, "absolute_difference": difference,
                "relative_improvement_percent": gain, "relative_ci_low": low, "relative_ci_high": high,
                "bootstrap_unit": "paired scenario within registered stratum; equal-stratum aggregation",
            })

    d4_manifest = json.loads((RESULTS / "d4_design" / "d4_manifest.json").read_text(encoding="utf-8"))
    d4_meta = d4_manifest["scenario_metadata"]
    d4_nested: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_csv(RESULTS / "d4_sensitivity" / "canonical" / "d4_records.csv"):
        d4_nested[row["scenario_id"]][row["method"]] = row
    d4_baselines = {
        "battlefield_structure": ("SCBBA", "DVCBBA", "B6", "B1m", "B4"),
        "reachability": ("SCBBA", "DVCBBA", "B6"),
    }
    for suite, baselines in d4_baselines.items():
        scenario_ids = [item for item, meta in d4_meta.items() if meta["suite"] == suite]
        for baseline in baselines:
            blocks: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for scenario_id in scenario_ids:
                methods = d4_nested[scenario_id]
                blocks[str(d4_meta[scenario_id]["index"])].append((f(methods["P"], "normalized_utility"), f(methods[baseline], "normalized_utility")))
            block_pairs = [(float(np.mean([p for p, _ in values])), float(np.mean([b for _, b in values]))) for values in blocks.values()]
            strata = {"shared_seed_block": block_pairs}
            p_mean, b_mean, difference, gain, low, high = ratio_summary(strata, f"relative|{suite}|P-{baseline}")
            rows.append({
                "suite": suite, "contrast": f"P-{baseline}", "comparison_role": "battlefield_sensitivity",
                "scenario_count": len(scenario_ids), "stratum_or_block_count": len(block_pairs),
                "p_mean": p_mean, "baseline_mean": b_mean, "absolute_difference": difference,
                "relative_improvement_percent": gain, "relative_ci_low": low, "relative_ci_high": high,
                "bootstrap_unit": "shared seed block",
            })
    write_csv("relative_gain_summary.csv", rows)


def communication_rows(
    suite_name: str,
    records_path: Path,
    events_path: Path,
    runtime_path: Path | None,
    planning_trace_path: Path | None,
    suite_manifest_path: Path | None = None,
) -> list[dict[str, object]]:
    records = read_csv(records_path)
    events = read_csv(events_path)
    event_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in events:
        event_groups[(row["scenario_id"], row["method"])].append(row)
    runtime_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    if runtime_path is not None:
        for row in read_csv(runtime_path):
            runtime_groups[(row["scenario_id"], row["method"])].append(row)
    elif planning_trace_path is not None:
        for row in read_csv(planning_trace_path):
            runtime_groups[(row["scenario_id"], row["method"])].append(row)
    suite_map = scenario_suites(suite_manifest_path) if suite_manifest_path else {}

    methods_with_screening = {"P", "B1m", "B2", "B3", "B4", "B5(2)", "B5(4)", "B5(8)", "B6", "DVCBBA", "SCBBA"}
    mode_count = {"B2": 1, "B3": 1, "B4": 2}
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for record in records:
        scenario_id, method = record["scenario_id"], record["method"]
        n_agents, n_targets = parse_nm(record["cell_id"])
        key = (suite_map.get(scenario_id, suite_name), method)
        acc = grouped[key]
        acc["episodes"] += 1
        scenario_events = event_groups[(scenario_id, method)]
        observation_packets = len(scenario_events) * max(0, n_agents - 1)
        observation_bytes = observation_packets * 28
        calls = runtime_groups[(scenario_id, method)]
        if planning_trace_path is not None and not calls:
            # The accepted D2 diagnostic trace covers only its registered methods.
            # Omit uncovered methods instead of encoding missing communication as zero.
            continue
        planning_calls = len(calls)
        if method in methods_with_screening:
            modes = mode_count.get(method, 3)
            screening_packets = planning_calls * n_agents * max(0, n_agents - 1)
            screening_entries = screening_packets * n_targets * modes
            screening_bytes = screening_packets * 12 + screening_entries * 8
        else:
            screening_packets = screening_entries = screening_bytes = 0
        if runtime_path is not None:
            cbba_packets = sum(i(row, "message_packets") for row in calls)
            cbba_entries = sum(i(row, "message_scalars") for row in calls)
        else:
            # D2 planning traces expose rounds but not per-call idle counts; use the
            # same full-fleet dense accounting convention as screening.
            cbba_packets = sum(i(row, "rounds") for row in calls) * n_agents * max(0, n_agents - 1)
            cbba_entries = cbba_packets * n_targets
            if method in {"B6", "CEX"}:
                cbba_packets = cbba_entries = 0
        cbba_bytes = cbba_packets * 12 + cbba_entries * 12
        acc["observation_ack_packets"] += observation_packets
        acc["observation_ack_bytes"] += observation_bytes
        acc["screening_packets_upper"] += screening_packets
        acc["screening_score_entries_upper"] += screening_entries
        acc["screening_bytes_upper"] += screening_bytes
        acc["cbba_packets"] += cbba_packets
        acc["cbba_target_entries"] += cbba_entries
        acc["cbba_bytes"] += cbba_bytes
    result: list[dict[str, object]] = []
    for (suite, method), acc in sorted(grouped.items()):
        episodes = int(acc["episodes"])
        total_packets = acc["observation_ack_packets"] + acc["screening_packets_upper"] + acc["cbba_packets"]
        total_bytes = acc["observation_ack_bytes"] + acc["screening_bytes_upper"] + acc["cbba_bytes"]
        result.append({
            "experiment_group": suite_name,
            "suite": suite,
            "method": method,
            "episodes": episodes,
            "observation_ack_packets_per_episode": acc["observation_ack_packets"] / episodes,
            "screening_packets_upper_per_episode": acc["screening_packets_upper"] / episodes,
            "cbba_packets_per_episode": acc["cbba_packets"] / episodes,
            "total_packets_upper_per_episode": total_packets / episodes,
            "observation_ack_bytes_per_episode": acc["observation_ack_bytes"] / episodes,
            "screening_bytes_upper_per_episode": acc["screening_bytes_upper"] / episodes,
            "cbba_bytes_per_episode": acc["cbba_bytes"] / episodes,
            "total_bytes_upper_per_episode": total_bytes / episodes,
            "accounting_convention": "recipient-specific packets; dense full-fleet screening upper bound; 28-byte public event; 12-byte packet header; float64 score; int32 winner plus float64 bid",
        })
    return result


def prepare_communication() -> None:
    rows: list[dict[str, object]] = []
    rows.extend(communication_rows(
        "core_comparison",
        RESULTS / "d2_confirmation" / "canonical" / "records.csv",
        RESULTS / "d2_confirmation" / "canonical" / "public_events.csv",
        None,
        RESULTS / "d2_diagnostics" / "diagnostic_planning_trace.csv",
        None,
    ))
    rows.extend(communication_rows(
        "external_validation",
        RESULTS / "d3_external_validation" / "canonical" / "d3_records.csv",
        RESULTS / "d3_external_validation" / "canonical" / "d3_public_events.csv",
        RESULTS / "d3_external_validation" / "canonical" / "d3_runtime.csv",
        None,
        RESULTS / "d3_design" / "d3_manifest.json",
    ))
    rows.extend(communication_rows(
        "battlefield_sensitivity",
        RESULTS / "d4_sensitivity" / "canonical" / "d4_records.csv",
        RESULTS / "d4_sensitivity" / "canonical" / "d4_public_events.csv",
        RESULTS / "d4_sensitivity" / "canonical" / "d4_runtime.csv",
        None,
        RESULTS / "d4_design" / "d4_manifest.json",
    ))
    write_csv("communication_accounting.csv", rows)


def prepare_d5_factorial() -> None:
    source = RESULTS / "d5_factorial_ablation" / "analysis"
    for source_name, output_name in (
        ("method_summary.csv", "table7_d5_method_summary.csv"),
        ("paired_contrasts.csv", "table7_d5_paired_contrasts.csv"),
        ("factorial_effects.csv", "table7_d5_factorial_effects.csv"),
    ):
        write_csv(output_name, read_csv(source / source_name))


def write_readme_and_inventory() -> None:
    readme = """# Manuscript-ready data bundle

This directory is generated from frozen experiment artifacts by
`experiments/prepare_manuscript_data.py`. It does not rerun or modify any policy.

- `fig5_*`: confirmatory forest-plot data.
- `fig6_*`: optional-BDA ECDF, stratification, and utility decomposition.
- `fig7_*`: allocator stability, cycle strata, and workload.
- `fig8_*`: scale, utility, runtime, rounds, and message trade-offs.
- `fig9_*`: battlefield-structure and reachability heatmaps.
- `table5_*`: main lifecycle and information-action results.
- `table6_*`: exact-reference and robustness summaries.
- `table7_d5_*`: registered raw/warped x retain/rebuild dynamic factorial results.
- `relative_gain_summary.csv`: ratio-of-means relative improvements with paired
  or shared-seed-block bootstrap confidence intervals.
- `communication_accounting.csv`: separate public-event, screening, and CBBA
  communication. Screening is a reproducible dense full-fleet upper bound, not
  a measured network trace; CBBA packet counts come from runtime diagnostics
  where available.

The statistical unit is the independent scenario. Method records are paired
repeated evaluations of the same scenario and are not additional samples.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    inventory = {}
    for path in sorted(OUT.iterdir()):
        if path.name == "inventory.json" or not path.is_file():
            continue
        inventory[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": sum(1 for _ in path.open(encoding="utf-8")) - 1 if path.suffix == ".csv" else None,
        }
    (OUT / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")


def validate_outputs() -> None:
    forest = read_csv(OUT / "fig5_confirmatory_forest.csv")
    primary = next(row for row in forest if row["panel"] == "pooled_contrast" and row["contrast"] == "P-B1m")
    bda = read_csv(OUT / "fig6_bda_ecdf.csv")
    relative = read_csv(OUT / "relative_gain_summary.csv")
    primary_relative = next(row for row in relative if row["suite"] == "core_comparison" and row["contrast"] == "P-B1m")
    checks = {
        "primary_mean_matches_frozen": abs(f(primary, "mean") - 0.07586910244785196) < 1e-15,
        "primary_ci_matches_frozen": abs(f(primary, "ci_low") - 0.07151290733130193) < 1e-15 and abs(f(primary, "ci_high") - 0.08031836304795326) < 1e-15,
        "bda_scenario_count": len(bda) == 4096,
        "bda_utility_identity": max(abs(f(row, "utility_identity_residual")) for row in bda) < 1e-10,
        "battlefield_grid_complete": len(read_csv(OUT / "fig9_battlefield_heatmap.csv")) == 80,
        "reachability_grid_complete": len(read_csv(OUT / "fig9_reachability_heatmap.csv")) == 27,
        "reachability_regimes_complete": len(read_csv(OUT / "fig9_reachability_regimes.csv")) == 9,
        "scale_method_grid_complete": len(read_csv(OUT / "fig8_scalability_tradeoff.csv")) == 49,
        "primary_relative_gain_matches_means": abs(f(primary_relative, "relative_improvement_percent") - 84.56649365289704) < 1e-10,
        "relative_gain_intervals_finite": all(
            math.isfinite(f(row, key))
            for row in relative for key in ("p_mean", "baseline_mean", "relative_improvement_percent", "relative_ci_low", "relative_ci_high")
        ),
        "relative_gain_baselines_positive": all(f(row, "baseline_mean") > 0 for row in relative),
        "communication_rows_nonnegative": all(
            all(f(row, key) >= 0 for key in (
                "observation_ack_packets_per_episode", "screening_packets_upper_per_episode",
                "cbba_packets_per_episode", "total_packets_upper_per_episode",
            ))
            for row in read_csv(OUT / "communication_accounting.csv")
        ),
        "d5_method_grid_complete": len(read_csv(OUT / "table7_d5_method_summary.csv")) == 8,
        "d5_factorial_grid_complete": len(read_csv(OUT / "table7_d5_factorial_effects.csv")) == 45,
    }
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "PASS":
        raise RuntimeError(json.dumps(report))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prepare_core()
    prepare_allocator_and_sensitivity()
    prepare_relative_gains()
    prepare_communication()
    prepare_d5_factorial()
    validate_outputs()
    write_readme_and_inventory()
    print(json.dumps({"status": "complete", "output": str(OUT), "files": len(list(OUT.iterdir()))}))


if __name__ == "__main__":
    main()
