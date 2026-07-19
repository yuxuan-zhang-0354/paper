"""Effect-blind D3 structural and runtime smoke; no utility contrasts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from time import process_time_ns

from uav_lifecycle.artifacts import write_json_atomic
from uav_lifecycle.dynamic_d3 import (
    ALLOCATION_PRESSURE_CONDITIONS,
    D3_SCALE_CELLS,
    MISMATCH_CONDITIONS,
    UTILITY_PROFILES,
    environment_model,
    generate_allocation_pressure,
    generate_cbba_isolation,
    generate_continuous,
    generate_mismatch,
    generate_scale,
    generate_weight,
    utility_config,
)
from uav_lifecycle.dynamic_policies import make_policy
from uav_lifecycle.dynamic_scenarios import D1_CELLS
from uav_lifecycle.dynamic_simulator import run_episode
from uav_lifecycle.dynamic_types import DynamicConfig, DynamicScenario, EnvironmentModel


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/dynamic_mainline/d3_design/structural_smoke.json"
METHODS = ("P", "B1m", "B4", "B5(4)", "B6", "SCBBA", "DVCBBA")
ISOLATION_METHODS = ("P", "SCBBA", "DVCBBA", "CEX")
PRESSURE_METHODS = ("P", "SCBBA", "DVCBBA", "B6")


@dataclass(frozen=True, slots=True)
class SmokeWork:
    suite: str
    condition: str
    scenario: DynamicScenario
    config: DynamicConfig
    environment: EnvironmentModel
    method: str


def _worker(work: SmokeWork) -> dict[str, object]:
    start = process_time_ns()
    try:
        policy = make_policy(work.method, work.config)
        result = run_episode(
            work.scenario, policy,
            config=work.config, environment_model=work.environment, method=work.method,
        )
        elapsed = process_time_ns() - start
        record = result.record
        audits = policy.allocator_audits
        return {
            "suite": work.suite, "condition": work.condition,
            "scenario_id": work.scenario.scenario_id, "method": work.method,
            "status": record.status, "gate_count": len(record.allocator_gates),
            "finite": all(map(lambda value: value == value and abs(value) != float("inf"), (
                record.normalized_utility, record.realized_utility, record.makespan,
            ))),
            "recon_count": record.recon_count, "attack_count": record.ammo_consumed,
            "bda_count": record.bda_count, "replan_count": record.replan_count,
            "allocator_calls": len(audits),
            "allocator_nonconverged": sum(
                audit["allocator_status"] != "converged" for audit in audits
            ),
            "rounds": sum(int(audit["rounds"]) for audit in audits),
            "message_packets": sum(int(audit["message_packets"]) for audit in audits),
            "positive_pair_count": sum(int(audit["positive_pair_count"]) for audit in audits),
            "process_time_ns": elapsed,
        }
    except Exception as error:
        return {
            "suite": work.suite, "condition": work.condition,
            "scenario_id": work.scenario.scenario_id, "method": work.method,
            "status": "failed", "gate_count": 0, "finite": False,
            "error_type": type(error).__name__, "error_message": str(error),
            "process_time_ns": process_time_ns() - start,
        }


def build_works() -> tuple[SmokeWork, ...]:
    nominal = DynamicConfig()
    env = EnvironmentModel.from_config(nominal)
    works: list[SmokeWork] = []
    for cell in D3_SCALE_CELLS:
        scenario = generate_scale(cell, 1900, nominal, smoke=True)
        works.extend(SmokeWork("scale", cell.cell_id, scenario, nominal, env, method) for method in METHODS)
    for cell in D1_CELLS:
        scenario = generate_continuous(cell, 2900, nominal, smoke=True)
        works.append(SmokeWork("continuous_belief", cell.cell_id, scenario, nominal, env, "P"))
    for condition in MISMATCH_CONDITIONS:
        scenario = generate_mismatch(D1_CELLS[-1], condition, 3900, nominal, smoke=True)
        works.append(SmokeWork(
            "model_mismatch", condition, scenario, nominal,
            environment_model(condition, nominal), "P",
        ))
    for profile in UTILITY_PROFILES:
        config = utility_config(profile)
        scenario = generate_weight(D1_CELLS[-1], profile, 4900, config, smoke=True)
        works.append(SmokeWork(
            "utility_profile", profile, scenario, config,
            EnvironmentModel.from_config(config), "P",
        ))
    isolation = generate_cbba_isolation(D1_CELLS[-1], 5900, nominal, smoke=True)
    works.extend(SmokeWork(
        "cbba_isolation", isolation.cell_id, isolation, nominal, env, method,
    ) for method in ISOLATION_METHODS)
    for condition in ALLOCATION_PRESSURE_CONDITIONS:
        scenario = generate_allocation_pressure(condition, 6900, nominal, smoke=True)
        works.extend(SmokeWork(
            "allocation_pressure", condition, scenario, nominal, env, method,
        ) for method in PRESSURE_METHODS)
    return tuple(works)


def run(workers: int = 22) -> dict[str, object]:
    works = build_works()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_worker, works))
    failures = [row for row in rows if row["status"] != "complete"]
    gates = sum(int(row["gate_count"]) for row in rows)
    nonfinite = sum(not bool(row["finite"]) for row in rows)
    modes = {
        "recon": sum(float(row.get("recon_count", 0)) > 0 for row in rows),
        "attack": sum(float(row.get("attack_count", 0)) > 0 for row in rows),
        "bda": sum(float(row.get("bda_count", 0)) > 0 for row in rows),
    }
    summary = {
        "status": "PASS" if not failures and gates == nonfinite == 0 and all(modes.values()) else "FAILED",
        "effect_statistics_computed": False,
        "work_count": len(rows), "failure_count": len(failures),
        "gate_count": gates, "nonfinite_count": nonfinite,
        "action_region_coverage": modes,
        "max_process_time_seconds": max(int(row["process_time_ns"]) for row in rows) / 1e9,
        "rows": rows,
    }
    write_json_atomic(OUTPUT, summary)
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
