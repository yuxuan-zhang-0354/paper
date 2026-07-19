import json
import subprocess
import sys

from experiments.run_second_batch import collect_stage_summaries
from uav_lifecycle.cbba_static import Method
from uav_lifecycle.second_batch import (
    evaluate_instance,
    exhaustive_instances,
    random_instance,
    random_pilot_instances,
    tier0_instances,
)
from uav_lifecycle.exact_allocation import solve_exact


def test_tier0_is_deterministic_and_johnson_passes_gate_e():
    first = tier0_instances()
    second = tier0_instances()
    assert first == second
    assert {instance.instance_id for instance in first} >= {
        "deadline_dmg",
        "warp_order",
        "warp_activation",
        "warp_competition",
        "two_agent_tie",
    }
    for instance in first:
        records = evaluate_instance(instance)
        johnson = next(record for record in records if record["method"] == Method.JOHNSON_WARPED.value)
        assert johnson["status"] == "converged"
        assert johnson["gate_e_failures"] == 0
    activation = evaluate_instance(next(i for i in first if i.instance_id == "warp_activation"))
    johnson = next(record for record in activation if record["method"] == Method.JOHNSON_WARPED.value)
    assert johnson["warped_bid_count"] > 0
    competition = evaluate_instance(next(i for i in first if i.instance_id == "warp_competition"))
    by_method = {record["method"]: record for record in competition}
    assert by_method[Method.JOHNSON_WARPED.value]["assigned"] == [0, 1, 2]
    assert by_method[Method.JOHNSON_WARPED.value]["paths"] != by_method[Method.FULL_REBUILD_RAW.value]["paths"]


def test_exhaustive_generator_has_exact_count_and_unique_ids():
    instances = list(exhaustive_instances())
    assert len(instances) == 24_576
    assert len({instance.instance_id for instance in instances}) == 24_576


def test_random_instance_replays_from_cell_and_seed():
    assert random_instance((3, 5), 713) == random_instance((3, 5), 713)
    assert random_instance((3, 5), 713) != random_instance((3, 5), 714)


def test_random_pilot_replaces_zero_optimum_instances():
    instances = list(random_pilot_instances(per_cell=3, cells=((2, 3),)))
    assert len(instances) == 3
    assert all(solve_exact(instance).score > 0.0 for instance in instances)


def test_runner_writes_passing_tier0_and_smoke_manifest(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.run_second_batch",
            "--stage",
            "smoke",
            "--workers",
            "2",
            "--smoke-count",
            "20",
            "--output",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((tmp_path / "second_batch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["verdict"] == "PASS"
    assert manifest["stages"]["tier0"]["johnson_gate_e_failures"] == 0
    assert manifest["stages"]["smoke"]["johnson_gate_e_failures"] == 0


def test_manifest_collection_preserves_completed_stage_summaries(tmp_path):
    for name, count in (("exhaustive", 24_576), ("random", 1_800)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "summary.json").write_text(
            json.dumps({"instance_count": count, "johnson_gate_e_failures": 0}),
            encoding="utf-8",
        )
    summaries = collect_stage_summaries(tmp_path)
    assert summaries["exhaustive"]["instance_count"] == 24_576
    assert summaries["random"]["instance_count"] == 1_800
