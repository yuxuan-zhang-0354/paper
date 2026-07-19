import json

import numpy as np

from experiments.reproduce_dmg_counterexample import write_counterexample_artifacts
from uav_lifecycle.path_score import (
    Task,
    all_insertions,
    best_insertion,
    evaluate_path,
    reproduce_deadline_counterexample,
)


def test_deadline_counterexample_matches_audited_values():
    result = reproduce_deadline_counterexample()
    np.testing.assert_allclose(result["raw_small"], 23.318506084536523, atol=1e-9)
    np.testing.assert_allclose(result["raw_large"], 22.519091091023625, atol=1e-9)
    np.testing.assert_allclose(
        result["feasible_small"], 9.877664575357215, atol=1e-9
    )
    np.testing.assert_allclose(
        result["feasible_large"], 22.519091091023625, atol=1e-9
    )
    assert result["raw_dmg_holds"] is True
    assert result["constrained_dmg_violated"] is True


def test_route_score_discounts_at_service_start_and_completes_after_service():
    task = Task("T", (3.0, 4.0), service_time=2.0, value=10.0)
    evaluation = evaluate_path((task,), discount_base=0.9)
    np.testing.assert_allclose(evaluation.score, 0.9**5 * 10.0)
    np.testing.assert_allclose(evaluation.start_times, [5.0])
    np.testing.assert_allclose(evaluation.completion_time, 7.0)


def test_all_insertions_preserve_existing_task_order():
    first = Task("A", (0.0, 1.0), 0.0, 1.0)
    second = Task("B", (0.0, 2.0), 0.0, 1.0)
    new = Task("J", (0.0, 3.0), 0.0, 1.0)
    inserted = all_insertions((first, second), new)
    assert [[task.task_id for task in path] for path in inserted] == [
        ["J", "A", "B"],
        ["A", "J", "B"],
        ["A", "B", "J"],
    ]


def test_best_insertion_returns_none_when_every_candidate_misses_deadline():
    base = (Task("A", (10.0, 0.0), 0.0, 1.0),)
    new = Task("J", (20.0, 0.0), 0.0, 1.0)
    assert best_insertion(base, new, discount_base=0.98, deadline=5.0) is None


def test_counterexample_runner_persists_machine_readable_artifacts(tmp_path):
    result = write_counterexample_artifacts(tmp_path)
    persisted = json.loads(
        (tmp_path / "reproduction.json").read_text(encoding="utf-8")
    )
    assert persisted == result
    assert {path.name for path in tmp_path.iterdir()} == {
        "config.json",
        "reproduction.json",
        "run.log",
    }
