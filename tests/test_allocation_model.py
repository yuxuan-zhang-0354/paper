import numpy as np
import pytest

from uav_lifecycle.allocation_model import (
    StaticAgent,
    StaticInstance,
    StaticTask,
    best_true_insertion,
    evaluate_agent_path,
)


def make_instance(deadline=10.0):
    return StaticInstance(
        agents=(StaticAgent(0, (0.0, 0.0), deadline, 3),),
        tasks=(
            StaticTask(0, (1.0, 0.0), 1.0, 10.0),
            StaticTask(1, (2.0, 0.0), 0.0, 20.0),
        ),
        discount=0.9,
    )


def test_path_uses_start_time_discount_and_completion_deadline():
    result = evaluate_agent_path(make_instance(), 0, (0, 1))
    np.testing.assert_allclose(result.score, 10 * 0.9**1 + 20 * 0.9**3)
    assert result.completion_time == 3.0
    assert result.feasible


def test_service_completion_beyond_deadline_is_infeasible():
    result = evaluate_agent_path(make_instance(deadline=2.9), 0, (0, 1))
    assert not result.feasible


def test_best_insertion_uses_earliest_position_on_exact_tie():
    instance = StaticInstance(
        agents=(StaticAgent(0, (0.0, 0.0), 10.0, 2),),
        tasks=(
            StaticTask(0, (0.0, 0.0), 0.0, 1.0),
            StaticTask(1, (0.0, 0.0), 0.0, 1.0),
        ),
        discount=1.0,
    )
    insertion = best_true_insertion(instance, 0, (0,), 1)
    assert insertion.position == 0
    assert insertion.path == (1, 0)
    assert insertion.marginal == 1.0


def test_duplicate_task_and_capacity_overflow_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_agent_path(make_instance(), 0, (0, 0))
    overflow = StaticInstance(
        agents=(StaticAgent(0, (0.0, 0.0), 10.0, 1),),
        tasks=make_instance().tasks,
        discount=0.9,
    )
    assert not evaluate_agent_path(overflow, 0, (0, 1)).feasible

