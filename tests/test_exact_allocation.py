import itertools

import numpy as np

from uav_lifecycle.allocation_model import StaticAgent, StaticInstance, StaticTask, evaluate_agent_path
from uav_lifecycle.exact_allocation import solve_exact


def test_exact_solver_may_leave_task_unassigned():
    instance = StaticInstance(
        agents=(StaticAgent(0, (0.0, 0.0), 1.5, 2),),
        tasks=(
            StaticTask(0, (1.0, 0.0), 0.0, 10.0),
            StaticTask(1, (10.0, 0.0), 0.0, 100.0),
        ),
        discount=1.0,
    )
    result = solve_exact(instance)
    assert result.paths == ((0,),)
    assert result.score == 10.0


def test_two_agent_solver_matches_direct_disjoint_enumeration():
    instance = StaticInstance(
        agents=(
            StaticAgent(0, (-1.0, 0.0), 10.0, 2),
            StaticAgent(1, (3.0, 0.0), 10.0, 2),
        ),
        tasks=tuple(StaticTask(j, (float(j), 0.0), 0.0, 10.0 + j) for j in range(3)),
        discount=0.95,
    )
    direct = -1.0
    for length0 in range(3):
        for p0 in itertools.permutations(range(3), length0):
            remaining = tuple(j for j in range(3) if j not in p0)
            for length1 in range(min(2, len(remaining)) + 1):
                for p1 in itertools.permutations(remaining, length1):
                    e0 = evaluate_agent_path(instance, 0, p0)
                    e1 = evaluate_agent_path(instance, 1, p1)
                    if e0.feasible and e1.feasible:
                        direct = max(direct, e0.score + e1.score)
    np.testing.assert_allclose(solve_exact(instance).score, direct)


def test_exact_tie_uses_lexicographically_small_path():
    instance = StaticInstance(
        agents=(StaticAgent(0, (0.0, 0.0), 10.0, 2),),
        tasks=(StaticTask(0, (0.0, 0.0), 0.0, 1.0), StaticTask(1, (0.0, 0.0), 0.0, 1.0)),
        discount=1.0,
    )
    assert solve_exact(instance).paths == ((0, 1),)
