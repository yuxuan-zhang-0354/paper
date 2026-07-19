from __future__ import annotations

from itertools import combinations, product
from random import Random

from .allocation_model import StaticAgent, StaticInstance, StaticTask, evaluate_agent_path
from .cbba_static import Method, run_static_cbba, validate_result
from .exact_allocation import solve_exact


def tier0_instances() -> tuple[StaticInstance, ...]:
    return (
        StaticInstance(
            agents=(StaticAgent(0, (0.0, 0.0), 4.0, 3),),
            tasks=(
                StaticTask(0, (1.0, 0.0), 1.0, 20.0),
                StaticTask(1, (3.0, 0.0), 0.0, 50.0),
                StaticTask(2, (-1.0, 0.0), 0.0, 20.0),
            ),
            discount=0.95,
            instance_id="deadline_dmg",
        ),
        StaticInstance(
            agents=(StaticAgent(0, (0.0, 0.0), 20.0, 3),),
            tasks=tuple(
                StaticTask(j, (float(j + 1), 0.0), 0.0, value)
                for j, value in enumerate((100.0, 30.0, 20.0))
            ),
            discount=0.9,
            instance_id="warp_order",
        ),
        StaticInstance(
            agents=(StaticAgent(0, (5.0, 3.0), 30.0, 3),),
            tasks=(
                StaticTask(0, (3.0, -1.0), 0.0, 60.0),
                StaticTask(1, (-10.0, 5.0), 1.0, 100.0),
                StaticTask(2, (7.0, 1.0), 0.0, 60.0),
            ),
            discount=0.98,
            instance_id="warp_activation",
        ),
        StaticInstance(
            agents=(
                StaticAgent(0, (5.0, 3.0), 30.0, 3),
                StaticAgent(1, (7.0, 10.3), 10.0, 1),
            ),
            tasks=(
                StaticTask(0, (3.0, -1.0), 0.0, 60.0),
                StaticTask(1, (-10.0, 5.0), 1.0, 100.0),
                StaticTask(2, (7.0, 1.0), 0.0, 60.0),
            ),
            discount=0.98,
            instance_id="warp_competition",
        ),
        StaticInstance(
            agents=(
                StaticAgent(0, (-2.0, 0.0), 8.0, 3),
                StaticAgent(1, (2.0, 0.0), 8.0, 3),
            ),
            tasks=(
                StaticTask(0, (0.0, 0.0), 0.0, 20.0),
                StaticTask(1, (-1.0, 0.0), 1.0, 50.0),
                StaticTask(2, (1.0, 0.0), 1.0, 50.0),
            ),
            discount=0.98,
            instance_id="two_agent_tie",
        ),
    )


def exhaustive_instances():
    locations = (-3.0, -1.0, 1.0, 3.0)
    index = 0
    for task_x in combinations(locations, 3):
        for origins in product(locations, repeat=2):
            for values in product((20.0, 50.0), repeat=3):
                for services in product((0.0, 1.0), repeat=3):
                    for discount in (0.95, 0.98):
                        for deadline in (4.0, 6.0, 8.0):
                            yield StaticInstance(
                                agents=tuple(
                                    StaticAgent(i, (origin, 0.0), deadline, 3)
                                    for i, origin in enumerate(origins)
                                ),
                                tasks=tuple(
                                    StaticTask(j, (x, 0.0), services[j], values[j])
                                    for j, x in enumerate(task_x)
                                ),
                                discount=discount,
                                instance_id=f"E-{index:05d}",
                            )
                            index += 1


def random_instance(cell: tuple[int, int], seed: int) -> StaticInstance:
    n_agents, n_tasks = cell
    if n_agents not in (2, 3, 4) or n_tasks not in range(3, 9):
        raise ValueError("random cell must have N in {2,3,4} and M in {3,...,8}")
    rng = Random((n_agents * 100_000_000) + (n_tasks * 1_000_000) + seed)
    tightness = ("tight", "medium", "loose")[seed % 3]
    deadline = {"tight": 16.0, "medium": 26.0, "loose": 40.0}[tightness]
    capacity_options = (2, 3, min(4, n_tasks))
    agents = tuple(
        StaticAgent(i, (float(rng.randint(-10, 10)), float(rng.randint(-10, 10))), deadline, rng.choice(capacity_options))
        for i in range(n_agents)
    )
    tasks = tuple(
        StaticTask(
            j,
            (float(rng.randint(-10, 10)), float(rng.randint(-10, 10))),
            float(rng.choice((0, 1, 2, 4))),
            float(rng.choice((20, 40, 60, 100))),
        )
        for j in range(n_tasks)
    )
    return StaticInstance(
        agents=agents,
        tasks=tasks,
        discount=rng.choice((0.95, 0.98, 1.0)),
        instance_id=f"R-N{n_agents}-M{n_tasks}-{tightness}-S{seed}",
    )


def random_pilot_instances(
    per_cell: int = 100,
    cells: tuple[tuple[int, int], ...] | None = None,
):
    if per_cell < 1:
        raise ValueError("per_cell must be positive")
    selected_cells = cells or tuple(
        (n_agents, n_tasks) for n_agents in (2, 3, 4) for n_tasks in range(3, 9)
    )
    for cell in selected_cells:
        accepted = 0
        seed = 0
        while accepted < per_cell:
            instance = random_instance(cell, seed)
            # All task values are positive, so J* > 0 iff at least one
            # single-task path is feasible for at least one agent.
            if any(
                evaluate_agent_path(instance, agent.agent_id, (task.task_id,)).feasible
                for agent in instance.agents
                for task in instance.tasks
            ):
                yield instance
                accepted += 1
            seed += 1


def evaluate_instance(instance: StaticInstance) -> list[dict]:
    exact = solve_exact(instance)
    records: list[dict] = []
    for method in Method:
        result = run_static_cbba(instance, method)
        report = validate_result(instance, result)
        gate_failures = sum(report.values())
        if result.status != "converged":
            gate_failures += 1
        assigned = sorted(
            task_id for task_id, winner in enumerate(result.winners) if winner is not None
        )
        records.append(
            {
                "instance_id": instance.instance_id,
                "method": method.value,
                "status": result.status,
                "rounds": result.rounds,
                "cycle_period": result.cycle_period,
                "true_score": result.true_score,
                "exact_score": exact.score,
                "ratio": result.true_score / exact.score if exact.score > 0 else 1.0,
                "regret": exact.score - result.true_score,
                "assigned": assigned,
                "paths": [list(state.path) for state in result.states],
                "warped_bid_count": sum(
                    entry.external_bid < entry.raw_score - 1e-12
                    for state in result.states
                    for entry in state.entries
                ) if method is Method.JOHNSON_WARPED else 0,
                "gate_e_failures": gate_failures,
                **report,
            }
        )
    return records
