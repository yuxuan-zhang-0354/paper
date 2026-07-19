"""Select a readable frozen D2 public snapshot for the Figure 3 mechanism plot."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from uav_lifecycle.dynamic_planning import build_planning_problem
from uav_lifecycle.dynamic_scenarios import D1_CELLS, generate_d2_scenario
from uav_lifecycle.dynamic_types import DynamicConfig, PublicAgent, PublicSnapshot, PublicTarget
from uav_lifecycle.mode_allocation import Mode, evaluate_mode_path
from uav_lifecycle.mode_cbba import screen_modes


P_GRID = np.linspace(0.02, 0.70, 137)
EXPECTED = ("defer", "recon", "bda", "attack")


def evaluate(item: tuple[str, int]) -> dict | None:
    cell_id, seed = item
    config = DynamicConfig()
    cell = next(cell for cell in D1_CELLS if cell.cell_id == cell_id)
    scenario = generate_d2_scenario(cell, seed, config.config_id)
    target = scenario.targets[0]
    agents = tuple(
        PublicAgent(a.agent_id, a.position, a.ammo.available, a.distance.available, None)
        for a in scenario.agents
    )
    rows = []
    for p in P_GRID:
        belief = (p*p, p*(1-p), (1-p)*p, (1-p)*(1-p))
        snapshot = PublicSnapshot(0, (PublicTarget(0, target.position, belief),), agents, (), (), ())
        problem = build_planning_problem(snapshot, config, scenario.t_max_tick)
        scores = {"defer": 0.0}
        for mode in Mode:
            values = []
            for agent_id in range(len(problem.instance.agents)):
                result = evaluate_mode_path(problem.instance, agent_id, ((0, mode),))
                if result.feasible:
                    values.append(result.score)
            scores[mode.value] = max(values) if values else float("-inf")
        selected = screen_modes(problem.instance)[0]
        best = "defer" if selected is None else selected.mode.value
        rows.append((float(p), scores, best))

    sequence = []
    starts = []
    for p, _, best in rows:
        if not sequence or best != sequence[-1]:
            sequence.append(best)
            starts.append(p)
    if tuple(sequence) != EXPECTED:
        return None
    starts.append(float(P_GRID[-1]))
    widths = {mode: starts[i+1]-starts[i] for i, mode in enumerate(sequence)}
    margins = {}
    for mode in ("recon", "bda"):
        margins[mode] = max(
            scores[mode] - max(value for other, value in scores.items() if other != mode)
            for _, scores, best in rows if best == mode
        )
    # Prefer two visible information-action intervals and clear winning margins.
    readability = min(widths["recon"], widths["bda"]) + 0.30 * min(margins.values()) + 0.10 * sum(margins.values())
    return {
        "scenario_id": scenario.scenario_id,
        "cell_id": cell_id,
        "seed": seed,
        "target_id": 0,
        "target_position": target.position,
        "agent_positions": [agent.position for agent in agents],
        "widths": widths,
        "margins": margins,
        "transition_p": starts[1:-1],
        "readability": readability,
    }


def main() -> None:
    items = [
        (cell_id, seed)
        for cell_id in ("N3-M5-Rtight", "N3-M5-Rloose")
        for seed in range(1000, 1512)
    ]
    with ProcessPoolExecutor(max_workers=22) as pool:
        candidates = [result for result in pool.map(evaluate, items, chunksize=4) if result is not None]
    candidates.sort(key=lambda row: (-row["readability"], row["scenario_id"]))
    output = {
        "rule": "exact D-R-B-A sequence; maximize minimum R/B interval and winning margins",
        "evaluated": len(items),
        "eligible": len(candidates),
        "top_candidates": candidates[:20],
    }
    path = Path("results/figure_design/figure3_reference_selection.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
