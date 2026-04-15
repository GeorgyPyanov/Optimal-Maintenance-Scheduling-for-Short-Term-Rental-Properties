import numpy as np
import pytest

from src.aco import AntColony, MultiAntColony


def _random_problem(seed: int, n_nodes: int):
    """Generate symmetric random routing instance with valid depot conventions."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(1.0, 100.0, size=(n_nodes, n_nodes))
    distances = (raw + raw.T) / 2.0
    np.fill_diagonal(distances, 0.0)

    priorities = rng.uniform(0.05, 1.0, size=n_nodes)
    priorities[0] = 0.0
    preferred_months = rng.integers(1, 13, size=n_nodes).astype(float)
    preferred_months[0] = 1.0
    return distances, priorities, preferred_months


@pytest.mark.parametrize("seed", [3, 7, 21, 44, 78])
def test_ant_colony_path_invariants_hold_for_random_inputs(seed):
    # Property-style test: invariants should hold across multiple random seeds.
    np.random.seed(seed)
    distances, priorities, preferred_months = _random_problem(seed, n_nodes=7)
    aco = AntColony(
        distances=distances,
        priorities=priorities,
        preferred_months=preferred_months,
        n_ants=5,
        n_iterations=5,
        decay=0.75,
        return_to_depot=True,
    )

    best_path, states, cost, _ = aco.run()

    assert best_path[0] == 0
    assert len(best_path) == 7
    assert set(best_path) == set(range(7))
    assert len(states) == len(best_path)
    assert np.isfinite(cost) and cost > 0
    assert all(states[i].elapsed_months <= states[i + 1].elapsed_months for i in range(len(states) - 1))


@pytest.mark.parametrize("seed", [4, 11, 19, 35])
def test_multi_ant_colony_assignment_invariants_hold_for_random_inputs(seed):
    # Property-style test: all nodes are assigned exactly once across routes.
    np.random.seed(seed)
    distances, priorities, preferred_months = _random_problem(seed, n_nodes=8)
    maco = MultiAntColony(
        distances=distances,
        priorities=priorities,
        preferred_months=preferred_months,
        n_ants=4,
        n_iterations=4,
        n_vehicles=3,
        decay=0.7,
        max_months_per_vehicle=4.0,
    )

    (routes, states_by_route), total_cost = maco.run()
    visited = [node for route in routes for node in route[1:]]

    assert set(visited) == set(range(1, 8))
    assert len(visited) == len(set(visited))
    assert all(route[0] == 0 for route in routes)
    assert all(len(route) == len(states) for route, states in zip(routes, states_by_route))
    assert np.isfinite(total_cost) and total_cost > 0
