import numpy as np
import pandas as pd
import pytest

from src.aco import (
    AntColony,
    MaintenanceCalendar,
    MultiAntColony,
    compute_distance_matrix,
    create_route_table,
    load_and_prepare_data,
)


# Validation checks for core calendar input parameters.
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"speed_km_per_month": 0.0}, "speed_km_per_month must be positive"),
        ({"service_time_months": -0.1}, "service_time_months must be non-negative"),
    ],
)
def test_calendar_input_validation(kwargs, message):
    base = dict(
        start_month=1.0,
        speed_km_per_month=100.0,
        service_time_months=0.1,
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        MaintenanceCalendar(**base)


def _small_problem():
    """Compact deterministic fixture for algorithm-level unit tests."""
    distances = np.array(
        [
            [0.0, 10.0, 20.0, 30.0],
            [10.0, 0.0, 15.0, 25.0],
            [20.0, 15.0, 0.0, 12.0],
            [30.0, 25.0, 12.0, 0.0],
        ]
    )
    priorities = np.array([0.0, 0.8, 0.5, 0.9])
    preferred_months = np.array([1.0, 4.0, 6.0, 9.0])
    return distances, priorities, preferred_months


def test_calendar_normalization_and_season_distance():
    # Month arithmetic must work correctly on a cyclic 12-month scale.
    assert MaintenanceCalendar.normalize_month(13) == pytest.approx(1.0)
    assert MaintenanceCalendar.normalize_month(0) == pytest.approx(12.0)
    assert MaintenanceCalendar.season_distance(12, 1) == pytest.approx(1.0)
    assert MaintenanceCalendar.season_distance(3, 9) == pytest.approx(6.0)


def test_calendar_schedule_waits_when_possible():
    # If waiting budget allows and month is outside tolerance, planner should wait.
    calendar = MaintenanceCalendar(
        start_month=1.0,
        speed_km_per_month=100.0,
        service_time_months=0.0,
        max_wait_months=3.0,
    )
    arrival = calendar.describe_time(1.2)  # month 2.2
    scheduled = calendar.schedule_visit(
        arrival_state=arrival,
        preferred_month=4.0,
        season_window_months=0.25,
        operating_months=(3.0, 9.0),
    )

    assert scheduled.elapsed_months > arrival.elapsed_months
    assert scheduled.month_of_year == pytest.approx(4.0)


def test_compute_distance_matrix_returns_expected_values():
    # 1 degree latitude is approximated as 111 km in the implementation.
    df = pd.DataFrame(
        [
            {"latitude": 0.0, "longitude": 0.0},
            {"latitude": 1.0, "longitude": 0.0},
        ]
    )
    distances = compute_distance_matrix(df)

    assert distances.shape == (2, 2)
    assert distances[0, 0] == pytest.approx(0.0)
    assert distances[0, 1] == pytest.approx(111.0)
    assert distances[1, 0] == pytest.approx(111.0)


def test_load_and_prepare_data_builds_depot_and_scaled_priorities(tmp_path):
    # Verify depot insertion and stable priority scaling.
    data = pd.DataFrame(
        [
            {
                "id": 101,
                "latitude": 39.9,
                "longitude": 116.4,
                "neighbourhood": "A",
                "aco_priority": 0.9,
                "best_maintenance_month": 2,
                "price": 100.0,
                "complaint_ratio": 0.2,
            },
            {
                "id": 102,
                "latitude": 39.8,
                "longitude": 116.3,
                "neighbourhood": "B",
                "aco_priority": 0.2,
                "best_maintenance_month": 5,
                "price": 80.0,
                "complaint_ratio": 0.1,
            },
            {
                "id": 103,
                "latitude": 40.0,
                "longitude": 116.5,
                "neighbourhood": "C",
                "aco_priority": 0.7,
                "best_maintenance_month": 8,
                "price": 120.0,
                "complaint_ratio": 0.3,
            },
        ]
    )
    csv_path = tmp_path / "dataset.csv"
    data.to_csv(csv_path, index=False)

    distances, priorities, preferred_months, top_df = load_and_prepare_data(str(csv_path), n_properties=2)

    assert distances.shape == (3, 3)  # depot + 2 properties
    assert priorities[0] == pytest.approx(0.0)
    assert np.all(priorities[1:] >= 1e-3)
    assert len(preferred_months) == 3
    assert len(top_df) == 2
    assert set(top_df["id"].tolist()) == {101, 103}


def test_load_and_prepare_data_raises_for_missing_required_columns(tmp_path):
    # Data contract guard: required columns must exist.
    broken = pd.DataFrame(
        [
            {
                "id": 1,
                "latitude": 10.0,
                "longitude": 20.0,
                "neighbourhood": "A",
                "aco_priority": 0.5,
                "best_maintenance_month": 4,
                "price": 100.0,
                # complaint_ratio intentionally missing
            }
        ]
    )
    csv_path = tmp_path / "broken.csv"
    broken.to_csv(csv_path, index=False)

    with pytest.raises(KeyError, match="Columns not found in dataset"):
        load_and_prepare_data(str(csv_path), n_properties=1)


def test_load_and_prepare_data_equal_priorities_are_clipped_to_positive(tmp_path):
    # All non-depot nodes must remain selectable even when priorities are flat.
    rows = []
    for i in range(3):
        rows.append(
            {
                "id": 10 + i,
                "latitude": 39.9 + i * 0.01,
                "longitude": 116.4 + i * 0.01,
                "neighbourhood": "N",
                "aco_priority": 0.5,
                "best_maintenance_month": 4.0,
                "price": 100.0,
                "complaint_ratio": 0.0,
            }
        )
    csv_path = tmp_path / "flat_priority.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    _, priorities, _, _ = load_and_prepare_data(str(csv_path), n_properties=3)

    assert priorities[0] == pytest.approx(0.0)
    assert np.all(priorities[1:] >= 1e-3)


def test_create_route_table_maps_indices_and_states():
    # Route table should preserve order and transfer state metadata correctly.
    df = pd.DataFrame(
        [
            {
                "id": 101,
                "latitude": 1.0,
                "longitude": 1.0,
                "neighbourhood": "A",
                "price": 100.0,
                "complaint_ratio": 0.1,
                "aco_priority": 0.9,
                "best_maintenance_month": 2.0,
            },
            {
                "id": 102,
                "latitude": 2.0,
                "longitude": 2.0,
                "neighbourhood": "B",
                "price": 110.0,
                "complaint_ratio": 0.2,
                "aco_priority": 0.8,
                "best_maintenance_month": 3.0,
            },
        ]
    )
    calendar = MaintenanceCalendar(1.0, 100.0, 0.1)
    states = [calendar.describe_time(0.5), calendar.describe_time(1.3)]
    table = create_route_table(df, obj_indices=[1, 2], time_states=states)

    assert table["visit_order"].tolist() == [1, 2]
    assert table["id"].tolist() == [101, 102]
    assert table["planned_month"].tolist() == pytest.approx([1.5, 2.3])
    assert table["planning_cycle"].tolist() == [1, 1]


def test_ant_colony_run_returns_valid_full_path():
    # Single-route solver must visit each node exactly once.
    np.random.seed(0)
    distances, priorities, preferred_months = _small_problem()
    aco = AntColony(
        distances=distances,
        priorities=priorities,
        preferred_months=preferred_months,
        n_ants=4,
        n_iterations=6,
        decay=0.8,
        beta=2.0,
        return_to_depot=True,
    )

    best_path, best_states, best_cost, history = aco.run()

    assert best_path[0] == 0
    assert len(best_path) == distances.shape[0]
    assert set(best_path) == set(range(distances.shape[0]))
    assert len(best_states) == len(best_path)
    assert best_cost > 0
    assert len(history) >= 1
    assert all(
        best_states[i].elapsed_months <= best_states[i + 1].elapsed_months
        for i in range(len(best_states) - 1)
    )


def test_route_cost_includes_return_distance_when_enabled():
    # Return-to-depot flag must affect total route distance.
    distances = np.array([[0.0, 10.0], [10.0, 0.0]])
    priorities = np.array([0.0, 1.0])
    preferred_months = np.array([1.0, 1.0])

    aco_no_return = AntColony(
        distances=distances,
        priorities=priorities,
        preferred_months=preferred_months,
        n_ants=1,
        n_iterations=1,
        decay=0.9,
        season_penalty_weight=0.0,
        time_penalty_weight=0.0,
        idle_penalty_weight=0.0,
        priority_delay_penalty_weight=0.0,
        return_to_depot=False,
    )
    aco_with_return = AntColony(
        distances=distances,
        priorities=priorities,
        preferred_months=preferred_months,
        n_ants=1,
        n_iterations=1,
        decay=0.9,
        season_penalty_weight=0.0,
        time_penalty_weight=0.0,
        idle_penalty_weight=0.0,
        priority_delay_penalty_weight=0.0,
        return_to_depot=True,
    )

    path = [0, 1]
    states = [aco_no_return.calendar.initial_state(), aco_no_return.calendar.advance(0.0, 10.0)]
    cost_no_return = aco_no_return._route_cost(path, states)
    cost_with_return = aco_with_return._route_cost(path, states)

    assert cost_no_return == pytest.approx(10.0)
    assert cost_with_return == pytest.approx(20.0)


def test_multi_ant_colony_constructs_routes_covering_all_nodes():
    # Multi-route solver must assign all non-depot nodes without duplication.
    np.random.seed(1)
    distances, priorities, preferred_months = _small_problem()
    maco = MultiAntColony(
        distances=distances,
        priorities=priorities,
        preferred_months=preferred_months,
        n_ants=3,
        n_iterations=4,
        n_vehicles=2,
        decay=0.7,
        vehicle_start_gap_months=0.5,
    )

    (routes, states), best_cost = maco.run()
    visited = [node for route in routes for node in route[1:]]

    assert set(visited) == {1, 2, 3}
    assert len(visited) == len(set(visited))
    assert best_cost > 0
    assert all(route[0] == 0 for route in routes)
    assert all(len(route) == len(state_list) for route, state_list in zip(routes, states))
