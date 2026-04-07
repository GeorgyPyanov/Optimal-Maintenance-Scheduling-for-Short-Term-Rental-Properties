from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class TimeState:
    elapsed_months: float
    month_of_year: float
    cycle: int


class MaintenanceCalendar:
    def __init__(
        self,
        start_month: float,
        speed_km_per_month: float,
        service_time_months: float,
        max_wait_months: float = 0.0,
    ):
        if speed_km_per_month <= 0:
            raise ValueError("speed_km_per_month must be positive")
        if service_time_months < 0:
            raise ValueError("service_time_months must be non-negative")

        self.start_month = float(start_month)
        self.speed_km_per_month = float(speed_km_per_month)
        self.service_time_months = float(service_time_months)
        self.max_wait_months = float(max_wait_months)

    @staticmethod
    def normalize_month(month: float) -> float:
        return ((float(month) - 1.0) % 12.0) + 1.0

    @staticmethod
    def season_distance(month_a: float, month_b: float) -> float:
        a = MaintenanceCalendar.normalize_month(month_a)
        b = MaintenanceCalendar.normalize_month(month_b)
        diff = abs(a - b)
        return min(diff, 12.0 - diff)

    def initial_state(self) -> TimeState:
        return self.describe_time(0.0)

    def describe_time(self, elapsed_months: float) -> TimeState:
        absolute_month = self.start_month + float(elapsed_months)
        cycle = int(np.floor((absolute_month - 1.0) / 12.0))
        return TimeState(
            elapsed_months=float(elapsed_months),
            month_of_year=self.normalize_month(absolute_month),
            cycle=cycle + 1,
        )

    def advance(self, elapsed_months: float, distance_km: float) -> TimeState:
        travel_time = float(distance_km) / self.speed_km_per_month
        new_elapsed = float(elapsed_months) + travel_time + self.service_time_months
        return self.describe_time(new_elapsed)

    def wait_for_preferred_month(self, current_state: TimeState, preferred_month: float) -> TimeState:
        if self.max_wait_months <= 0:
            return current_state

        current_month = current_state.month_of_year
        preferred = self.normalize_month(preferred_month)
        wait_months = (preferred - current_month) % 12.0

        if wait_months <= 0 or wait_months > self.max_wait_months:
            return current_state

        return self.describe_time(current_state.elapsed_months + wait_months)

    @staticmethod
    def is_within_operating_window(month: float, operating_months: Optional[Tuple[float, float]]) -> bool:
        if operating_months is None:
            return True

        start, end = operating_months
        month = MaintenanceCalendar.normalize_month(month)
        start = MaintenanceCalendar.normalize_month(start)
        end = MaintenanceCalendar.normalize_month(end)

        if start <= end:
            return start <= month <= end
        return month >= start or month <= end

    def schedule_visit(
        self,
        arrival_state: TimeState,
        preferred_month: float,
        season_window_months: float,
        operating_months: Optional[Tuple[float, float]],
    ) -> TimeState:
        if self.is_within_operating_window(arrival_state.month_of_year, operating_months):
            if self.season_distance(arrival_state.month_of_year, preferred_month) <= season_window_months:
                return arrival_state

        waited_state = self.wait_for_preferred_month(arrival_state, preferred_month)
        if waited_state.elapsed_months == arrival_state.elapsed_months:
            return arrival_state

        if not self.is_within_operating_window(waited_state.month_of_year, operating_months):
            return arrival_state

        if self.season_distance(waited_state.month_of_year, preferred_month) <= season_window_months:
            return waited_state

        return arrival_state


def compute_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    coords = df[["latitude", "longitude"]].to_numpy(dtype=float)
    n = len(coords)
    dist_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            lat_dist = (coords[i, 0] - coords[j, 0]) * 111.0
            lon_dist = (coords[i, 1] - coords[j, 1]) * 85.0
            dist_matrix[i, j] = np.hypot(lat_dist, lon_dist)
    return dist_matrix


def load_and_prepare_data(csv_path: str = "result_dataset.csv", n_properties: int = 50):
    df = pd.read_csv(csv_path)

    required_cols = [
        "id",
        "latitude",
        "longitude",
        "neighbourhood",
        "aco_priority",
        "best_maintenance_month",
        "price",
        "complaint_ratio",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in dataset: {missing_cols}")

    df = df[required_cols].dropna().reset_index(drop=True)
    df = df.nlargest(n_properties, "aco_priority").reset_index(drop=True)

    depot = pd.DataFrame(
        [
            {
                "id": -1,
                "latitude": df["latitude"].mean(),
                "longitude": df["longitude"].mean(),
                "neighbourhood": "Depot",
                "aco_priority": 0.0,
                "best_maintenance_month": 1.0,
                "price": 0.0,
                "complaint_ratio": 0.0,
            }
        ]
    )
    planning_df = pd.concat([depot, df], ignore_index=True)

    distances = compute_distance_matrix(planning_df)
    raw_priorities = planning_df["aco_priority"].to_numpy().reshape(-1, 1)
    priorities = MinMaxScaler().fit_transform(raw_priorities).flatten()
    priorities[0] = 0.0
    priorities[1:] = np.clip(priorities[1:], 1e-3, None)
    preferred_months = planning_df["best_maintenance_month"].astype(float).to_numpy()

    return distances, priorities, preferred_months, df


def create_route_table(df: pd.DataFrame, obj_indices: Sequence[int], time_states: Sequence[TimeState]) -> pd.DataFrame:
    df_indices = [idx - 1 for idx in obj_indices]
    route_df = df.iloc[df_indices].copy().reset_index(drop=True)
    route_df["visit_order"] = np.arange(1, len(route_df) + 1)
    route_df["planned_month"] = [state.month_of_year for state in time_states]
    route_df["planning_cycle"] = [state.cycle for state in time_states]
    route_df["elapsed_months"] = [state.elapsed_months for state in time_states]

    cols = [
        "visit_order",
        "id",
        "latitude",
        "longitude",
        "neighbourhood",
        "price",
        "complaint_ratio",
        "aco_priority",
        "best_maintenance_month",
        "planned_month",
        "planning_cycle",
        "elapsed_months",
    ]
    return route_df[cols]


class BaseAntColony:
    def __init__(
        self,
        distances: np.ndarray,
        priorities: Sequence[float],
        preferred_months: Sequence[float],
        n_ants: int,
        n_iterations: int,
        decay: float,
        alpha: float = 1.0,
        beta: float = 2.0,
        q: float = 100.0,
        start_month: float = 1.0,
        speed_km_per_month: float = 100.0,
        service_time_months: float = 0.1,
        season_penalty_weight: float = 1.0,
        max_wait_months: float = 0.0,
        time_penalty_weight: float = 1.0,
        idle_penalty_weight: float = 1.0,
        season_window_months: float = 0.5,
        priority_window_boost: float = 0.5,
        priority_delay_penalty_weight: float = 1.0,
        return_to_depot: bool = True,
        operating_months: Optional[Tuple[float, float]] = None,
    ):
        self.distances = np.asarray(distances, dtype=float)
        self.priorities = np.asarray(priorities, dtype=float)
        self.preferred_months = np.asarray(preferred_months, dtype=float)

        if self.distances.ndim != 2 or self.distances.shape[0] != self.distances.shape[1]:
            raise ValueError("distances must be a square matrix")
        if len(self.priorities) != self.distances.shape[0]:
            raise ValueError("priorities size must match distances")
        if len(self.preferred_months) != self.distances.shape[0]:
            raise ValueError("preferred_months size must match distances")

        self.n_ants = int(n_ants)
        self.n_iterations = int(n_iterations)
        self.decay = float(decay)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.Q = float(q)
        self.calendar = MaintenanceCalendar(
            start_month=start_month,
            speed_km_per_month=speed_km_per_month,
            service_time_months=service_time_months,
            max_wait_months=max_wait_months,
        )
        self.season_penalty_weight = float(season_penalty_weight)
        self.time_penalty_weight = float(time_penalty_weight)
        self.idle_penalty_weight = float(idle_penalty_weight)
        self.season_window_months = float(season_window_months)
        self.priority_window_boost = float(priority_window_boost)
        self.priority_delay_penalty_weight = float(priority_delay_penalty_weight)
        self.return_to_depot = bool(return_to_depot)
        self.operating_months = operating_months

        n = self.distances.shape[0]
        self.pheromones = np.ones((n, n), dtype=float) / max(n, 1)

    def _transition_score(self, current: int, candidate: int, elapsed_months: float) -> float:
        distance = self.distances[current, candidate] + 1e-6
        arrival = self.calendar.advance(elapsed_months, distance)
        scheduled = self.calendar.schedule_visit(
            arrival,
            self.preferred_months[candidate],
            self.season_window_months,
            self.operating_months,
        )
        season_gap = self.calendar.season_distance(self.preferred_months[candidate], scheduled.month_of_year)

        tau = max(self.pheromones[current, candidate], 1e-12)
        eta_distance = 1.0 / distance
        eta_priority = self.priorities[candidate]
        if season_gap <= self.season_window_months:
            eta_priority *= 1.0 + self.priority_window_boost
        wait_penalty = 1.0 / (1.0 + max(0.0, scheduled.elapsed_months - arrival.elapsed_months))
        priority_delay_penalty = 1.0 / (
            1.0 + self.priority_delay_penalty_weight * self.priorities[candidate] * scheduled.elapsed_months
        )
        eta_season = 1.0 / (1.0 + self.season_penalty_weight * season_gap)
        eta = eta_distance * eta_priority * eta_season * wait_penalty * priority_delay_penalty
        return (tau ** self.alpha) * (eta ** self.beta)

    def _select_next(self, current: int, remaining: Sequence[int], elapsed_months: float) -> Tuple[int, TimeState]:
        choices = list(remaining)
        raw_scores = np.array(
            [self._transition_score(current, candidate, elapsed_months) for candidate in choices],
            dtype=float,
        )

        total = raw_scores.sum()
        if total <= 0 or not np.isfinite(total):
            probabilities = np.full(len(choices), 1.0 / len(choices), dtype=float)
        else:
            probabilities = raw_scores / total

        chosen_idx = int(np.random.choice(len(choices), p=probabilities))
        chosen = choices[chosen_idx]
        arrival = self.calendar.advance(elapsed_months, self.distances[current, chosen])
        new_state = self.calendar.schedule_visit(
            arrival,
            self.preferred_months[chosen],
            self.season_window_months,
            self.operating_months,
        )
        return chosen, new_state

    def _route_cost(self, path: Sequence[int], time_states: Sequence[TimeState]) -> float:
        if len(path) <= 1:
            return 0.0

        total_distance = 0.0
        total_season_penalty = 0.0
        total_priority_delay_penalty = 0.0
        route_start_elapsed = time_states[0].elapsed_months
        for index in range(1, len(path)):
            prev_node = path[index - 1]
            node = path[index]
            total_distance += self.distances[prev_node, node]
            total_season_penalty += self.calendar.season_distance(
                self.preferred_months[node],
                time_states[index].month_of_year,
            )
            total_priority_delay_penalty += self.priorities[node] * (time_states[index].elapsed_months - route_start_elapsed)

        initial_service_time = self.distances[path[0], path[1]] / self.calendar.speed_km_per_month + self.calendar.service_time_months
        initial_idle_time = max(0.0, time_states[1].elapsed_months - route_start_elapsed - initial_service_time)

        route_duration = time_states[-1].elapsed_months - route_start_elapsed
        if self.return_to_depot:
            return_distance = self.distances[path[-1], 0]
            total_distance += return_distance
            route_duration += return_distance / self.calendar.speed_km_per_month + self.calendar.service_time_months

        return (
            total_distance
            + self.season_penalty_weight * total_season_penalty
            + self.time_penalty_weight * route_duration
            + self.idle_penalty_weight * initial_idle_time
            + self.priority_delay_penalty_weight * total_priority_delay_penalty
        )

    def _update_pheromones(self, solutions, costs: Sequence[float]):
        self.pheromones *= self.decay
        for solution, cost in zip(solutions, costs):
            if cost <= 0:
                continue
            deposit = self.Q / (cost + 1e-6)
            for path in self._paths_for_pheromones(solution):
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    self.pheromones[a, b] += deposit
                    self.pheromones[b, a] += deposit

    @staticmethod
    def _paths_for_pheromones(solution):
        raise NotImplementedError


class AntColony(BaseAntColony):
    @staticmethod
    def _paths_for_pheromones(solution):
        return [solution[0]]

    def _construct_path(self) -> Tuple[List[int], List[TimeState]]:
        path = [0]
        states = [self.calendar.initial_state()]
        current = 0
        elapsed_months = 0.0
        remaining = set(range(1, self.distances.shape[0]))

        while remaining:
            next_node, next_state = self._select_next(current, remaining, elapsed_months)
            path.append(next_node)
            states.append(next_state)
            remaining.remove(next_node)
            current = next_node
            elapsed_months = next_state.elapsed_months

        return path, states

    def run(self):
        best_path = None
        best_states = None
        best_cost = float("inf")
        best_history = []

        for _ in range(self.n_iterations):
            solutions = [self._construct_path() for _ in range(self.n_ants)]
            costs = [self._route_cost(path, states) for path, states in solutions]
            self._update_pheromones(solutions, costs)

            best_idx = int(np.argmin(costs))
            if costs[best_idx] < best_cost:
                best_path, best_states = solutions[best_idx]
                best_cost = costs[best_idx]
                best_history.append((best_path.copy(), best_cost))

        return best_path, best_states, best_cost, best_history


class MultiAntColony(BaseAntColony):
    def __init__(
        self,
        distances: np.ndarray,
        priorities: Sequence[float],
        preferred_months: Sequence[float],
        n_ants: int = 50,
        n_iterations: int = 200,
        n_vehicles: int = 5,
        decay: float = 0.7,
        alpha: float = 1.0,
        beta: float = 3.0,
        q: float = 100.0,
        start_month: float = 1.0,
        speed_km_per_month: float = 100.0,
        service_time_months: float = 0.1,
        season_penalty_weight: float = 2.0,
        max_months_per_vehicle: Optional[float] = None,
        max_wait_months: float = 0.0,
        time_penalty_weight: float = 1.0,
        idle_penalty_weight: float = 1.0,
        season_window_months: float = 0.5,
        priority_window_boost: float = 0.5,
        priority_delay_penalty_weight: float = 1.0,
        return_to_depot: bool = True,
        operating_months: Optional[Tuple[float, float]] = None,
        vehicle_start_gap_months: float = 0.0,
    ):
        super().__init__(
            distances=distances,
            priorities=priorities,
            preferred_months=preferred_months,
            n_ants=n_ants,
            n_iterations=n_iterations,
            decay=decay,
            alpha=alpha,
            beta=beta,
            q=q,
            start_month=start_month,
            speed_km_per_month=speed_km_per_month,
            service_time_months=service_time_months,
            season_penalty_weight=season_penalty_weight,
            max_wait_months=max_wait_months,
            time_penalty_weight=time_penalty_weight,
            idle_penalty_weight=idle_penalty_weight,
            season_window_months=season_window_months,
            priority_window_boost=priority_window_boost,
            priority_delay_penalty_weight=priority_delay_penalty_weight,
            return_to_depot=return_to_depot,
            operating_months=operating_months,
        )
        self.n_vehicles = int(n_vehicles)
        self.max_months_per_vehicle = None if max_months_per_vehicle is None else float(max_months_per_vehicle)
        self.vehicle_start_gap_months = float(vehicle_start_gap_months)

    @staticmethod
    def _paths_for_pheromones(solution):
        routes, _ = solution
        return routes

    def _route_limit(self, unvisited_count: int) -> int:
        if self.n_vehicles <= 0:
            return unvisited_count
        return int(np.ceil(unvisited_count / self.n_vehicles))

    def _vehicle_start_offset(self, vehicle_idx: int) -> float:
        return max(0.0, vehicle_idx) * self.vehicle_start_gap_months

    def _construct_single_route(self, unvisited: set[int], start_offset: float = 0.0) -> Tuple[List[int], List[TimeState]]:
        path = [0]
        states = [self.calendar.describe_time(start_offset)]
        current = 0
        elapsed_months = start_offset
        remaining = set(unvisited)
        limit = self._route_limit(len(remaining))

        while remaining and len(path) - 1 < limit:
            next_node, next_state = self._select_next(current, remaining, elapsed_months)
            if self.max_months_per_vehicle is not None and next_state.elapsed_months > self.max_months_per_vehicle:
                break

            path.append(next_node)
            states.append(next_state)
            remaining.remove(next_node)
            current = next_node
            elapsed_months = next_state.elapsed_months

        return path, states

    def _construct_multi_route(self) -> Tuple[List[List[int]], List[List[TimeState]]]:
        unvisited = set(range(1, self.distances.shape[0]))
        routes: List[List[int]] = []
        states_by_route: List[List[TimeState]] = []

        for vehicle_idx in range(self.n_vehicles):
            if not unvisited:
                break
            path, states = self._construct_single_route(unvisited, start_offset=self._vehicle_start_offset(vehicle_idx))
            if len(path) == 1:
                break

            routes.append(path)
            states_by_route.append(states)
            for node in path[1:]:
                unvisited.discard(node)

        while unvisited:
            target_route_idx = min(
                range(len(routes)),
                key=lambda idx: states_by_route[idx][-1].elapsed_months if routes[idx] else 0.0,
            )
            route = routes[target_route_idx]
            states = states_by_route[target_route_idx]
            current = route[-1]
            elapsed_months = states[-1].elapsed_months

            next_node, next_state = self._select_next(current, unvisited, elapsed_months)
            route.append(next_node)
            states.append(next_state)
            unvisited.remove(next_node)

        return routes, states_by_route

    def _solution_cost(self, routes: Sequence[Sequence[int]], states_by_route: Sequence[Sequence[TimeState]]) -> float:
        return sum(self._route_cost(route, states) for route, states in zip(routes, states_by_route))

    def run(self):
        best_routes = None
        best_states = None
        best_cost = float("inf")

        for _ in range(self.n_iterations):
            solutions = [self._construct_multi_route() for _ in range(self.n_ants)]
            costs = [self._solution_cost(routes, states) for routes, states in solutions]
            self._update_pheromones(solutions, costs)

            best_idx = int(np.argmin(costs))
            if costs[best_idx] < best_cost:
                best_routes, best_states = solutions[best_idx]
                best_cost = costs[best_idx]

        return (best_routes, best_states), best_cost


if __name__ == "__main__":
    distances, priorities, preferred_months, df = load_and_prepare_data("result_dataset.csv", n_properties=30)

    aco = MultiAntColony(
        distances=distances,
        priorities=priorities,
        preferred_months=preferred_months,
        n_ants=40,
        n_iterations=150,
        n_vehicles=5,
        decay=0.75,
        alpha=1.0,
        beta=3.0,
        q=100.0,
        start_month=1.0,
        speed_km_per_month=100.0,
        service_time_months=0.1,
        season_penalty_weight=8.0,
        max_months_per_vehicle=6.0,
        max_wait_months=4.0,
        time_penalty_weight=3.0,
        idle_penalty_weight=4.0,
        season_window_months=0.35,
        priority_window_boost=0.75,
        priority_delay_penalty_weight=6.0,
        return_to_depot=True,
        operating_months=(3.5, 9.5),
        vehicle_start_gap_months=0.2,
    )

    (best_routes, best_states), best_cost = aco.run()

    print(f"Total properties: {len(df)}")
    print(f"Total cost: {best_cost:.2f}")
    print(f"Routes built: {len(best_routes)}")

    for brigade_idx, (route, states) in enumerate(zip(best_routes, best_states), start=1):
        route_table = create_route_table(df, route[1:], states[1:])
        route_table.to_csv(f"route_brigade_{brigade_idx}.csv", index=False)
        print(f"\nRoute {brigade_idx}")
        for _, row in route_table.iterrows():
            print(
                f"{int(row['visit_order']):3d} | ID {int(row['id']):10d} | "
                f"priority {row['aco_priority']:.3f} | best month {row['best_maintenance_month']:.1f} | "
                f"planned month {row['planned_month']:.2f} | cycle {int(row['planning_cycle'])}"
            )
