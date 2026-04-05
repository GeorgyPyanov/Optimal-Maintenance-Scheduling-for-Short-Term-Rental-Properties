import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple


class MultiAntColony:
    def __init__(self, distances, priorities, preferred_months,
                 n_ants=50, n_iterations=200, n_vehicles=5,
                 decay=0.7, alpha=1.0, beta=3.0, Q=100,
                 start_month=1.0, speed_km_per_month=100.0,
                 service_time_months=0.1, season_penalty_weight=2.0,
                 max_months_per_vehicle=None):

        self.distances = distances
        self.priorities = priorities
        self.preferred_months = preferred_months
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.n_vehicles = n_vehicles
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.Q = Q

        self.start_month = start_month
        self.speed_km_per_month = speed_km_per_month
        self.service_time_months = service_time_months
        self.season_penalty_weight = season_penalty_weight
        self.max_months_per_vehicle = max_months_per_vehicle

        n = distances.shape[0]
        self.pheromones = np.ones((n, n)) / n

    def run(self):
        best_solutions = None
        best_total_cost = float('inf')

        for iteration in range(self.n_iterations):
            all_routes, all_months_seq = self._construct_solutions()
            total_costs = [self._solution_cost(routes, months) for routes, months in zip(all_routes, all_months_seq)]

            self._update_pheromones(all_routes, total_costs)

            best_idx = np.argmin(total_costs)
            if total_costs[best_idx] < best_total_cost:
                best_total_cost = total_costs[best_idx]
                best_solutions = (all_routes[best_idx], all_months_seq[best_idx])

        return best_solutions, best_total_cost

    def _construct_solutions(self):
        all_routes = []
        all_months = []
        for _ in range(self.n_ants):
            routes, months_seq = self._construct_multi_route()
            all_routes.append(routes)
            all_months.append(months_seq)
        return all_routes, all_months

    def _construct_multi_route(self) -> Tuple[List[List[int]], List[List[float]]]:
        unvisited = set(range(1, self.distances.shape[0]))
        routes = []
        months_sequences = []

        while unvisited:
            route, months = self._construct_single_route(unvisited)
            if len(route) <= 1:
                break

            routes.append(route)
            months_sequences.append(months)

            for city in route[1:]:
                unvisited.discard(city)

        if unvisited and routes:
            last_route, last_months = routes[-1], months_sequences[-1]
            current = last_route[-1]
            current_month = last_months[-1]

            extra_remaining = unvisited.copy()
            for city in extra_remaining:
                next_city, new_month = self._select_next(current, {city}, current_month)
                last_route.append(next_city)
                last_months.append(new_month)
                current = next_city
                current_month = new_month
                unvisited.remove(city)

        return routes, months_sequences

    def _construct_single_route(self, unvisited: set) -> Tuple[List[int], List[float]]:
        if not unvisited:
            return [0], [self.start_month]

        path = [0]
        current_month = self.start_month
        months = [current_month]
        remaining = unvisited.copy()
        current = 0

        max_objects_per_route = max(8, len(unvisited) // self.n_vehicles + 2)

        while remaining and len(path) - 1 < max_objects_per_route:
            next_city, new_month = self._select_next(current, remaining, current_month)

            if (self.max_months_per_vehicle is not None and
                    new_month - self.start_month > self.max_months_per_vehicle):
                break

            path.append(next_city)
            months.append(new_month)
            remaining.remove(next_city)
            current = next_city
            current_month = new_month

        return path, months
    def _select_next(self, current, remaining, current_month):
        choices = list(remaining)
        probs_raw = []

        for j in choices:
            d = self.distances[current][j] + 1e-8

            travel_time = d / self.speed_km_per_month
            arrival_month = current_month + travel_time + self.service_time_months

            pref = self.preferred_months[j]
            diff = abs(pref - arrival_month)
            diff = min(diff, 12 - diff)

            base_eta = (1.0 / d) * self.priorities[j]
            season_factor = 1.0 / (1.0 + self.season_penalty_weight * diff)
            eta = base_eta * season_factor

            tau = max(self.pheromones[current][j], 1e-10)  # защита от нуля

            prob = (tau ** self.alpha) * (eta ** self.beta)
            probs_raw.append(prob)

        probs = np.array(probs_raw, dtype=np.float64)

        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(choices), dtype=np.float64) / len(choices)

        probs = np.maximum(probs, 0.0)
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            probs = np.ones(len(choices), dtype=np.float64) / len(choices)

        chosen_idx = np.random.choice(len(choices), p=probs)
        chosen = choices[chosen_idx]

        d_chosen = self.distances[current][chosen] + 1e-8
        new_month = current_month + (d_chosen / self.speed_km_per_month) + self.service_time_months

        return chosen, new_month
    def _solution_cost(self, routes: List[List[int]], months_seq: List[List[float]]) -> float:
        return sum(self._single_route_cost(route, months) for route, months in zip(routes, months_seq))

    def _single_route_cost(self, path, months):
        if len(path) <= 1:
            return 0.0
        total_distance = sum(self.distances[path[i]][path[i + 1]] for i in range(len(path) - 1))
        total_season_penalty = 0.0
        for i in range(1, len(path)):
            diff = abs(self.preferred_months[path[i]] - months[i])
            diff = min(diff, 12 - diff)
            total_season_penalty += diff
        return total_distance + self.season_penalty_weight * total_season_penalty

    def _update_pheromones(self, all_routes, total_costs):
        self.pheromones *= self.decay
        for routes, cost in zip(all_routes, total_costs):
            if cost <= 0: continue
            deposit = self.Q / cost
            for route in routes:
                for i in range(len(route) - 1):
                    self.pheromones[route[i]][route[i + 1]] += deposit
                    self.pheromones[route[i + 1]][route[i]] += deposit


def load_and_prepare_data(csv_path='result_dataset.csv', n_properties=50):
    df = pd.read_csv(csv_path)

    required_cols = ['id', 'latitude', 'longitude', 'neighbourhood',
                     'aco_priority', 'best_maintenance_month',
                     'price', 'complaint_ratio']
    df = df[required_cols].dropna().reset_index(drop=True)
    df = df.nlargest(n_properties, 'aco_priority').reset_index(drop=True)

    n = len(df)
    coords = df[['latitude', 'longitude']].values
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            lat_dist = (coords[i, 0] - coords[j, 0]) * 111
            lon_dist = (coords[i, 1] - coords[j, 1]) * 85
            dist_matrix[i, j] = np.sqrt(lat_dist ** 2 + lon_dist ** 2)

    priorities = MinMaxScaler().fit_transform(df['aco_priority'].values.reshape(-1, 1)).flatten()
    preferred_months = df['best_maintenance_month'].values.astype(float)

    return dist_matrix, priorities, preferred_months, df


def create_route_table(df, obj_indices, months_sequence):
    route_df = df.iloc[obj_indices].copy()
    route_df['visit_order'] = range(1, len(route_df) + 1)
    route_df['planned_month'] = months_sequence
    cols = ['visit_order', 'id', 'latitude', 'longitude', 'neighbourhood',
            'price', 'complaint_ratio', 'aco_priority',
            'best_maintenance_month', 'planned_month']
    return route_df[cols]

#DEBUG
# if __name__ == "__main__":
#     distances, priorities, preferred_months, df = load_and_prepare_data('result_dataset.csv', n_properties=30)
#
#     aco = MultiAntColony(
#         distances=distances,
#         priorities=priorities,
#         preferred_months=preferred_months,
#         n_ants=30,
#         n_iterations=80,
#         n_vehicles=5,
#         decay=0.8,
#         alpha=1.0,
#         beta=2.5,
#         Q=80,
#         start_month=1.0,
#         speed_km_per_month=100.0,
#         service_time_months=0.12,
#         season_penalty_weight=1.5,
#         max_months_per_vehicle=6.0
#     )
#
#     (best_routes, best_months_seq), best_cost = aco.run()
#
#     print(f"\nResult")
#     print(f"Overall cost: {best_cost:.2f}")
#     print(f"Number of routes: {len(best_routes)}")
#
#     for v_idx, (route, months) in enumerate(zip(best_routes, best_months_seq)):
#         obj_indices = route[1:]
#         obj_months = months[1:]
#         print(f"\n--- Route {v_idx + 1} | Objects: {len(obj_indices)} ---")
#         table = create_route_table(df, obj_indices, obj_months)
#         print(table[['visit_order', 'id', 'neighbourhood', 'planned_month', 'aco_priority']])
#
#         table.to_csv(f'route_brigade_{v_idx + 1}.csv', index=False)