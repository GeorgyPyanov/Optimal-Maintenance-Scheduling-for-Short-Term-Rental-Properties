from __future__ import annotations

import numpy as np
from typing import List, Tuple
from collections import defaultdict

class NearestNeighbor:
    def __init__(self, distances: np.ndarray):
        self.distances = distances

    def solve_multi_route(self, distances: np.ndarray, calendar,
                          preferred_months: np.ndarray,
                          max_months: float = 6.0,
                          n_vehicles: int = 5) -> Tuple[List[List[int]], List]:
        unvisited = set(range(1, len(distances)))
        routes = []
        states_by_route = []

        for vehicle_idx in range(n_vehicles):
            if not unvisited:
                break

            route = [0]
            current = 0
            elapsed = vehicle_idx * 0.2
            states = [calendar.describe_time(elapsed)]
            route_time = 0.0

            while unvisited:
                best_node = None
                best_dist = float('inf')
                best_state = None

                for node in unvisited:
                    dist = distances[current, node]
                    arrival = calendar.advance(elapsed, dist)
                    scheduled = calendar.schedule_visit(
                        arrival, preferred_months[node], 0.35, (3.5, 9.5)
                    )

                    time_in_route = scheduled.elapsed_months - elapsed
                    if route_time + time_in_route <= max_months:
                        if dist < best_dist:
                            best_dist = dist
                            best_node = node
                            best_state = scheduled

                if best_node is None:
                    break

                route.append(best_node)
                states.append(best_state)
                unvisited.remove(best_node)
                current = best_node
                elapsed = best_state.elapsed_months
                route_time = elapsed - states[0].elapsed_months

            if len(route) > 1:
                routes.append(route)
                states_by_route.append(states)

        return routes, states_by_route

class BinPacking:
    def __init__(self, distances: np.ndarray, calendar,
                 max_months_per_vehicle: float = 6.0):
        self.distances = distances
        self.calendar = calendar
        self.max_months = max_months_per_vehicle

    def _estimate_route_time(self, nodes: List[int]) -> float:
        if not nodes:
            return 0.0
        total = 0.0
        current = 0
        for node in nodes:
            total += self.distances[current, node] / self.calendar.speed_km_per_month
            total += self.calendar.service_time_months
            current = node
        total += self.distances[current, 0] / self.calendar.speed_km_per_month
        return total

    def solve(self, nodes: List[int], max_bins: int = 5) -> List[List[int]]:
        if not nodes:
            return []

        items = [(node, self._estimate_route_time([node])) for node in nodes]
        items.sort(key=lambda x: x[1], reverse=True)

        bins = []
        bin_loads = []

        for node, weight in items:
            placed = False
            for i in range(len(bins)):
                temp_route = bins[i] + [node]
                new_load = self._estimate_route_time(temp_route)
                if new_load <= self.max_months:
                    bins[i].append(node)
                    bin_loads[i] = new_load
                    placed = True
                    break

            if not placed and len(bins) < max_bins:
                bins.append([node])
                bin_loads.append(self._estimate_route_time([node]))

        return bins

class PrimMST:
    def __init__(self, distances: np.ndarray):
        self.distances = distances
        self.V = len(distances)

    def _min_key(self, key, mst_set):
        min_val = float('inf')
        min_index = -1
        for v in range(self.V):
            if key[v] < min_val and not mst_set[v]:
                min_val = key[v]
                min_index = v
        return min_index

    def prim_mst(self) -> Tuple[List[Tuple[int, int]], float]:
        key = [float('inf')] * self.V
        parent = [-1] * self.V
        mst_set = [False] * self.V

        key[0] = 0
        parent[0] = -1

        for _ in range(self.V):
            u = self._min_key(key, mst_set)
            if u == -1:
                break
            mst_set[u] = True
            for v in range(self.V):
                if (self.distances[u, v] > 0 and
                    not mst_set[v] and
                    key[v] > self.distances[u, v]):
                    key[v] = self.distances[u, v]
                    parent[v] = u

        edges = []
        total_weight = 0.0
        for v in range(1, self.V):
            if parent[v] != -1:
                edges.append((parent[v], v))
                total_weight += self.distances[parent[v], v]

        return edges, total_weight

    def get_dfs_tour(self, edges: List[Tuple[int, int]]) -> List[int]:
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        visited = set()
        tour = []

        def dfs(node):
            visited.add(node)
            tour.append(node)
            for neighbor in sorted(adj[node]):
                if neighbor not in visited:
                    dfs(neighbor)

        dfs(0)
        tour.append(0)
        return tour

    def split_tour_into_routes(self, tour: List[int], calendar,
                                preferred_months: np.ndarray,
                                distances: np.ndarray,
                                max_months: float = 6.0) -> Tuple[List[List[int]], List]:
        routes = []
        states_by_route = []

        current_route = [0]
        current_time = 0.0
        elapsed = 0.0
        states = [calendar.describe_time(0)]

        for node in tour[1:]:
            if node == 0:
                continue

            dist = distances[current_route[-1], node]
            travel_time = dist / calendar.speed_km_per_month

            if current_time + travel_time <= max_months:
                current_route.append(node)
                current_time += travel_time + calendar.service_time_months
                arrival = calendar.advance(elapsed, dist)
                scheduled = calendar.schedule_visit(
                    arrival, preferred_months[node], 0.35, (3.5, 9.5)
                )
                states.append(scheduled)
                elapsed = scheduled.elapsed_months
            else:
                if len(current_route) > 1:
                    routes.append(current_route)
                    states_by_route.append(states)

                current_route = [0, node]
                new_start = len(routes) * 0.2
                elapsed = new_start
                current_time = 0.0
                states = [calendar.describe_time(new_start)]
                arrival = calendar.advance(new_start, distances[0, node])
                scheduled = calendar.schedule_visit(
                    arrival, preferred_months[node], 0.35, (3.5, 9.5)
                )
                states.append(scheduled)
                elapsed = scheduled.elapsed_months
                current_time = elapsed - new_start

        if len(current_route) > 1:
            routes.append(current_route)
            states_by_route.append(states)

        return routes, states_by_route

class Adapter:
    def __init__(self, distances: np.ndarray, priorities: np.ndarray,
                 preferred_months: np.ndarray, calendar,
                 method: str = "nearest_neighbor"):
        self.distances = distances
        self.priorities = priorities
        self.preferred_months = preferred_months
        self.calendar = calendar
        self.method = method

    def construct_routes(self, n_vehicles: int = 5, max_months_per_vehicle: float = 6.0):
        if self.method == "nearest_neighbor":
            nn = NearestNeighbor(self.distances)
            return nn.solve_multi_route(self.distances, self.calendar,
                                         self.preferred_months, max_months_per_vehicle, n_vehicles)
        elif self.method == "bin_packing":
            return self._bin_packing_routes(n_vehicles, max_months_per_vehicle)
        elif self.method == "mst":
            return self._mst_routes(n_vehicles, max_months_per_vehicle)
        else:
            return self._nearest_neighbor_routes(n_vehicles, max_months_per_vehicle)

    def _build_states(self, route: List[int], start_offset: float) -> List:
        states = [self.calendar.describe_time(start_offset)]
        elapsed = start_offset

        for i in range(1, len(route)):
            dist = self.distances[route[i-1], route[i]]
            arrival = self.calendar.advance(elapsed, dist)
            scheduled = self.calendar.schedule_visit(
                arrival, self.preferred_months[route[i]], 0.35, (3.5, 9.5)
            )
            states.append(scheduled)
            elapsed = scheduled.elapsed_months

        return states

    def _bin_packing_routes(self, n_vehicles: int, max_months: float):
        all_nodes = list(range(1, len(self.distances)))
        bp = BinPacking(self.distances, self.calendar, max_months)
        routes_list = bp.solve(all_nodes, n_vehicles)

        routes = [[0] + route for route in routes_list]
        states = [self._build_states(route, idx * 0.2) for idx, route in enumerate(routes)]

        return routes, states

    def _mst_routes(self, n_vehicles: int, max_months: float):
        mst_solver = PrimMST(self.distances)
        edges, _ = mst_solver.prim_mst()
        tour = mst_solver.get_dfs_tour(edges)
        routes, states = mst_solver.split_tour_into_routes(
            tour, self.calendar, self.preferred_months, self.distances, max_months
        )
        return routes, states

    def _nearest_neighbor_routes(self, n_vehicles: int, max_months: float):
        nn = NearestNeighbor(self.distances)
        return nn.solve_multi_route(self.distances, self.calendar,
                                     self.preferred_months, max_months, n_vehicles)

def calculate_full_cost(routes, states, distances, calendar, preferred_months, priorities):
    total = 0.0

    for route, states_list in zip(routes, states):
        if len(route) <= 1:
            continue

        total_distance = 0.0
        total_season_penalty = 0.0
        total_priority_delay = 0.0
        route_start = states_list[0].elapsed_months if states_list else 0.0

        for i in range(1, len(route)):
            total_distance += distances[route[i-1], route[i]]

            if states_list and i < len(states_list):
                season_gap = calendar.season_distance(
                    preferred_months[route[i]],
                    states_list[i].month_of_year
                )
                total_season_penalty += 8.0 * season_gap

                priority_delay = priorities[route[i]] * (states_list[i].elapsed_months - route_start)
                total_priority_delay += 6.0 * priority_delay

        total_distance += distances[route[-1], 0]

        if states_list:
            route_duration = states_list[-1].elapsed_months - route_start
            route_duration += distances[route[-1], 0] / calendar.speed_km_per_month
            time_penalty = 3.0 * route_duration
            total += total_distance + total_season_penalty + time_penalty + total_priority_delay
        else:
            total += total_distance

    return total

def check_route_time(route, distances, calendar):
    actual_time = 0.0
    current = 0
    for node in route[1:]:
        actual_time += distances[current, node] / calendar.speed_km_per_month
        actual_time += calendar.service_time_months
        current = node
    actual_time += distances[current, 0] / calendar.speed_km_per_month
    return actual_time