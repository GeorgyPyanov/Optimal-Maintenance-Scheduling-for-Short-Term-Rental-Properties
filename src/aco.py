import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class AntColony:
    def __init__(self, distances, priorities, preferred_months,
                 n_ants, n_iterations, decay, alpha=1, beta=2, Q=100,
                 start_month=1.0, speed_km_per_month=100.0,
                 service_time_months=0.1, season_penalty_weight=1.0):
        """
        distances: square matrix of distances (km)
        priorities: array of property priorities (higher = more urgent)
        preferred_months: array of optimal maintenance month (float 1-12) for each property
        n_ants: number of ants
        n_iterations: number of iterations
        decay: pheromone evaporation rate
        alpha: pheromone importance
        beta: heuristic importance
        Q: pheromone deposit factor
        start_month: starting month (float)
        speed_km_per_month: how many km can be covered in one month (for time progression)
        service_time_months: fixed time per service (months)
        season_penalty_weight: weight for season deviation in cost
        """
        self.distances = distances
        self.priorities = priorities
        self.preferred_months = preferred_months
        self.pheromones = np.ones(distances.shape) / len(distances)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.start_month = start_month
        self.speed_km_per_month = speed_km_per_month
        self.service_time_months = service_time_months
        self.season_penalty_weight = season_penalty_weight

    def run(self):
        shortest_path = None
        shortest_cost = float('inf')
        best_paths = []

        for iteration in range(self.n_iterations):
            paths, months_seq = self._construct_solutions()
            costs = [self._path_cost(path, months) for path, months in zip(paths, months_seq)]

            self._update_pheromones(paths, costs)

            best_idx = np.argmin(costs)
            if costs[best_idx] < shortest_cost:
                shortest_cost = costs[best_idx]
                shortest_path = paths[best_idx]
                best_paths.append((shortest_path, shortest_cost))

        return shortest_path, shortest_cost, best_paths

    def _construct_solutions(self):
        paths = []
        months_sequences = []
        for _ in range(self.n_ants):
            path, months = self._construct_path()
            paths.append(path)
            months_sequences.append(months)
        return paths, months_sequences

    def _construct_path(self):
        path = [0]  # Start from depot (index 0)
        current_month = self.start_month
        months = [current_month]
        remaining = set(range(1, self.distances.shape[0]))
        current = 0

        while remaining:
            next_city, new_month = self._select_next(current, remaining, current_month)
            path.append(next_city)
            months.append(new_month)
            remaining.remove(next_city)
            current = next_city
            current_month = new_month
        return path, months

    def _select_next(self, current, remaining, current_month):
        choices = list(remaining)
        probs = []

        for j in choices:
            d = self.distances[current][j] + 1e-6

            # Travel time from current to j (months)
            travel_time = d / self.speed_km_per_month
            # Arrival month
            arrival_month = current_month + travel_time + self.service_time_months

            # Circular month difference (preferred vs arrival)
            pref = self.preferred_months[j]
            diff = abs(pref - arrival_month)
            diff = min(diff, 12 - diff)  # circular distance

            # Heuristic: higher priority and smaller season diff = higher eta
            base_eta = 1 / d * self.priorities[j]  # priority multiplies directly
            season_factor = 1 / (1 + self.season_penalty_weight * diff)
            eta = base_eta * season_factor

            tau = self.pheromones[current][j]
            probs.append((tau ** self.alpha) * (eta ** self.beta))

        probs = np.array(probs)
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.ones(len(choices)) / len(choices)

        chosen = np.random.choice(choices, p=probs)

        # Compute the actual arrival month for the chosen property
        d_chosen = self.distances[current][chosen] + 1e-6
        travel_time = d_chosen / self.speed_km_per_month
        new_month = current_month + travel_time + self.service_time_months

        return chosen, new_month

    def _path_cost(self, path, months):
        """Total cost = total travel distance + season penalty."""
        total_distance = 0
        total_season_penalty = 0
        # months[0] is start month, months[1:] are arrival months for each visited property
        for idx in range(1, len(path)):
            prop = path[idx]
            pref = self.preferred_months[prop]
            arrival = months[idx]
            diff = abs(pref - arrival)
            diff = min(diff, 12 - diff)
            total_season_penalty += diff
        # Sum distances
        for i in range(len(path) - 1):
            total_distance += self.distances[path[i]][path[i + 1]]
        return total_distance + self.season_penalty_weight * total_season_penalty

    def _update_pheromones(self, paths, costs):
        self.pheromones *= self.decay
        for path, cost in zip(paths, costs):
            deposit = self.Q / (cost + 1e-6)
            for i in range(len(path) - 1):
                self.pheromones[path[i]][path[i + 1]] += deposit
                self.pheromones[path[i + 1]][path[i]] += deposit


def load_and_prepare_data(csv_path='result_dataset.csv', n_properties=50):
    """Load and prepare data: compute distances and normalize priorities."""
    df = pd.read_csv(csv_path)

    required_cols = ['id', 'latitude', 'longitude', 'neighbourhood',
                     'aco_priority', 'best_maintenance_month',
                     'price', 'complaint_ratio']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Column {col} not found in dataset")

    df = df[required_cols].dropna().reset_index(drop=True)

    # Take top N by priority
    df = df.nlargest(n_properties, 'aco_priority').reset_index(drop=True)

    n = len(df)

    # Compute geographic distances in km (Beijing area)
    coords = df[['latitude', 'longitude']].values
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            lat_dist = (coords[i, 0] - coords[j, 0]) * 111  # ~111 km per degree latitude
            lon_dist = (coords[i, 1] - coords[j, 1]) * 85  # ~85 km per degree longitude at Beijing
            dist_matrix[i, j] = np.sqrt(lat_dist ** 2 + lon_dist ** 2)

    # Normalize priorities for heuristic (0-1)
    priorities = df['aco_priority'].values
    scaler = MinMaxScaler()
    priorities_norm = scaler.fit_transform(priorities.reshape(-1, 1)).flatten()

    # Preferred months (float)
    preferred_months = df['best_maintenance_month'].values.astype(float)

    return dist_matrix, priorities_norm, preferred_months, df


def create_route_table(df, obj_indices, months_sequence):
    """
    Create a formatted table with visit order and planned months.

    Parameters:
    df: DataFrame with property data
    obj_indices: list of indices (from df) in visit order (excluding depot)
    months_sequence: list of planned months for each property (same length as obj_indices)
    """
    route_df = df.iloc[obj_indices].copy()
    route_df['visit_order'] = range(1, len(route_df) + 1)
    route_df['planned_month'] = months_sequence
    cols = ['visit_order', 'id', 'latitude', 'longitude', 'neighbourhood',
            'price', 'complaint_ratio', 'aco_priority',
            'best_maintenance_month', 'planned_month']
    return route_df[cols]


if __name__ == "__main__":
    # Load data
    distances, priorities, preferred_months, df = load_and_prepare_data('result_dataset.csv', n_properties=30)

    # Initialize ACO with real-time progression
    aco = AntColony(
        distances=distances,
        priorities=priorities,
        preferred_months=preferred_months,
        n_ants=50,
        n_iterations=200,
        decay=0.7,
        alpha=1.0,
        beta=3.0,
        Q=100,
        start_month=1.0,  # start in January
        speed_km_per_month=100.0,  # 100 km per month (roughly 3 km/day)
        service_time_months=0.1,  # ~3 days per service
        season_penalty_weight=2.0  # moderately penalize off-month visits
    )

    best_route, best_cost, _ = aco.run()

    # Simulate months for the best route
    current_month = aco.start_month
    months_seq = [current_month]  # month at depot
    for i in range(len(best_route) - 1):
        d = distances[best_route[i]][best_route[i + 1]]
        travel_time = d / aco.speed_km_per_month
        current_month = current_month + travel_time + aco.service_time_months
        months_seq.append(current_month)

    # Separate actual properties (exclude depot at index 0)
    obj_route = best_route[1:]  # property indices in visit order
    obj_months = months_seq[1:]  # corresponding planned months

    # Create and save route table
    route_table = create_route_table(df, obj_route, obj_months)
    route_table.to_csv('optimized_maintenance_route_real_time.csv', index=False)

    # Print summary
    print(f"Total properties: {len(df)}")
    print(f"Total cost (distance + season penalty): {best_cost:.2f}")
    print("Visit schedule")
    print("-" * 85)
    for _, row in route_table.iterrows():
        print(f"{int(row['visit_order']):3d} | ID {int(row['id']):10d} | "
              f"{str(row['neighbourhood']):12s} | priority {row['aco_priority']:.3f} | "
              f"best month {row['best_maintenance_month']:.1f} | planned month {row['planned_month']:.2f}")
    print(f"\nFile saved: optimized_maintenance_route_real_time.csv")