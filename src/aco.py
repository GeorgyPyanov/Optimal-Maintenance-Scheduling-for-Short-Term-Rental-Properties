import numpy as np
import pandas as pd


class AntColony:
    def __init__(self, distances, priorities, n_ants, n_iterations, decay, alpha=1, beta=1):
        self.distances = distances
        self.priorities = priorities
        self.pheromones = np.ones(distances.shape) / len(distances)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        shortest_distance = float('inf')

        for i in range(self.n_iterations):
            paths = self._construct_solutions()
            distances = [self._path_distance(path) for path in paths]


            self._update_pheromones(paths, distances)

            best_idx = np.argmin(distances)
            if distances[best_idx] < shortest_distance:
                shortest_distance = distances[best_idx]
                shortest_path = paths[best_idx]

        return shortest_path, shortest_distance

    def _construct_solutions(self):
        paths = []
        for _ in range(self.n_ants):
            path = self._construct_path()
            paths.append(path)
        return paths

    def _construct_path(self):
        path = [0]
        remaining = set(range(1, self.distances.shape[0]))
        current = 0

        while remaining:
            next_city = self._select_next(current, remaining)
            path.append(next_city)
            remaining.remove(next_city)
            current = next_city
        return path

    def _select_next(self, current, remaining):
        choices = list(remaining)
        probs = []

        for j in choices:
            d = self.distances[current][j] + 1e-6
            eta = 1 / (d * self.priorities[j])
            tau = self.pheromones[current][j]
            probs.append((tau ** self.alpha) * (eta ** self.beta))

        probs = np.array(probs)
        probs /= probs.sum()
        return np.random.choice(choices, p=probs)

    def _distance_heuristic(self, i, j):
        d = self.distances[i][j] + 1e-6
        return 1 / (d * self.priorities[j])

    def _path_distance(self, path):
        return sum(self.distances[path[i]][path[i + 1]] for i in range(len(path) - 1))

    def _update_pheromones(self, paths, distances):
        self.pheromones *= self.decay
        for path, dist in zip(paths, distances):
            deposit = 100 / dist if dist > 0 else 0  # Q=100
            for i in range(len(path) - 1):
                self.pheromones[path[i]][path[i + 1]] += deposit
                self.pheromones[path[i + 1]][path[i]] += deposit


def load_data():
    df = pd.read_csv('../data/preprocessed/result_dataset.csv')
    df = df[['id', 'latitude', 'longitude', 'aco_priority']].dropna().reset_index(drop=True)
    df = df.head(30).copy() #uses only first 30 rows for testing

    n = len(df)
    coordinates = df[['latitude', 'longitude']].values
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.sqrt((coordinates[i, 0] - coordinates[j, 0]) ** 2 + (coordinates[i, 1] - coordinates[j, 1]) ** 2)

    priorities = df['aco_priority'].values
    return dist_matrix, priorities, df


if __name__ == "__main__":
    distances, priorities, df = load_data()

    aco = AntColony(
        distances=distances,
        priorities=priorities,
        n_ants=30,
        n_iterations=300,
        decay=0.5,
        alpha=1.0,
        beta=2.0
    )

    best_route, length = aco.run()

    print(f"processed: {len(df)}")
    print("example aco_priority:", np.round(priorities[:5], 3))
    print("\nbest route")
    print("index:", best_route)
    print("overall distance:", round(length, 2))
    print("objects IDs:", df.iloc[best_route]['id'].astype(int).tolist())
    print("\nwell, it works. cool")