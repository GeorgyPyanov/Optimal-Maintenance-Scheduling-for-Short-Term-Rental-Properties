# import numpy as np
# import pandas as pd
#
# def load_preprocessed_data():
#     df = pd.read_csv('../data/preprocessed/result_dataset.csv')
#
#     df = df[['id', 'latitude', 'longitude', 'aco_priority']].copy()
#     df = df.dropna(subset=['latitude', 'longitude', 'aco_priority']).reset_index(drop=True)
#
#     #for testing
#     df = df.head(30).copy()
#
#     n = len(df)
#
#     coordinates = df[['latitude', 'longitude']].values
#     dist_matrix = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             dist_matrix[i, j] = np.sqrt(
#                 (coordinates[i, 0] - coordinates[j, 0]) ** 2 +
#                 (coordinates[i, 1] - coordinates[j, 1]) ** 2
#             )
#
#     priorities = df['aco_priority'].values
#
#     return dist_matrix, priorities, df
#
# def ant_colony_optimization(dist_matrix, priorities,
#                             #can be modified since now it full of basic values
#                             n_ants=30,
#                             n_iterations=300,
#                             alpha=1.0,
#                             beta=2.0,
#                             evaporation_rate=0.5,
#                             Q=100):
#     n = dist_matrix.shape[0]
#     pheromone = np.ones((n, n)) * 0.1
#     best_path = None
#     best_length = float('inf')
#
#     for it in range(n_iterations):
#         for ant in range(n_ants):
#             visited = [False] * n
#             current = np.random.randint(n)
#             visited[current] = True
#             path = [current]
#             length = 0.0
#
#             for _ in range(n - 1):
#                 unvisited = [j for j in range(n) if not visited[j]]
#                 probs = []
#                 for j in unvisited:
#                     d = dist_matrix[current, j] + 1e-6
#                     eta = 1 / (d * priorities[j])
#                     tau = pheromone[current, j]
#                     probs.append((tau ** alpha) * (eta ** beta))
#
#                 probs = np.array(probs)
#                 probs /= probs.sum()
#                 next_city = np.random.choice(unvisited, p=probs)
#
#                 path.append(next_city)
#                 length += dist_matrix[current, next_city]
#                 visited[next_city] = True
#                 current = next_city
#
#             length += dist_matrix[path[-1], path[0]]
#
#             if length < best_length:
#                 best_length = length
#                 best_path = path[:]
#
#         pheromone *= evaporation_rate
#         deposit = Q / best_length if best_length > 0 else 0
#         for i in range(len(best_path) - 1):
#             pheromone[best_path[i], best_path[i + 1]] += deposit
#         pheromone[best_path[-1], best_path[0]] += deposit
#
#     return best_path, best_length
#
#
# if __name__ == "__main__":
#     dist_matrix, priorities, df = load_preprocessed_data()
#
#     p

# project/src/aco_seminar.py
# ТОЧНО структура семинара + адаптация под наш проект

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