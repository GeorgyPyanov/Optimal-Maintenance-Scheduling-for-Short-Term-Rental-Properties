# Optimal Maintenance Scheduling for Short-Term Rental Properties

## About
This project develops and evaluates a route-planning approach for maintenance teams servicing short-term rental properties. It combines geospatial distance, urgency/complaint signals, pricing impact, and seasonal availability into a unified Ant Colony Optimization workflow, then produces optimized single-team or multi-team maintenance schedules with explicit time progression, seasonal constraints, and cost-based pheromone learning.

## Project Goal
- Minimize total maintenance routing cost.
- Prioritize high-impact properties (`aco_priority`).
- Respect seasonality (`best_maintenance_month`).

## How The Algorithm Works
Implementation: [src/aco.py](/C:/Users/Я/PycharmProjects/Optimal-Maintenance-Scheduling-for-Short-Term-Rental-Properties/src/aco.py)

1. Data preparation:
- Read `result_dataset.csv`.
- Select required columns (`id`, coordinates, priority, preferred month, etc.).
- Add a virtual depot (node `0`) at mean coordinates.
- Build distance matrix.
- Scale priorities and clip lower bound for numeric stability.

2. Time model (`MaintenanceCalendar`):
- Convert distance to travel time with `speed_km_per_month`.
- Add per-visit service time.
- Check operating window constraints.
- Optionally wait for preferred service month (`max_wait_months`).

3. Transition scoring:
- For each candidate node, compute a score from:
  - pheromone level;
  - inverse distance;
  - node priority;
  - seasonal mismatch penalty;
  - waiting penalty;
  - delay penalty for high-priority nodes.
- Convert scores to probabilities and sample the next node.

4. Route construction:
- `AntColony`: one route visits all nodes.
- `MultiAntColony`: multiple routes for multiple brigades/vehicles.

5. Cost function:
- Travel distance.
- Seasonal penalty.
- Route duration penalty.
- Initial idle-time penalty.
- Priority-delay penalty.
- Optional return-to-depot cost.

6. Pheromone update:
- Evaporate pheromones (`decay`).
- Deposit pheromones on better solutions (`Q / cost`).

## Data Description
Dataset: [src/result_dataset.csv](/C:/Users/Я/PycharmProjects/Optimal-Maintenance-Scheduling-for-Short-Term-Rental-Properties/src/result_dataset.csv)

Important fields:
- `id`: property id.
- `latitude`, `longitude`: coordinates.
- `price`: price level.
- `complaint_ratio`: complaint signal.
- `best_maintenance_month`: preferred maintenance month.
- `aco_priority`: final optimization priority.

## Tests
Сoverage for:
- calendar and season logic;
- data loading and validation;
- ACO route invariants;
- multi-route assignment invariants;
- dataset-level integration checks.
