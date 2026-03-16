# Optimal-Maintenance-Scheduling-for-Short-Term-Rental-Properties

## Data preprocessing

## Dataset Description: `result_dataset.csv`

This file contains merged and preprocessed data from listings, calendar, and reviews. All data is prepared specifically for the Ant Colony Optimization (ACO) algorithm.

### 1. Identification & Location

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| `id` | Unique property identifier | int | 12345 |
| `latitude` | Latitude coordinate | float | 39.9042 |
| `longitude` | Longitude coordinate | float | 116.4074 |
| `neighbourhood` | Beijing district (original Chinese name) | string | '朝阳区' |
| `neighbourhood_en` | Beijing district (English translation) | string | 'Chaoyang' |

### 2. Economic Indicators

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| `price` | Nightly price from listings (USD) | float | 150.0 |
| `avg_calendar_price` | Average price from calendar data (may differ due to discounts/surges) | float | 145.50 |

### 3. Usage Intensity (Proxy for Wear & Tear)

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| `reviews_per_month` | Average number of reviews per month. Higher = more guest turnover → more wear | float | 3.5 |
| `number_of_reviews` | Total number of reviews all time | int | 45 |

### 4. Availability (Maintenance Windows)

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| `availability_ratio` | Fraction of time property is available (0 = always booked, 1 = always empty) | float | 0.35 |
| `best_maintenance_month` | Month with maximum availability (1-12). Suggestion for scheduling | int | 2 |

### 5. Problems & Complaints (Urgency)

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| `total_reviews` | Total number of reviews for the property | int | 45 |
| `complaint_count` | Number of reviews containing complaints (keywords: broken, dirty, repair, etc.) | int | 8 |
| `complaint_ratio` | Complaint ratio = complaint_count / total_reviews | float | 0.18 |
| `urgency_score` | Normalized urgency (0-1). 1 = maximum complaints among all properties | float | 0.75 |

### 6. Weights & Final ACO Priority

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| `neighbourhood_weight` | District weight. Central/tourist districts have higher weight (Chaoyang = 1.3, Dongcheng = 1.5, etc.) | float | 1.3 |
| `aco_priority` | **Final priority for Ant Colony Algorithm**. Higher = more important to service. Formula: (wear + urgency) × availability × price × district weight | float | 0.87 |

---

## **How to Use in ACO**

In the ant transition formula:

P(k,ij) = [τij]^α · [ηij]^β / Σ

where ηij = 1/(dij · priority_j)


**`priority_j` = `aco_priority`** from this dataset.

---

## **Dataset Statistics**

- **Number of properties:** `{final_output.shape[0]}`
- **Number of columns:** `{final_output.shape[1]}`
- **Priority range:** from `{final_output['aco_priority'].min():.3f}` to `{final_output['aco_priority'].max():.3f}`

---

## **Example Row**

| Column | Value | Meaning |
|--------|-------|---------|
| id | 2818 | Property #2818 |
| neighbourhood | Chaoyang | Chaoyang district (central Beijing) |
| price | 199.0 | $199 per night |
| reviews_per_month | 4.2 | High turnover |
| availability_ratio | 0.28 | Only 28% available time |
| complaint_ratio | 0.32 | 32% of reviews have complaints |
| urgency_score | 0.91 | Very urgent! |
| best_maintenance_month | 2 | Best to service in February |
| aco_priority | 0.94 | Maximum priority |

**Interpretation:** Central property, expensive, highly demanded, many complaints, but few available windows. The ant should try to include it in the route at all costs!

---

## **Notes**

- English translation (`neighbourhood`) added for convenience
- All missing values are filled: properties without reviews have `complaint_ratio = 0`, properties without calendar data have `availability_ratio = 0`
