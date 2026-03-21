import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load datasets
listings = pd.read_csv('../raw/dataset/listings_clean.csv')
calendar = pd.read_csv('../raw/dataset/calendar_detail.csv')
reviews = pd.read_csv('../raw/dataset/reviews_detail.csv')

print(f"Listings: {listings.shape}")
print(f"Calendar: {calendar.shape}")
print(f"Reviews: {reviews.shape}")

# Clean listings
base_columns = [
    'id',
    'latitude',
    'longitude',
    'neighbourhood',
    'price',
    'reviews_per_month',
    'number_of_reviews'
]

available_columns = [col for col in base_columns if col in listings.columns]
listings_clean = listings[available_columns].copy()

# Clean price column
listings_clean['price'] = listings_clean['price'].astype(str).str.replace('$', '', regex=False)
listings_clean['price'] = listings_clean['price'].str.replace(',', '', regex=False)
listings_clean['price'] = pd.to_numeric(listings_clean['price'], errors='coerce')

# Fill missing review metrics
listings_clean['reviews_per_month'] = listings_clean['reviews_per_month'].fillna(0)
listings_clean['number_of_reviews'] = listings_clean['number_of_reviews'].fillna(0)

# ------------------- Process calendar -------------------
calendar['date'] = pd.to_datetime(calendar['date'])
calendar['available'] = calendar['available'] == 't'
calendar['month'] = calendar['date'].dt.month

calendar['price'] = calendar['price'].astype(str).str.replace('$', '', regex=False)
calendar['price'] = calendar['price'].str.replace(',', '', regex=False)
calendar['price'] = pd.to_numeric(calendar['price'], errors='coerce')

# Aggregate calendar data per listing
calendar_agg = calendar.groupby('listing_id').agg({
    'available': 'mean',
    'price': 'mean',
}).reset_index()
calendar_agg.columns = ['listing_id', 'availability_ratio', 'avg_calendar_price']

# Find month with highest availability (best maintenance month)
best_month = calendar.groupby('listing_id').apply(
    lambda x: x.loc[x['available'].idxmax(), 'month'] if x['available'].any() else None
).reset_index()
best_month.columns = ['listing_id', 'best_maintenance_month']

calendar_agg = calendar_agg.merge(best_month, on='listing_id', how='left')

# Process reviews
issue_keywords = ['broken', 'not working', 'repair', 'fix', 'noisy', 'dirty', 'mold', 'leak', 'bad']

def count_issues(comments):
    """Count how many issue keywords appear in review text"""
    if pd.isna(comments):
        return 0
    comments = str(comments).lower()
    return sum(1 for word in issue_keywords if word in comments)

reviews_agg = reviews.groupby('listing_id').agg({
    'id': 'count',
    'comments': lambda x: sum(count_issues(comment) for comment in x)
}).reset_index()
reviews_agg.columns = ['listing_id', 'total_reviews', 'complaint_count']

reviews_agg['complaint_ratio'] = reviews_agg['complaint_count'] / reviews_agg['total_reviews'].clip(lower=1)

# Normalize complaint ratio to urgency score
scaler = MinMaxScaler()
reviews_agg['urgency_score'] = scaler.fit_transform(reviews_agg[['complaint_ratio']])

# Merge all data
final_df = listings_clean.copy()
final_df = final_df.merge(calendar_agg, left_on='id', right_on='listing_id', how='left')
final_df = final_df.merge(reviews_agg, left_on='id', right_on='listing_id', how='left')
final_df = final_df.drop(columns=['listing_id_x', 'listing_id_y'], errors='ignore')

# Fill missing values
final_df['availability_ratio'] = final_df['availability_ratio'].fillna(0)
final_df['avg_calendar_price'] = final_df['avg_calendar_price'].fillna(final_df['price'])
final_df['best_maintenance_month'] = final_df['best_maintenance_month'].fillna(6)  # Default to June
final_df['total_reviews'] = final_df['total_reviews'].fillna(0)
final_df['complaint_count'] = final_df['complaint_count'].fillna(0)
final_df['complaint_ratio'] = final_df['complaint_ratio'].fillna(0)
final_df['urgency_score'] = final_df['urgency_score'].fillna(0)

# Normalize selected features
features_to_normalize = ['reviews_per_month', 'urgency_score', 'availability_ratio', 'price']
scaler = MinMaxScaler()

for col in features_to_normalize:
    if col in final_df.columns:
        final_df[f'{col}_norm'] = scaler.fit_transform(final_df[[col]].fillna(0))

# Calculate ACO Priority
# Priority = (wear + urgency) * availability * price * neighborhood weight

# Neighborhood importance weights (central/tourist areas get higher priority)
neighbourhood_importance = {
    'Dongcheng': 1.5, 'Xicheng': 1.5, 'Chaoyang': 1.3, 'Haidian': 1.2,
    'Fengtai': 1.0, 'Shijingshan': 1.0, 'Tongzhou': 0.9, 'Shunyi': 0.8
}
final_df['neighbourhood_weight'] = final_df['neighbourhood'].map(neighbourhood_importance).fillna(1.0)

# Calculate priority score
final_df['aco_priority'] = (
    (final_df['reviews_per_month_norm'] * 0.3 +      # Wear from usage frequency
     final_df['urgency_score_norm'] * 0.7)           # Urgency from complaints (higher weight)
    * (1 + final_df['availability_ratio_norm'])      # Availability bonus
    * (1 + final_df['price_norm'])                   # Price bonus (higher price = higher priority)
    * final_df['neighbourhood_weight']               # Neighborhood importance
)

final_df = final_df.sort_values('aco_priority', ascending=False)

# Prepare output
output_columns = [
    'id', 'latitude', 'longitude',
    'neighbourhood', 'neighbourhood_group' if 'neighbourhood_group' in final_df.columns else None,
    'price', 'reviews_per_month', 'number_of_reviews',
    'availability_ratio', 'avg_calendar_price', 'best_maintenance_month',
    'total_reviews', 'complaint_count', 'complaint_ratio', 'urgency_score',
    'neighbourhood_weight', 'aco_priority'
]

output_columns = [col for col in output_columns if col is not None and col in final_df.columns]
final_output = final_df[output_columns].copy()

# Translate neighborhoods to English
neighbourhood_translation = {
    '朝阳区': 'Chaoyang',
    '东城区': 'Dongcheng',
    '西城区': 'Xicheng',
    '海淀区': 'Haidian',
    '丰台区': 'Fengtai',
    '通州区': 'Tongzhou',
    '顺义区': 'Shunyi',
    '昌平区': 'Changping',
    '大兴区': 'Daxing',
    '石景山区': 'Shijingshan',
    '房山区': 'Fangshan',
    '门头沟区': 'Mentougou',
    '怀柔区': 'Huairou',
    '平谷区': 'Pinggu',
    '密云区': 'Miyun',
    '延庆区': 'Yanqing'
}

final_output['neighbourhood'] = final_output['neighbourhood'].map(neighbourhood_translation)
final_output.to_csv('result_dataset.csv', index=False)