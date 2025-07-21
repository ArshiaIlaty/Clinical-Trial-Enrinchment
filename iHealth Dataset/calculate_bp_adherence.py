import pandas as pd
import numpy as np
from datetime import datetime

# Load BP measurements
df = pd.read_csv('bp_measurements.csv')

# Standardize column names (strip spaces, upper/lower)
df.columns = [c.strip().upper() for c in df.columns]

# Parse timestamps
df['MEASURMENT_TS'] = pd.to_datetime(df['MEASURMENT_TS'], errors='coerce')

# Remove rows with missing patient ID or timestamp
df = df.dropna(subset=['HASHED_PATIENT_ID', 'MEASURMENT_TS'])

# Group by patient
groups = df.groupby('HASHED_PATIENT_ID')

adherence_data = []

for pid, group in groups:
    group = group.sort_values('MEASURMENT_TS')
    if len(group) < 2:
        continue  # Need at least 2 readings for gap/variance
    # Total readings
    total_readings = len(group)
    # Weekly gap
    group['WEEK'] = group['MEASURMENT_TS'].dt.to_period('W')
    week_counts = group.groupby('WEEK').size()
    week_dates = group['MEASURMENT_TS'].dt.to_period('W').drop_duplicates().sort_values()
    week_gaps = week_dates[1:].astype(int) - week_dates[:-1].astype(int)
    avg_weekly_gap = week_gaps.mean() if len(week_gaps) > 0 else np.nan
    # Duration (weeks with readings)
    duration_weeks = week_counts.count()
    # Std of weekly readings
    std_weekly_readings = week_counts.std() if len(week_counts) > 1 else 0
    adherence_data.append({
        'HASHED_PATIENT_ID': pid,
        'TOTAL_READINGS': total_readings,
        'AVG_WEEKLY_GAP': avg_weekly_gap,
        'DURATION_WEEKS': duration_weeks,
        'STD_WEEKLY_READINGS': std_weekly_readings
    })

adherence_df = pd.DataFrame(adherence_data)

# Min-max normalization (handle NaN by filling with min)
def minmax(series):
    return (series - series.min()) / (series.max() - series.min()) if series.max() > series.min() else series*0

adherence_df['NORM_TOTAL_READINGS'] = minmax(adherence_df['TOTAL_READINGS'])
adherence_df['NORM_INV_AVG_GAP'] = minmax(1 / (adherence_df['AVG_WEEKLY_GAP'].replace(0, np.nan)))
adherence_df['NORM_DURATION'] = minmax(adherence_df['DURATION_WEEKS'])
adherence_df['NORM_INV_STD'] = minmax(1 / (adherence_df['STD_WEEKLY_READINGS'].replace(0, np.nan) + 1e-6))

# Composite adherence score (mean of normalized features)
adherence_df['ADHERENCE_SCORE'] = adherence_df[[
    'NORM_TOTAL_READINGS', 'NORM_INV_AVG_GAP', 'NORM_DURATION', 'NORM_INV_STD']].mean(axis=1)

# Assign quintiles (1=lowest, 5=highest)
adherence_df['ADHERENCE_QUINTILE'] = pd.qcut(adherence_df['ADHERENCE_SCORE'], 5, labels=[1,2,3,4,5])

# Output
adherence_df[['HASHED_PATIENT_ID', 'ADHERENCE_SCORE', 'ADHERENCE_QUINTILE']].to_csv('bp_adherence_scores.csv', index=False)

print('Saved bp_adherence_scores.csv with adherence scores and quintiles.') 