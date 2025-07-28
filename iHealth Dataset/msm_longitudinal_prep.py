import pandas as pd
import numpy as np
from datetime import timedelta

# Load all relevant datasets
bp = pd.read_csv('bp_measurements.csv')
food = pd.read_csv('foodlogs.csv')
chat = pd.read_csv('chat_count_daily.csv')
visits = pd.read_csv('visits.csv')
demo = pd.read_csv('demographics.csv')

# Standardize columns and parse dates
bp['MEASURMENT_TS'] = pd.to_datetime(bp['MEASURMENT_TS'], errors='coerce')
food['COLLECTED_TS'] = pd.to_datetime(food['COLLECTED_TS'], errors='coerce')
chat['MESSAGE_DATE'] = pd.to_datetime(chat['MESSAGE_DATE'], errors='coerce')
visits['CHECKED_IN_TS'] = pd.to_datetime(visits['CHECKED_IN_TS'], errors='coerce')

bp['WEEK'] = bp.groupby('HASHED_PATIENT_ID')['MEASURMENT_TS'].transform(lambda x: ((x - x.min()).dt.days // 7) + 1)
food['WEEK'] = food.groupby('HASHED_PATIENT_ID')['COLLECTED_TS'].transform(lambda x: ((x - x.min()).dt.days // 7) + 1)
chat['WEEK'] = chat.groupby('HASHED_PATIENT_ID')['MESSAGE_DATE'].transform(lambda x: ((x - x.min()).dt.days // 7) + 1)
visits['WEEK'] = visits.groupby('HASHED_PATIENT_ID')['CHECKED_IN_TS'].transform(lambda x: ((x - x.min()).dt.days // 7) + 1)

# Get all patients
all_patients = set(bp['HASHED_PATIENT_ID']).union(food['HASHED_PATIENT_ID']).union(chat['HASHED_PATIENT_ID']).union(visits['HASHED_PATIENT_ID'])

rows = []
for pid in all_patients:
    # Find the first measurement date across all sources
    min_dates = []
    for df, col in [
        (bp, 'MEASURMENT_TS'),
        (food, 'COLLECTED_TS'),
        (chat, 'MESSAGE_DATE'),
        (visits, 'CHECKED_IN_TS')]:
        if pid in df['HASHED_PATIENT_ID'].values:
            min_dates.append(df[df['HASHED_PATIENT_ID'] == pid][col].min())
    if not min_dates:
        continue
    first_date = min(min_dates)
    # Generate weeks 1-52
    for week in range(1, 53):
        week_start = first_date + timedelta(days=(week-1)*7)
        week_end = week_start + timedelta(days=6)
        # BP readings
        bp_count = bp[(bp['HASHED_PATIENT_ID'] == pid) & (bp['MEASURMENT_TS'] >= week_start) & (bp['MEASURMENT_TS'] <= week_end)].shape[0]
        # Food logs
        food_count = food[(food['HASHED_PATIENT_ID'] == pid) & (food['COLLECTED_TS'] >= week_start) & (food['COLLECTED_TS'] <= week_end)].shape[0]
        # Chats
        chat_count = chat[(chat['HASHED_PATIENT_ID'] == pid) & (chat['MESSAGE_DATE'] >= week_start) & (chat['MESSAGE_DATE'] <= week_end)]['CHAT_COUNT'].sum()
        # Visits
        visit_count = visits[(visits['HASHED_PATIENT_ID'] == pid) & (visits['CHECKED_IN_TS'] >= week_start) & (visits['CHECKED_IN_TS'] <= week_end)].shape[0]
        # Composite adherence: sum of normalized (0/1) indicators for each metric
        metrics = [bp_count > 0, food_count > 0, chat_count > 0, visit_count > 0]
        composite = np.mean(metrics)
        rows.append({
            'HASHED_PATIENT_ID': pid,
            'WEEK': week,
            'bp_count': bp_count,
            'food_count': food_count,
            'chat_count': chat_count,
            'visit_count': visit_count,
            'composite_adherence': composite,
            'adherence': int(composite > 0)
        })

panel = pd.DataFrame(rows)

# Merge in demographics
panel = panel.merge(demo, on='HASHED_PATIENT_ID', how='left')

# Add previous adherence
panel = panel.sort_values(['HASHED_PATIENT_ID','WEEK'])
panel['prev_adherence'] = panel.groupby('HASHED_PATIENT_ID')['adherence'].shift(1).fillna(0)

# Save for MSM analysis
panel.to_csv('msm_long_panel_composite.csv', index=False)
print('Saved msm_long_panel_composite.csv with all patient-week combinations and composite adherence.') 