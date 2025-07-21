import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=== QUICK ADHERENCE ANALYSIS ===\n")

# Load adherence scores
print("Loading adherence scores...")
adherence_df = pd.read_csv('bp_adherence_scores.csv')
print(f"Loaded {len(adherence_df)} patients with adherence scores")

# Load and sample large datasets for efficiency
print("Loading and sampling datasets...")

# Sample BP data (take 10% for efficiency)
bp_df = pd.read_csv('bp_measurements.csv', nrows=1000000)  # Sample 1M rows
bp_df.columns = [c.strip().upper() for c in bp_df.columns]
bp_df['MEASURMENT_TS'] = pd.to_datetime(bp_df['MEASURMENT_TS'], errors='coerce')
bp_df['SYSTOLIC_BLOOD_PRESSURE'] = pd.to_numeric(bp_df['SYSTOLIC_BLOOD_PRESSURE'], errors='coerce')
bp_df['DIASTOLIC_BLOOD_PRESSURE'] = pd.to_numeric(bp_df['DIASTOLIC_BLOOD_PRESSURE'], errors='coerce')

# Sample other datasets
a1c_df = pd.read_csv('a1c_measurements.csv')
a1c_df.columns = [c.strip().upper() for c in a1c_df.columns]
a1c_df['COLLECTED_DATE'] = pd.to_datetime(a1c_df['COLLECTED_DATE'], errors='coerce')
a1c_df['A1C_VALUE'] = pd.to_numeric(a1c_df['A1C_VALUE'], errors='coerce')

food_df = pd.read_csv('foodlogs.csv', nrows=500000)  # Sample 500K rows
food_df.columns = [c.strip().upper() for c in food_df.columns]
food_df['COLLECTED_TS'] = pd.to_datetime(food_df['COLLECTED_TS'], errors='coerce')
food_df['RATING'] = pd.to_numeric(food_df['RATING'], errors='coerce')

chat_df = pd.read_csv('chat_count_daily.csv', nrows=1000000)  # Sample 1M rows
chat_df.columns = [c.strip().upper() for c in chat_df.columns]
chat_df['MESSAGE_DATE'] = pd.to_datetime(chat_df['MESSAGE_DATE'], errors='coerce')
chat_df['CHAT_COUNT'] = pd.to_numeric(chat_df['CHAT_COUNT'], errors='coerce')

print("Datasets loaded successfully!")

# ============================================================================
# PART 1: CLINICAL OUTCOMES ANALYSIS (Simplified)
# ============================================================================

print("\n1. ANALYZING CLINICAL OUTCOMES...")

# Calculate BP changes (simplified approach)
print("  Calculating BP changes...")
bp_patients = bp_df['HASHED_PATIENT_ID'].unique()
bp_changes = []

for pid in bp_patients[:1000]:  # Limit to first 1000 patients for speed
    patient_data = bp_df[bp_df['HASHED_PATIENT_ID'] == pid].sort_values('MEASURMENT_TS')
    if len(patient_data) >= 2:
        baseline = patient_data.iloc[0]['SYSTOLIC_BLOOD_PRESSURE']
        follow_up = patient_data.iloc[-1]['SYSTOLIC_BLOOD_PRESSURE']
        change = follow_up - baseline
        bp_changes.append({
            'HASHED_PATIENT_ID': pid,
            'BP_CHANGE': change,
            'BASELINE_BP': baseline,
            'FOLLOW_UP_BP': follow_up
        })

bp_changes_df = pd.DataFrame(bp_changes)
bp_changes_df = bp_changes_df.merge(adherence_df[['HASHED_PATIENT_ID', 'ADHERENCE_QUINTILE']], 
                                   on='HASHED_PATIENT_ID', how='inner')

# Calculate A1C changes
print("  Calculating A1C changes...")
a1c_patients = a1c_df['HASHED_PATIENT_ID'].unique()
a1c_changes = []

for pid in a1c_patients[:500]:  # Limit to first 500 patients
    patient_data = a1c_df[a1c_df['HASHED_PATIENT_ID'] == pid].sort_values('COLLECTED_DATE')
    if len(patient_data) >= 2:
        baseline = patient_data.iloc[0]['A1C_VALUE']
        follow_up = patient_data.iloc[-1]['A1C_VALUE']
        change = follow_up - baseline
        a1c_changes.append({
            'HASHED_PATIENT_ID': pid,
            'A1C_CHANGE': change,
            'BASELINE_A1C': baseline,
            'FOLLOW_UP_A1C': follow_up
        })

a1c_changes_df = pd.DataFrame(a1c_changes)
a1c_changes_df = a1c_changes_df.merge(adherence_df[['HASHED_PATIENT_ID', 'ADHERENCE_QUINTILE']], 
                                     on='HASHED_PATIENT_ID', how='inner')

# ============================================================================
# PART 2: ENGAGEMENT METRICS (Simplified)
# ============================================================================

print("\n2. ANALYZING ENGAGEMENT METRICS...")

# Food log metrics
print("  Calculating food log metrics...")
food_metrics = food_df.groupby('HASHED_PATIENT_ID').agg({
    'RATING': ['count', 'mean'],
    'IS_REVIEWED': 'mean'
}).reset_index()
food_metrics.columns = ['HASHED_PATIENT_ID', 'FOOD_LOG_COUNT', 'FOOD_AVG_RATING', 'FOOD_REVIEWED_RATE']

# Chat metrics
print("  Calculating chat metrics...")
chat_metrics = chat_df.groupby('HASHED_PATIENT_ID').agg({
    'CHAT_COUNT': ['sum', 'mean']
}).reset_index()
chat_metrics.columns = ['HASHED_PATIENT_ID', 'TOTAL_CHATS', 'AVG_DAILY_CHATS']

# Merge engagement metrics with adherence
engagement_df = adherence_df.copy()
engagement_df = engagement_df.merge(food_metrics, on='HASHED_PATIENT_ID', how='left')
engagement_df = engagement_df.merge(chat_metrics, on='HASHED_PATIENT_ID', how='left')

# ============================================================================
# PART 3: CREATE VISUALIZATIONS
# ============================================================================

print("\n3. CREATING VISUALIZATIONS...")

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# 1. Clinical Outcomes by Quintile
fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
fig1.suptitle('Clinical Outcomes by Adherence Quintile', fontsize=16, fontweight='bold')

# BP changes by quintile
bp_quintile_means = bp_changes_df.groupby('ADHERENCE_QUINTILE')['BP_CHANGE'].agg(['mean', 'std', 'count']).reset_index()
axes1[0].bar(bp_quintile_means['ADHERENCE_QUINTILE'], bp_quintile_means['mean'], 
             yerr=bp_quintile_means['std'], capsize=5, alpha=0.7, color='red')
axes1[0].set_title('Systolic BP Change by Quintile')
axes1[0].set_xlabel('Adherence Quintile')
axes1[0].set_ylabel('BP Change (mmHg)')
axes1[0].grid(True, alpha=0.3)

# A1C changes by quintile
a1c_quintile_means = a1c_changes_df.groupby('ADHERENCE_QUINTILE')['A1C_CHANGE'].agg(['mean', 'std', 'count']).reset_index()
axes1[1].bar(a1c_quintile_means['ADHERENCE_QUINTILE'], a1c_quintile_means['mean'], 
             yerr=a1c_quintile_means['std'], capsize=5, alpha=0.7, color='orange')
axes1[1].set_title('A1C Change by Quintile')
axes1[1].set_xlabel('Adherence Quintile')
axes1[1].set_ylabel('A1C Change (%)')
axes1[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quick_clinical_outcomes.png', dpi=300, bbox_inches='tight')

# 2. Engagement Correlations
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
fig2.suptitle('Engagement Metrics vs Adherence Score', fontsize=16, fontweight='bold')

# Food logs vs adherence
valid_food = engagement_df.dropna(subset=['FOOD_LOG_COUNT'])
axes2[0, 0].scatter(valid_food['ADHERENCE_SCORE'], valid_food['FOOD_LOG_COUNT'], alpha=0.5, color='green')
axes2[0, 0].set_title('Food Log Count vs Adherence Score')
axes2[0, 0].set_xlabel('Adherence Score')
axes2[0, 0].set_ylabel('Food Log Count')
axes2[0, 0].grid(True, alpha=0.3)

# Chat activity vs adherence
valid_chat = engagement_df.dropna(subset=['AVG_DAILY_CHATS'])
axes2[0, 1].scatter(valid_chat['ADHERENCE_SCORE'], valid_chat['AVG_DAILY_CHATS'], alpha=0.5, color='purple')
axes2[0, 1].set_title('Average Daily Chats vs Adherence Score')
axes2[0, 1].set_xlabel('Adherence Score')
axes2[0, 1].set_ylabel('Avg Daily Chats')
axes2[0, 1].grid(True, alpha=0.3)

# Food rating vs adherence
valid_rating = engagement_df.dropna(subset=['FOOD_AVG_RATING'])
axes2[1, 0].scatter(valid_rating['ADHERENCE_SCORE'], valid_rating['FOOD_AVG_RATING'], alpha=0.5, color='blue')
axes2[1, 0].set_title('Food Log Rating vs Adherence Score')
axes2[1, 0].set_xlabel('Adherence Score')
axes2[1, 0].set_ylabel('Average Food Rating')
axes2[1, 0].grid(True, alpha=0.3)

# Engagement metrics by quintile
engagement_quintile = engagement_df.groupby('ADHERENCE_QUINTILE').agg({
    'FOOD_LOG_COUNT': 'mean',
    'AVG_DAILY_CHATS': 'mean',
    'FOOD_AVG_RATING': 'mean'
}).reset_index()

engagement_quintile.plot(x='ADHERENCE_QUINTILE', y=['FOOD_LOG_COUNT', 'AVG_DAILY_CHATS'], 
                        kind='bar', ax=axes2[1, 1], alpha=0.7)
axes2[1, 1].set_title('Engagement Metrics by Quintile')
axes2[1, 1].set_xlabel('Adherence Quintile')
axes2[1, 1].set_ylabel('Average Value')
axes2[1, 1].legend()
axes2[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quick_engagement_correlations.png', dpi=300, bbox_inches='tight')

# ============================================================================
# PART 4: SAVE SUMMARY STATISTICS
# ============================================================================

print("\n4. SAVING SUMMARY STATISTICS...")

# Clinical outcomes summary
clinical_summary = pd.DataFrame({
    'BP_Change_Mean': bp_quintile_means['mean'],
    'BP_Change_Std': bp_quintile_means['std'],
    'BP_Sample_Size': bp_quintile_means['count'],
    'A1C_Change_Mean': a1c_quintile_means['mean'],
    'A1C_Change_Std': a1c_quintile_means['std'],
    'A1C_Sample_Size': a1c_quintile_means['count']
}, index=bp_quintile_means['ADHERENCE_QUINTILE'])

clinical_summary.to_csv('quick_clinical_summary.csv')

# Engagement correlations
correlation_cols = ['ADHERENCE_SCORE', 'FOOD_LOG_COUNT', 'AVG_DAILY_CHATS', 'FOOD_AVG_RATING']
correlation_matrix = engagement_df[correlation_cols].corr()
correlation_matrix.to_csv('quick_engagement_correlations.csv')

# Print key findings
print("\n=== KEY FINDINGS ===")
print(f"BP Changes Analysis:")
print(f"  - Patients analyzed: {len(bp_changes_df)}")
print(f"  - Mean BP change by quintile:")
for quintile in range(1, 6):
    mean_change = bp_changes_df[bp_changes_df['ADHERENCE_QUINTILE'] == quintile]['BP_CHANGE'].mean()
    print(f"    Q{quintile}: {mean_change:.2f} mmHg")

print(f"\nA1C Changes Analysis:")
print(f"  - Patients analyzed: {len(a1c_changes_df)}")
print(f"  - Mean A1C change by quintile:")
for quintile in range(1, 6):
    mean_change = a1c_changes_df[a1c_changes_df['ADHERENCE_QUINTILE'] == quintile]['A1C_CHANGE'].mean()
    print(f"    Q{quintile}: {mean_change:.2f} %")

print(f"\nEngagement Correlations:")
print(f"  - Food logs vs adherence: {correlation_matrix.loc['ADHERENCE_SCORE', 'FOOD_LOG_COUNT']:.3f}")
print(f"  - Chat activity vs adherence: {correlation_matrix.loc['ADHERENCE_SCORE', 'AVG_DAILY_CHATS']:.3f}")
print(f"  - Food rating vs adherence: {correlation_matrix.loc['ADHERENCE_SCORE', 'FOOD_AVG_RATING']:.3f}")

print("\n=== QUICK ANALYSIS COMPLETE ===")
print("Files saved:")
print("- quick_clinical_outcomes.png")
print("- quick_engagement_correlations.png")
print("- quick_clinical_summary.csv")
print("- quick_engagement_correlations.csv") 