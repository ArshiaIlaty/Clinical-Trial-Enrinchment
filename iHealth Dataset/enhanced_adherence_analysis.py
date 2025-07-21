import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=== ENHANCED COMPREHENSIVE ADHERENCE ANALYSIS ===\n")

# Load adherence scores
print("Loading adherence scores...")
adherence_df = pd.read_csv('bp_adherence_scores.csv')
print(f"Loaded {len(adherence_df)} patients with BP adherence scores")

# Load and sample datasets
print("Loading and sampling datasets...")

# BP data (already processed)
bp_df = pd.read_csv('bp_measurements.csv', nrows=1000000)
bp_df.columns = [c.strip().upper() for c in bp_df.columns]
bp_df['MEASURMENT_TS'] = pd.to_datetime(bp_df['MEASURMENT_TS'], errors='coerce')
bp_df['SYSTOLIC_BLOOD_PRESSURE'] = pd.to_numeric(bp_df['SYSTOLIC_BLOOD_PRESSURE'], errors='coerce')

# A1C data
a1c_df = pd.read_csv('a1c_measurements.csv')
a1c_df.columns = [c.strip().upper() for c in a1c_df.columns]
a1c_df['COLLECTED_DATE'] = pd.to_datetime(a1c_df['COLLECTED_DATE'], errors='coerce')
a1c_df['A1C_VALUE'] = pd.to_numeric(a1c_df['A1C_VALUE'], errors='coerce')

# Food logs
food_df = pd.read_csv('foodlogs.csv', nrows=500000)
food_df.columns = [c.strip().upper() for c in food_df.columns]
food_df['COLLECTED_TS'] = pd.to_datetime(food_df['COLLECTED_TS'], errors='coerce')
food_df['RATING'] = pd.to_numeric(food_df['RATING'], errors='coerce')

# Chat data
chat_df = pd.read_csv('chat_count_daily.csv', nrows=1000000)
chat_df.columns = [c.strip().upper() for c in chat_df.columns]
chat_df['MESSAGE_DATE'] = pd.to_datetime(chat_df['MESSAGE_DATE'], errors='coerce')
chat_df['CHAT_COUNT'] = pd.to_numeric(chat_df['CHAT_COUNT'], errors='coerce')

# Visits data
visits_df = pd.read_csv('visits.csv')
visits_df.columns = [c.strip().upper() for c in visits_df.columns]
# Fix duplicate column names
visits_df.columns = ['HASHED_PATIENT_ID', 'VISIT_TYPE', 'VISIT_MODE', 'CHECKED_IN_TS', 'CHECKED_OUT_TS']
visits_df['CHECKED_IN_TS'] = pd.to_datetime(visits_df['CHECKED_IN_TS'], errors='coerce')
visits_df['CHECKED_OUT_TS'] = pd.to_datetime(visits_df['CHECKED_OUT_TS'], errors='coerce')

# Weight data
weight_df = pd.read_csv('weight_measurements.csv', nrows=500000)
weight_df.columns = [c.strip().upper() for c in weight_df.columns]
weight_df['MEASURMENT_TS'] = pd.to_datetime(weight_df['MEASURMENT_TS'], errors='coerce')
weight_df['BODY_WEIGHT_VALUE'] = pd.to_numeric(weight_df['BODY_WEIGHT_VALUE'], errors='coerce')

print("Datasets loaded successfully!")

# ============================================================================
# PART 1: CALCULATE ENHANCED ADHERENCE SCORES
# ============================================================================

print("\n1. CALCULATING ENHANCED ADHERENCE SCORES...")

def calculate_visit_adherence(df, patient_col='HASHED_PATIENT_ID'):
    """Calculate visit adherence metrics"""
    visit_metrics = []
    
    for pid in df[patient_col].unique():
        patient_visits = df[df[patient_col] == pid]
        if len(patient_visits) == 0:
            continue
            
        total_visits = len(patient_visits)
        no_shows = patient_visits['CHECKED_OUT_TS'].isnull().sum()
        no_show_rate = no_shows / total_visits if total_visits > 0 else 1.0
        
        # Visit frequency (visits per month)
        if len(patient_visits) > 1:
            first_visit = patient_visits['CHECKED_IN_TS'].min()
            last_visit = patient_visits['CHECKED_IN_TS'].max()
            days_between = (last_visit - first_visit).days
            visits_per_month = (total_visits * 30) / max(days_between, 1)
        else:
            visits_per_month = 0
        
        visit_metrics.append({
            'HASHED_PATIENT_ID': pid,
            'TOTAL_VISITS': total_visits,
            'NO_SHOW_RATE': no_show_rate,
            'VISITS_PER_MONTH': visits_per_month,
            'VISIT_ADHERENCE_SCORE': (1 - no_show_rate) * min(visits_per_month / 2, 1)  # Normalize to 0-1
        })
    
    return pd.DataFrame(visit_metrics)

def calculate_weight_adherence(df, patient_col='HASHED_PATIENT_ID'):
    """Calculate weight measurement adherence metrics"""
    weight_metrics = []
    
    for pid in df[patient_col].unique():
        patient_weight = df[df[patient_col] == pid].sort_values('MEASURMENT_TS')
        if len(patient_weight) < 2:
            continue
            
        total_measurements = len(patient_weight)
        
        # Weekly frequency
        patient_weight['WEEK'] = patient_weight['MEASURMENT_TS'].dt.to_period('W')
        weeks_with_measurements = patient_weight['WEEK'].nunique()
        
        # Consistency (inverse of standard deviation of weekly measurements)
        weekly_counts = patient_weight.groupby('WEEK').size()
        consistency = 1 / (weekly_counts.std() + 1) if len(weekly_counts) > 1 else 0
        
        # Duration (weeks with measurements)
        duration_score = min(weeks_with_measurements / 24, 1)  # Normalize to 6 months
        
        weight_metrics.append({
            'HASHED_PATIENT_ID': pid,
            'TOTAL_WEIGHT_MEASUREMENTS': total_measurements,
            'WEEKS_WITH_MEASUREMENTS': weeks_with_measurements,
            'WEIGHT_CONSISTENCY': consistency,
            'WEIGHT_DURATION': duration_score,
            'WEIGHT_ADHERENCE_SCORE': (consistency + duration_score) / 2
        })
    
    return pd.DataFrame(weight_metrics)

# Calculate enhanced adherence metrics
print("  Calculating visit adherence...")
visit_metrics = calculate_visit_adherence(visits_df)

print("  Calculating weight adherence...")
weight_metrics = calculate_weight_adherence(weight_df)

print("  Calculating food log metrics...")
food_metrics = food_df.groupby('HASHED_PATIENT_ID').agg({
    'RATING': ['count', 'mean'],
    'IS_REVIEWED': 'mean'
}).reset_index()
food_metrics.columns = ['HASHED_PATIENT_ID', 'FOOD_LOG_COUNT', 'FOOD_AVG_RATING', 'FOOD_REVIEWED_RATE']

print("  Calculating chat metrics...")
chat_metrics = chat_df.groupby('HASHED_PATIENT_ID').agg({
    'CHAT_COUNT': ['sum', 'mean']
}).reset_index()
chat_metrics.columns = ['HASHED_PATIENT_ID', 'TOTAL_CHATS', 'AVG_DAILY_CHATS']

# Merge all metrics
print("  Creating comprehensive adherence score...")
enhanced_df = adherence_df.copy()
enhanced_df = enhanced_df.merge(visit_metrics, on='HASHED_PATIENT_ID', how='left')
enhanced_df = enhanced_df.merge(weight_metrics, on='HASHED_PATIENT_ID', how='left')
enhanced_df = enhanced_df.merge(food_metrics, on='HASHED_PATIENT_ID', how='left')
enhanced_df = enhanced_df.merge(chat_metrics, on='HASHED_PATIENT_ID', how='left')

# Fill missing values with 0 for adherence scores
adherence_cols = ['VISIT_ADHERENCE_SCORE', 'WEIGHT_ADHERENCE_SCORE']
for col in adherence_cols:
    enhanced_df[col] = enhanced_df[col].fillna(0)

# Create comprehensive adherence score
enhanced_df['COMPREHENSIVE_ADHERENCE_SCORE'] = enhanced_df[[
    'ADHERENCE_SCORE',  # BP adherence (original)
    'VISIT_ADHERENCE_SCORE',
    'WEIGHT_ADHERENCE_SCORE'
]].mean(axis=1)

# Assign quintiles for comprehensive score
enhanced_df['COMPREHENSIVE_QUINTILE'] = pd.qcut(
    enhanced_df['COMPREHENSIVE_ADHERENCE_SCORE'], 
    5, 
    labels=[1, 2, 3, 4, 5]
)

# ============================================================================
# PART 2: CLINICAL OUTCOMES ANALYSIS
# ============================================================================

print("\n2. ANALYZING CLINICAL OUTCOMES...")

# Calculate BP changes
print("  Calculating BP changes...")
bp_patients = bp_df['HASHED_PATIENT_ID'].unique()
bp_changes = []

for pid in bp_patients[:1000]:
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
bp_changes_df = bp_changes_df.merge(enhanced_df[['HASHED_PATIENT_ID', 'COMPREHENSIVE_QUINTILE']], 
                                   on='HASHED_PATIENT_ID', how='inner')

# Calculate A1C changes
print("  Calculating A1C changes...")
a1c_patients = a1c_df['HASHED_PATIENT_ID'].unique()
a1c_changes = []

for pid in a1c_patients[:500]:
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
a1c_changes_df = a1c_changes_df.merge(enhanced_df[['HASHED_PATIENT_ID', 'COMPREHENSIVE_QUINTILE']], 
                                     on='HASHED_PATIENT_ID', how='inner')

# ============================================================================
# PART 3: CREATE VISUALIZATIONS
# ============================================================================

print("\n3. CREATING ENHANCED VISUALIZATIONS...")

plt.style.use('default')
sns.set_palette("husl")

# 1. Comprehensive Adherence Score Distribution
fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
fig1.suptitle('Enhanced Comprehensive Adherence Analysis', fontsize=16, fontweight='bold')

# Original vs Comprehensive adherence
axes1[0, 0].scatter(enhanced_df['ADHERENCE_SCORE'], enhanced_df['COMPREHENSIVE_ADHERENCE_SCORE'], 
                    alpha=0.5, color='blue')
axes1[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.7)
axes1[0, 0].set_title('Original vs Comprehensive Adherence Score')
axes1[0, 0].set_xlabel('Original BP Adherence Score')
axes1[0, 0].set_ylabel('Comprehensive Adherence Score')
axes1[0, 0].grid(True, alpha=0.3)

# Component scores by quintile
component_scores = enhanced_df.groupby('COMPREHENSIVE_QUINTILE')[
    ['ADHERENCE_SCORE', 'VISIT_ADHERENCE_SCORE', 'WEIGHT_ADHERENCE_SCORE']
].mean()

component_scores.plot(kind='bar', ax=axes1[0, 1], alpha=0.7)
axes1[0, 1].set_title('Component Scores by Comprehensive Quintile')
axes1[0, 1].set_xlabel('Comprehensive Adherence Quintile')
axes1[0, 1].set_ylabel('Mean Score')
axes1[0, 1].legend(['BP', 'Visits', 'Weight'])
axes1[0, 1].grid(True, alpha=0.3)

# Clinical outcomes by comprehensive quintile
bp_quintile_means = bp_changes_df.groupby('COMPREHENSIVE_QUINTILE')['BP_CHANGE'].agg(['mean', 'std', 'count']).reset_index()
axes1[1, 0].bar(bp_quintile_means['COMPREHENSIVE_QUINTILE'], bp_quintile_means['mean'], 
                yerr=bp_quintile_means['std'], capsize=5, alpha=0.7, color='red')
axes1[1, 0].set_title('BP Change by Comprehensive Quintile')
axes1[1, 0].set_xlabel('Comprehensive Adherence Quintile')
axes1[1, 0].set_ylabel('BP Change (mmHg)')
axes1[1, 0].grid(True, alpha=0.3)

# A1C changes by comprehensive quintile
a1c_quintile_means = a1c_changes_df.groupby('COMPREHENSIVE_QUINTILE')['A1C_CHANGE'].agg(['mean', 'std', 'count']).reset_index()
axes1[1, 1].bar(a1c_quintile_means['COMPREHENSIVE_QUINTILE'], a1c_quintile_means['mean'], 
                yerr=a1c_quintile_means['std'], capsize=5, alpha=0.7, color='orange')
axes1[1, 1].set_title('A1C Change by Comprehensive Quintile')
axes1[1, 1].set_xlabel('Comprehensive Adherence Quintile')
axes1[1, 1].set_ylabel('A1C Change (%)')
axes1[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('enhanced_adherence_analysis.png', dpi=300, bbox_inches='tight')

# 2. Engagement Metrics Correlations
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
fig2.suptitle('Enhanced Engagement Metrics vs Comprehensive Adherence', fontsize=16, fontweight='bold')

# Food logs vs comprehensive adherence
valid_food = enhanced_df.dropna(subset=['FOOD_LOG_COUNT'])
axes2[0, 0].scatter(valid_food['COMPREHENSIVE_ADHERENCE_SCORE'], valid_food['FOOD_LOG_COUNT'], 
                    alpha=0.5, color='green')
axes2[0, 0].set_title('Food Log Count vs Comprehensive Adherence')
axes2[0, 0].set_xlabel('Comprehensive Adherence Score')
axes2[0, 0].set_ylabel('Food Log Count')
axes2[0, 0].grid(True, alpha=0.3)

# Chat activity vs comprehensive adherence
valid_chat = enhanced_df.dropna(subset=['AVG_DAILY_CHATS'])
axes2[0, 1].scatter(valid_chat['COMPREHENSIVE_ADHERENCE_SCORE'], valid_chat['AVG_DAILY_CHATS'], 
                    alpha=0.5, color='purple')
axes2[0, 1].set_title('Average Daily Chats vs Comprehensive Adherence')
axes2[0, 1].set_xlabel('Comprehensive Adherence Score')
axes2[0, 1].set_ylabel('Avg Daily Chats')
axes2[0, 1].grid(True, alpha=0.3)

# Visit adherence vs comprehensive adherence
valid_visits = enhanced_df.dropna(subset=['VISIT_ADHERENCE_SCORE'])
axes2[1, 0].scatter(valid_visits['COMPREHENSIVE_ADHERENCE_SCORE'], valid_visits['VISIT_ADHERENCE_SCORE'], 
                    alpha=0.5, color='red')
axes2[1, 0].set_title('Visit Adherence vs Comprehensive Adherence')
axes2[1, 0].set_xlabel('Comprehensive Adherence Score')
axes2[1, 0].set_ylabel('Visit Adherence Score')
axes2[1, 0].grid(True, alpha=0.3)

# Weight adherence vs comprehensive adherence
valid_weight = enhanced_df.dropna(subset=['WEIGHT_ADHERENCE_SCORE'])
axes2[1, 1].scatter(valid_weight['COMPREHENSIVE_ADHERENCE_SCORE'], valid_weight['WEIGHT_ADHERENCE_SCORE'], 
                    alpha=0.5, color='brown')
axes2[1, 1].set_title('Weight Adherence vs Comprehensive Adherence')
axes2[1, 1].set_xlabel('Comprehensive Adherence Score')
axes2[1, 1].set_ylabel('Weight Adherence Score')
axes2[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('enhanced_engagement_correlations.png', dpi=300, bbox_inches='tight')

# ============================================================================
# PART 4: SAVE RESULTS
# ============================================================================

print("\n4. SAVING ENHANCED RESULTS...")

# Save enhanced adherence scores
enhanced_df.to_csv('enhanced_adherence_scores.csv', index=False)

# Clinical outcomes summary
clinical_summary = pd.DataFrame({
    'BP_Change_Mean': bp_quintile_means['mean'],
    'BP_Change_Std': bp_quintile_means['std'],
    'BP_Sample_Size': bp_quintile_means['count'],
    'A1C_Change_Mean': a1c_quintile_means['mean'],
    'A1C_Change_Std': a1c_quintile_means['std'],
    'A1C_Sample_Size': a1c_quintile_means['count']
}, index=bp_quintile_means['COMPREHENSIVE_QUINTILE'])

clinical_summary.to_csv('enhanced_clinical_summary.csv')

# Enhanced correlations
correlation_cols = ['COMPREHENSIVE_ADHERENCE_SCORE', 'ADHERENCE_SCORE', 'VISIT_ADHERENCE_SCORE', 
                   'WEIGHT_ADHERENCE_SCORE', 'FOOD_LOG_COUNT', 'AVG_DAILY_CHATS']
correlation_matrix = enhanced_df[correlation_cols].corr()
correlation_matrix.to_csv('enhanced_correlations.csv')

# Print key findings
print("\n=== ENHANCED KEY FINDINGS ===")
print(f"Enhanced Adherence Analysis:")
print(f"  - Patients with comprehensive scores: {len(enhanced_df)}")
print(f"  - Comprehensive score range: {enhanced_df['COMPREHENSIVE_ADHERENCE_SCORE'].min():.3f} - {enhanced_df['COMPREHENSIVE_ADHERENCE_SCORE'].max():.3f}")
print(f"  - Mean comprehensive score: {enhanced_df['COMPREHENSIVE_ADHERENCE_SCORE'].mean():.3f}")

print(f"\nBP Changes by Comprehensive Quintile:")
for quintile in range(1, 6):
    mean_change = bp_changes_df[bp_changes_df['COMPREHENSIVE_QUINTILE'] == quintile]['BP_CHANGE'].mean()
    print(f"  Q{quintile}: {mean_change:.2f} mmHg")

print(f"\nEnhanced Correlations:")
print(f"  - BP vs Comprehensive: {correlation_matrix.loc['ADHERENCE_SCORE', 'COMPREHENSIVE_ADHERENCE_SCORE']:.3f}")
print(f"  - Visits vs Comprehensive: {correlation_matrix.loc['VISIT_ADHERENCE_SCORE', 'COMPREHENSIVE_ADHERENCE_SCORE']:.3f}")
print(f"  - Weight vs Comprehensive: {correlation_matrix.loc['WEIGHT_ADHERENCE_SCORE', 'COMPREHENSIVE_ADHERENCE_SCORE']:.3f}")
print(f"  - Food logs vs Comprehensive: {correlation_matrix.loc['FOOD_LOG_COUNT', 'COMPREHENSIVE_ADHERENCE_SCORE']:.3f}")

print("\n=== ENHANCED ANALYSIS COMPLETE ===")
print("Files saved:")
print("- enhanced_adherence_scores.csv")
print("- enhanced_adherence_analysis.png")
print("- enhanced_engagement_correlations.png")
print("- enhanced_clinical_summary.csv")
print("- enhanced_correlations.csv") 