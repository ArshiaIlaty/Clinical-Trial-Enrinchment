import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=== COMPREHENSIVE ADHERENCE ANALYSIS ===\n")

# Load all datasets
print("Loading datasets...")
adherence_df = pd.read_csv('bp_adherence_scores.csv')
bp_df = pd.read_csv('bp_measurements.csv')
a1c_df = pd.read_csv('a1c_measurements.csv')
weight_df = pd.read_csv('weight_measurements.csv')
food_df = pd.read_csv('foodlogs.csv')
chat_df = pd.read_csv('chat_count_daily.csv')
visits_df = pd.read_csv('visits.csv')
demo_df = pd.read_csv('demographics.csv')

# Standardize column names
for df in [bp_df, a1c_df, weight_df, food_df, chat_df, visits_df, demo_df]:
    df.columns = [c.strip().upper() for c in df.columns]

# Parse timestamps
for df, ts_col in [(bp_df, 'MEASURMENT_TS'), (a1c_df, 'COLLECTED_DATE'), 
                   (weight_df, 'MEASURMENT_TS'), (food_df, 'COLLECTED_TS'),
                   (chat_df, 'MESSAGE_DATE'), (visits_df, 'CHECKED_IN_TS')]:
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')

print("Datasets loaded successfully!")

# ============================================================================
# PART 1: CLINICAL OUTCOMES BY QUINTILE
# ============================================================================

print("\n1. ANALYZING CLINICAL OUTCOMES BY QUINTILE...")

def calculate_clinical_changes(df, value_col, ts_col, patient_col, time_window_days=180):
    """Calculate baseline and follow-up values for clinical outcomes"""
    changes = []
    
    for pid in df[patient_col].unique():
        patient_data = df[df[patient_col] == pid].sort_values(ts_col)
        if len(patient_data) < 2:
            continue
            
        baseline = patient_data.iloc[0]
        baseline_date = baseline[ts_col]
        
        # Find follow-up measurements within time window
        follow_up_data = patient_data[
            (patient_data[ts_col] > baseline_date) & 
            (patient_data[ts_col] <= baseline_date + timedelta(days=time_window_days))
        ]
        
        if len(follow_up_data) > 0:
            # Use the latest measurement in the window
            follow_up = follow_up_data.iloc[-1]
            change = follow_up[value_col] - baseline[value_col]
            changes.append({
                'HASHED_PATIENT_ID': pid,
                'BASELINE_VALUE': baseline[value_col],
                'FOLLOW_UP_VALUE': follow_up[value_col],
                'CHANGE': change,
                'DAYS_BETWEEN': (follow_up[ts_col] - baseline[ts_col]).days
            })
    
    return pd.DataFrame(changes)

# Calculate BP changes
print("  Calculating BP changes...")
bp_changes = calculate_clinical_changes(bp_df, 'SYSTOLIC_BLOOD_PRESSURE', 'MEASURMENT_TS', 'HASHED_PATIENT_ID')
bp_changes = bp_changes.merge(adherence_df[['HASHED_PATIENT_ID', 'ADHERENCE_QUINTILE']], on='HASHED_PATIENT_ID')

# Calculate A1C changes
print("  Calculating A1C changes...")
a1c_changes = calculate_clinical_changes(a1c_df, 'A1C_VALUE', 'COLLECTED_DATE', 'HASHED_PATIENT_ID')
a1c_changes = a1c_changes.merge(adherence_df[['HASHED_PATIENT_ID', 'ADHERENCE_QUINTILE']], on='HASHED_PATIENT_ID')

# Calculate weight changes
print("  Calculating weight changes...")
weight_changes = calculate_clinical_changes(weight_df, 'BODY_WEIGHT_VALUE', 'MEASURMENT_TS', 'HASHED_PATIENT_ID')
weight_changes = weight_changes.merge(adherence_df[['HASHED_PATIENT_ID', 'ADHERENCE_QUINTILE']], on='HASHED_PATIENT_ID')

# ============================================================================
# PART 2: ENGAGEMENT METRICS CORRELATIONS
# ============================================================================

print("\n2. INVESTIGATING ENGAGEMENT METRICS CORRELATIONS...")

def calculate_engagement_metrics(df, patient_col, ts_col, metric_name, time_window_days=180):
    """Calculate engagement metrics for each patient"""
    metrics = []
    
    for pid in df[patient_col].unique():
        patient_data = df[df[patient_col] == pid]
        
        if metric_name == 'food_logs':
            # Food log frequency and quality
            total_logs = len(patient_data)
            avg_rating = patient_data['RATING'].mean() if 'RATING' in patient_data.columns else np.nan
            reviewed_rate = patient_data['IS_REVIEWED'].mean() if 'IS_REVIEWED' in patient_data.columns else np.nan
            
            metrics.append({
                'HASHED_PATIENT_ID': pid,
                'FOOD_LOG_COUNT': total_logs,
                'FOOD_AVG_RATING': avg_rating,
                'FOOD_REVIEWED_RATE': reviewed_rate
            })
            
        elif metric_name == 'chat':
            # Chat engagement
            total_chats = patient_data['CHAT_COUNT'].sum()
            chat_days = len(patient_data)
            avg_daily_chats = total_chats / chat_days if chat_days > 0 else 0
            
            metrics.append({
                'HASHED_PATIENT_ID': pid,
                'TOTAL_CHATS': total_chats,
                'CHAT_DAYS': chat_days,
                'AVG_DAILY_CHATS': avg_daily_chats
            })
            
        elif metric_name == 'visits':
            # Visit adherence
            total_visits = len(patient_data)
            no_shows = patient_data['CHECKED_OUT_TS'].isnull().sum()
            no_show_rate = no_shows / total_visits if total_visits > 0 else 0
            
            metrics.append({
                'HASHED_PATIENT_ID': pid,
                'TOTAL_VISITS': total_visits,
                'NO_SHOWS': no_shows,
                'NO_SHOW_RATE': no_show_rate
            })
    
    return pd.DataFrame(metrics)

# Calculate engagement metrics
print("  Calculating food log metrics...")
food_metrics = calculate_engagement_metrics(food_df, 'HASHED_PATIENT_ID', 'COLLECTED_TS', 'food_logs')

print("  Calculating chat metrics...")
chat_metrics = calculate_engagement_metrics(chat_df, 'HASHED_PATIENT_ID', 'MESSAGE_DATE', 'chat')

print("  Calculating visit metrics...")
visit_metrics = calculate_engagement_metrics(visits_df, 'HASHED_PATIENT_ID', 'CHECKED_IN_TS', 'visits')

# Merge all engagement metrics with adherence scores
engagement_df = adherence_df.copy()
engagement_df = engagement_df.merge(food_metrics, on='HASHED_PATIENT_ID', how='left')
engagement_df = engagement_df.merge(chat_metrics, on='HASHED_PATIENT_ID', how='left')
engagement_df = engagement_df.merge(visit_metrics, on='HASHED_PATIENT_ID', how='left')

# ============================================================================
# PART 3: PATIENT JOURNEY VISUALIZATIONS
# ============================================================================

print("\n3. CREATING PATIENT JOURNEY VISUALIZATIONS...")

def create_patient_journey(patient_id, days=180):
    """Create a comprehensive patient journey visualization"""
    # Get patient data
    patient_bp = bp_df[bp_df['HASHED_PATIENT_ID'] == patient_id].sort_values('MEASURMENT_TS')
    patient_food = food_df[food_df['HASHED_PATIENT_ID'] == patient_id].sort_values('COLLECTED_TS')
    patient_chat = chat_df[chat_df['HASHED_PATIENT_ID'] == patient_id].sort_values('MESSAGE_DATE')
    
    if len(patient_bp) == 0:
        return None
    
    # Create timeline
    start_date = patient_bp['MEASURMENT_TS'].min()
    end_date = start_date + timedelta(days=days)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'Patient Journey: {patient_id[:8]}...', fontsize=16, fontweight='bold')
    
    # BP measurements over time
    ax1.plot(patient_bp['MEASURMENT_TS'], patient_bp['SYSTOLIC_BLOOD_PRESSURE'], 
             'o-', color='red', alpha=0.7, label='Systolic BP')
    ax1.plot(patient_bp['MEASURMENT_TS'], patient_bp['DIASTOLIC_BLOOD_PRESSURE'], 
             's-', color='blue', alpha=0.7, label='Diastolic BP')
    ax1.set_title('Blood Pressure Measurements')
    ax1.set_ylabel('BP (mmHg)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Food logs over time
    if len(patient_food) > 0:
        food_dates = patient_food['COLLECTED_TS']
        food_ratings = patient_food['RATING']
        ax2.scatter(food_dates, food_ratings, alpha=0.6, color='green', s=50)
        ax2.set_title('Food Log Ratings')
        ax2.set_ylabel('Rating (0-5)')
        ax2.set_ylim(0, 5)
        ax2.grid(True, alpha=0.3)
    
    # Chat activity over time
    if len(patient_chat) > 0:
        chat_dates = patient_chat['MESSAGE_DATE']
        chat_counts = patient_chat['CHAT_COUNT']
        ax3.bar(chat_dates, chat_counts, alpha=0.7, color='purple')
        ax3.set_title('Daily Chat Activity')
        ax3.set_ylabel('Chat Count')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Create sample patient journeys for each quintile
print("  Creating sample patient journeys...")
sample_patients = []
for quintile in range(1, 6):
    quintile_patients = adherence_df[adherence_df['ADHERENCE_QUINTILE'] == quintile]['HASHED_PATIENT_ID'].head(3)
    sample_patients.extend(quintile_patients)

# ============================================================================
# PART 4: CREATE VISUALIZATIONS AND SAVE RESULTS
# ============================================================================

print("\n4. CREATING VISUALIZATIONS AND SAVING RESULTS...")

# 1. Clinical Outcomes by Quintile
fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
fig1.suptitle('Clinical Outcomes by Adherence Quintile', fontsize=16, fontweight='bold')

# BP changes
bp_quintile_means = bp_changes.groupby('ADHERENCE_QUINTILE')['CHANGE'].agg(['mean', 'std', 'count']).reset_index()
axes1[0, 0].bar(bp_quintile_means['ADHERENCE_QUINTILE'], bp_quintile_means['mean'], 
                yerr=bp_quintile_means['std'], capsize=5, alpha=0.7, color='red')
axes1[0, 0].set_title('Systolic BP Change by Quintile')
axes1[0, 0].set_xlabel('Adherence Quintile')
axes1[0, 0].set_ylabel('BP Change (mmHg)')
axes1[0, 0].grid(True, alpha=0.3)

# A1C changes
a1c_quintile_means = a1c_changes.groupby('ADHERENCE_QUINTILE')['CHANGE'].agg(['mean', 'std', 'count']).reset_index()
axes1[0, 1].bar(a1c_quintile_means['ADHERENCE_QUINTILE'], a1c_quintile_means['mean'], 
                yerr=a1c_quintile_means['std'], capsize=5, alpha=0.7, color='orange')
axes1[0, 1].set_title('A1C Change by Quintile')
axes1[0, 1].set_xlabel('Adherence Quintile')
axes1[0, 1].set_ylabel('A1C Change (%)')
axes1[0, 1].grid(True, alpha=0.3)

# Weight changes
weight_quintile_means = weight_changes.groupby('ADHERENCE_QUINTILE')['CHANGE'].agg(['mean', 'std', 'count']).reset_index()
axes1[1, 0].bar(weight_quintile_means['ADHERENCE_QUINTILE'], weight_quintile_means['mean'], 
                yerr=weight_quintile_means['std'], capsize=5, alpha=0.7, color='green')
axes1[1, 0].set_title('Weight Change by Quintile')
axes1[1, 0].set_xlabel('Adherence Quintile')
axes1[1, 0].set_ylabel('Weight Change (kg)')
axes1[1, 0].grid(True, alpha=0.3)

# Sample size by quintile
sample_sizes = pd.DataFrame({
    'BP': bp_quintile_means['count'],
    'A1C': a1c_quintile_means['count'],
    'Weight': weight_quintile_means['count']
}, index=bp_quintile_means['ADHERENCE_QUINTILE'])
sample_sizes.plot(kind='bar', ax=axes1[1, 1], alpha=0.7)
axes1[1, 1].set_title('Sample Sizes by Quintile')
axes1[1, 1].set_xlabel('Adherence Quintile')
axes1[1, 1].set_ylabel('Number of Patients')
axes1[1, 1].legend()
axes1[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('clinical_outcomes_by_quintile.png', dpi=300, bbox_inches='tight')

# 2. Engagement Metrics Correlations
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

# Visit adherence vs adherence
valid_visits = engagement_df.dropna(subset=['NO_SHOW_RATE'])
axes2[1, 0].scatter(valid_visits['ADHERENCE_SCORE'], valid_visits['NO_SHOW_RATE'], alpha=0.5, color='red')
axes2[1, 0].set_title('No-Show Rate vs Adherence Score')
axes2[1, 0].set_xlabel('Adherence Score')
axes2[1, 0].set_ylabel('No-Show Rate')
axes2[1, 0].grid(True, alpha=0.3)

# Engagement metrics by quintile
engagement_quintile = engagement_df.groupby('ADHERENCE_QUINTILE').agg({
    'FOOD_LOG_COUNT': 'mean',
    'AVG_DAILY_CHATS': 'mean',
    'NO_SHOW_RATE': 'mean'
}).reset_index()

engagement_quintile.plot(x='ADHERENCE_QUINTILE', y=['FOOD_LOG_COUNT', 'AVG_DAILY_CHATS'], 
                        kind='bar', ax=axes2[1, 1], alpha=0.7)
axes2[1, 1].set_title('Engagement Metrics by Quintile')
axes2[1, 1].set_xlabel('Adherence Quintile')
axes2[1, 1].set_ylabel('Average Value')
axes2[1, 1].legend()
axes2[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('engagement_correlations.png', dpi=300, bbox_inches='tight')

# 3. Patient Journey Examples
print("  Creating patient journey examples...")
for i, patient_id in enumerate(sample_patients[:5]):  # First 5 patients
    journey_fig = create_patient_journey(patient_id)
    if journey_fig:
        journey_fig.savefig(f'patient_journey_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.close(journey_fig)

# ============================================================================
# PART 5: SAVE SUMMARY STATISTICS
# ============================================================================

print("\n5. SAVING SUMMARY STATISTICS...")

# Clinical outcomes summary
clinical_summary = pd.DataFrame({
    'BP_Change_Mean': bp_quintile_means['mean'],
    'BP_Change_Std': bp_quintile_means['std'],
    'BP_Sample_Size': bp_quintile_means['count'],
    'A1C_Change_Mean': a1c_quintile_means['mean'],
    'A1C_Change_Std': a1c_quintile_means['std'],
    'A1C_Sample_Size': a1c_quintile_means['count'],
    'Weight_Change_Mean': weight_quintile_means['mean'],
    'Weight_Change_Std': weight_quintile_means['std'],
    'Weight_Sample_Size': weight_quintile_means['count']
}, index=bp_quintile_means['ADHERENCE_QUINTILE'])

clinical_summary.to_csv('clinical_outcomes_summary.csv')

# Engagement correlations
correlation_matrix = engagement_df[['ADHERENCE_SCORE', 'FOOD_LOG_COUNT', 'AVG_DAILY_CHATS', 'NO_SHOW_RATE']].corr()
correlation_matrix.to_csv('engagement_correlations_matrix.csv')

# Print key findings
print("\n=== KEY FINDINGS ===")
print(f"BP Changes Analysis:")
print(f"  - Patients analyzed: {len(bp_changes)}")
print(f"  - Mean BP change by quintile:")
for quintile in range(1, 6):
    mean_change = bp_changes[bp_changes['ADHERENCE_QUINTILE'] == quintile]['CHANGE'].mean()
    print(f"    Q{quintile}: {mean_change:.2f} mmHg")

print(f"\nA1C Changes Analysis:")
print(f"  - Patients analyzed: {len(a1c_changes)}")
print(f"  - Mean A1C change by quintile:")
for quintile in range(1, 6):
    mean_change = a1c_changes[a1c_changes['ADHERENCE_QUINTILE'] == quintile]['CHANGE'].mean()
    print(f"    Q{quintile}: {mean_change:.2f} %")

print(f"\nEngagement Correlations:")
print(f"  - Food logs vs adherence: {correlation_matrix.loc['ADHERENCE_SCORE', 'FOOD_LOG_COUNT']:.3f}")
print(f"  - Chat activity vs adherence: {correlation_matrix.loc['ADHERENCE_SCORE', 'AVG_DAILY_CHATS']:.3f}")
print(f"  - No-show rate vs adherence: {correlation_matrix.loc['ADHERENCE_SCORE', 'NO_SHOW_RATE']:.3f}")

print("\n=== ANALYSIS COMPLETE ===")
print("Files saved:")
print("- clinical_outcomes_by_quintile.png")
print("- engagement_correlations.png")
print("- patient_journey_*.png (5 examples)")
print("- clinical_outcomes_summary.csv")
print("- engagement_correlations_matrix.csv") 