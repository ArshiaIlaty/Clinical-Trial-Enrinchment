import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=== DROPOUT ANALYSIS: ZERO ADHERENCE PATIENTS ===\n")

# Load adherence scores
print("Loading adherence scores...")
adherence_df = pd.read_csv('bp_adherence_scores.csv')
print(f"Loaded {len(adherence_df)} patients with adherence scores")

# Identify dropout vs engaged patients
zero_adherence = adherence_df[adherence_df['ADHERENCE_SCORE'] == 0]
engaged_patients = adherence_df[adherence_df['ADHERENCE_SCORE'] > 0]

print(f"Zero adherence patients (dropouts): {len(zero_adherence)} ({len(zero_adherence)/len(adherence_df)*100:.1f}%)")
print(f"Engaged patients: {len(engaged_patients)} ({len(engaged_patients)/len(adherence_df)*100:.1f}%)")

# Load datasets for analysis
print("\nLoading datasets for dropout analysis...")

# Demographics
demo_df = pd.read_csv('demographics.csv')
demo_df.columns = [c.strip().upper() for c in demo_df.columns]

# BP measurements
bp_df = pd.read_csv('bp_measurements.csv', nrows=1000000)
bp_df.columns = [c.strip().upper() for c in bp_df.columns]
bp_df['MEASURMENT_TS'] = pd.to_datetime(bp_df['MEASURMENT_TS'], errors='coerce')
bp_df['SYSTOLIC_BLOOD_PRESSURE'] = pd.to_numeric(bp_df['SYSTOLIC_BLOOD_PRESSURE'], errors='coerce')

# A1C measurements
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
visits_df.columns = ['HASHED_PATIENT_ID', 'VISIT_TYPE', 'VISIT_MODE', 'CHECKED_IN_TS', 'CHECKED_OUT_TS']
visits_df['CHECKED_IN_TS'] = pd.to_datetime(visits_df['CHECKED_IN_TS'], errors='coerce')
visits_df['CHECKED_OUT_TS'] = pd.to_datetime(visits_df['CHECKED_OUT_TS'], errors='coerce')

print("Datasets loaded successfully!")

# ============================================================================
# PART 1: DEMOGRAPHIC ANALYSIS
# ============================================================================

print("\n1. ANALYZING DEMOGRAPHIC DIFFERENCES...")

# Merge demographics with adherence groups
zero_demo = zero_adherence.merge(demo_df, on='HASHED_PATIENT_ID', how='left')
engaged_demo = engaged_patients.merge(demo_df, on='HASHED_PATIENT_ID', how='left')

# Demographic comparisons
demo_comparison = pd.DataFrame()

# Age group analysis
zero_age = zero_demo['CALCULATED_AGE_GROUP'].value_counts(normalize=True)
engaged_age = engaged_demo['CALCULATED_AGE_GROUP'].value_counts(normalize=True)
demo_comparison['ZERO_ADHERENCE_AGE'] = zero_age
demo_comparison['ENGAGED_AGE'] = engaged_age

# Gender analysis
zero_gender = zero_demo['GENDER_AT_BIRTH'].value_counts(normalize=True)
engaged_gender = engaged_demo['GENDER_AT_BIRTH'].value_counts(normalize=True)
demo_comparison['ZERO_ADHERENCE_GENDER'] = zero_gender
demo_comparison['ENGAGED_GENDER'] = engaged_gender

# BMI category analysis
zero_bmi = zero_demo['BMI_CATEGORY'].value_counts(normalize=True)
engaged_bmi = engaged_demo['BMI_CATEGORY'].value_counts(normalize=True)
demo_comparison['ZERO_ADHERENCE_BMI'] = zero_bmi
demo_comparison['ENGAGED_BMI'] = engaged_bmi

# Treatment type analysis
zero_treatment = zero_demo['ASSIGNED_TREATMENT'].value_counts(normalize=True)
engaged_treatment = engaged_demo['ASSIGNED_TREATMENT'].value_counts(normalize=True)
demo_comparison['ZERO_ADHERENCE_TREATMENT'] = zero_treatment
demo_comparison['ENGAGED_TREATMENT'] = engaged_treatment

# ============================================================================
# PART 2: CLINICAL OUTCOMES ANALYSIS
# ============================================================================

print("\n2. ANALYZING CLINICAL OUTCOMES...")

def analyze_clinical_outcomes(df, patient_ids, outcome_df, value_col, ts_col, patient_col):
    """Analyze clinical outcomes for a group of patients"""
    group_data = outcome_df[outcome_df[patient_col].isin(patient_ids)]
    if len(group_data) == 0:
        return None
    
    outcomes = []
    for pid in patient_ids[:500]:  # Limit for efficiency
        patient_data = group_data[group_data[patient_col] == pid].sort_values(ts_col)
        if len(patient_data) >= 2:
            baseline = patient_data.iloc[0][value_col]
            follow_up = patient_data.iloc[-1][value_col]
            change = follow_up - baseline
            outcomes.append({
                'PATIENT_ID': pid,
                'BASELINE': baseline,
                'FOLLOW_UP': follow_up,
                'CHANGE': change
            })
    
    return pd.DataFrame(outcomes) if outcomes else None

# BP outcomes comparison
print("  Analyzing BP outcomes...")
zero_bp_outcomes = analyze_clinical_outcomes(
    zero_demo, zero_demo['HASHED_PATIENT_ID'], bp_df, 
    'SYSTOLIC_BLOOD_PRESSURE', 'MEASURMENT_TS', 'HASHED_PATIENT_ID'
)
engaged_bp_outcomes = analyze_clinical_outcomes(
    engaged_demo, engaged_demo['HASHED_PATIENT_ID'], bp_df, 
    'SYSTOLIC_BLOOD_PRESSURE', 'MEASURMENT_TS', 'HASHED_PATIENT_ID'
)

# A1C outcomes comparison
print("  Analyzing A1C outcomes...")
zero_a1c_outcomes = analyze_clinical_outcomes(
    zero_demo, zero_demo['HASHED_PATIENT_ID'], a1c_df, 
    'A1C_VALUE', 'COLLECTED_DATE', 'HASHED_PATIENT_ID'
)
engaged_a1c_outcomes = analyze_clinical_outcomes(
    engaged_demo, engaged_demo['HASHED_PATIENT_ID'], a1c_df, 
    'A1C_VALUE', 'COLLECTED_DATE', 'HASHED_PATIENT_ID'
)

# ============================================================================
# PART 3: ENGAGEMENT PATTERN ANALYSIS
# ============================================================================

print("\n3. ANALYZING ENGAGEMENT PATTERNS...")

def analyze_engagement_patterns(df, patient_ids, engagement_df, metric_name):
    """Analyze engagement patterns for a group of patients"""
    group_data = engagement_df[engagement_df['HASHED_PATIENT_ID'].isin(patient_ids)]
    
    if metric_name == 'food':
        metrics = group_data.groupby('HASHED_PATIENT_ID').agg({
            'RATING': ['count', 'mean'],
            'IS_REVIEWED': 'mean'
        }).reset_index()
        metrics.columns = ['HASHED_PATIENT_ID', 'FOOD_COUNT', 'FOOD_RATING', 'FOOD_REVIEWED']
        
    elif metric_name == 'chat':
        metrics = group_data.groupby('HASHED_PATIENT_ID').agg({
            'CHAT_COUNT': ['sum', 'mean']
        }).reset_index()
        metrics.columns = ['HASHED_PATIENT_ID', 'TOTAL_CHATS', 'AVG_DAILY_CHATS']
        
    elif metric_name == 'visits':
        metrics = group_data.groupby('HASHED_PATIENT_ID').agg({
            'CHECKED_OUT_TS': lambda x: x.isnull().sum() / len(x) if len(x) > 0 else 1
        }).reset_index()
        metrics.columns = ['HASHED_PATIENT_ID', 'NO_SHOW_RATE']
        
    return metrics

# Food logging patterns
print("  Analyzing food logging patterns...")
zero_food = analyze_engagement_patterns(zero_demo, zero_demo['HASHED_PATIENT_ID'], food_df, 'food')
engaged_food = analyze_engagement_patterns(engaged_demo, engaged_demo['HASHED_PATIENT_ID'], food_df, 'food')

# Chat patterns
print("  Analyzing chat patterns...")
zero_chat = analyze_engagement_patterns(zero_demo, zero_demo['HASHED_PATIENT_ID'], chat_df, 'chat')
engaged_chat = analyze_engagement_patterns(engaged_demo, engaged_demo['HASHED_PATIENT_ID'], chat_df, 'chat')

# Visit patterns
print("  Analyzing visit patterns...")
zero_visits = analyze_engagement_patterns(zero_demo, zero_demo['HASHED_PATIENT_ID'], visits_df, 'visits')
engaged_visits = analyze_engagement_patterns(engaged_demo, engaged_demo['HASHED_PATIENT_ID'], visits_df, 'visits')

# ============================================================================
# PART 4: CREATE VISUALIZATIONS
# ============================================================================

print("\n4. CREATING DROPOUT ANALYSIS VISUALIZATIONS...")

plt.style.use('default')
sns.set_palette("husl")

# 1. Demographic Comparisons
fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
fig1.suptitle('Dropout vs Engaged Patients: Demographic Analysis', fontsize=16, fontweight='bold')

# Age group comparison
age_data = pd.DataFrame({
    'Zero Adherence': zero_age,
    'Engaged': engaged_age
}).fillna(0)
age_data.plot(kind='bar', ax=axes1[0, 0], alpha=0.7)
axes1[0, 0].set_title('Age Group Distribution')
axes1[0, 0].set_xlabel('Age Group')
axes1[0, 0].set_ylabel('Proportion')
axes1[0, 0].legend()
axes1[0, 0].grid(True, alpha=0.3)

# Gender comparison
gender_data = pd.DataFrame({
    'Zero Adherence': zero_gender,
    'Engaged': engaged_gender
}).fillna(0)
gender_data.plot(kind='bar', ax=axes1[0, 1], alpha=0.7)
axes1[0, 1].set_title('Gender Distribution')
axes1[0, 1].set_xlabel('Gender')
axes1[0, 1].set_ylabel('Proportion')
axes1[0, 1].legend()
axes1[0, 1].grid(True, alpha=0.3)

# BMI comparison
bmi_data = pd.DataFrame({
    'Zero Adherence': zero_bmi,
    'Engaged': engaged_bmi
}).fillna(0)
bmi_data.plot(kind='bar', ax=axes1[1, 0], alpha=0.7)
axes1[1, 0].set_title('BMI Category Distribution')
axes1[1, 0].set_xlabel('BMI Category')
axes1[1, 0].set_ylabel('Proportion')
axes1[1, 0].legend()
axes1[1, 0].grid(True, alpha=0.3)

# Treatment comparison
treatment_data = pd.DataFrame({
    'Zero Adherence': zero_treatment,
    'Engaged': engaged_treatment
}).fillna(0)
treatment_data.plot(kind='bar', ax=axes1[1, 1], alpha=0.7)
axes1[1, 1].set_title('Treatment Type Distribution')
axes1[1, 1].set_xlabel('Treatment Type')
axes1[1, 1].set_ylabel('Proportion')
axes1[1, 1].legend()
axes1[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dropout_demographics.png', dpi=300, bbox_inches='tight')

# 2. Clinical Outcomes Comparison
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
fig2.suptitle('Dropout vs Engaged Patients: Clinical Outcomes', fontsize=16, fontweight='bold')

# BP changes comparison
if zero_bp_outcomes is not None and engaged_bp_outcomes is not None:
    bp_data = [zero_bp_outcomes['CHANGE'], engaged_bp_outcomes['CHANGE']]
    axes2[0].boxplot(bp_data, labels=['Zero Adherence', 'Engaged'], patch_artist=True)
    axes2[0].set_title('BP Change Comparison')
    axes2[0].set_ylabel('BP Change (mmHg)')
    axes2[0].grid(True, alpha=0.3)

# A1C changes comparison
if zero_a1c_outcomes is not None and engaged_a1c_outcomes is not None:
    a1c_data = [zero_a1c_outcomes['CHANGE'], engaged_a1c_outcomes['CHANGE']]
    axes2[1].boxplot(a1c_data, labels=['Zero Adherence', 'Engaged'], patch_artist=True)
    axes2[1].set_title('A1C Change Comparison')
    axes2[1].set_ylabel('A1C Change (%)')
    axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dropout_clinical_outcomes.png', dpi=300, bbox_inches='tight')

# 3. Engagement Patterns Comparison
fig3, axes3 = plt.subplots(2, 2, figsize=(15, 12))
fig3.suptitle('Dropout vs Engaged Patients: Engagement Patterns', fontsize=16, fontweight='bold')

# Food logging comparison
if zero_food is not None and engaged_food is not None:
    food_data = [zero_food['FOOD_COUNT'], engaged_food['FOOD_COUNT']]
    axes3[0, 0].boxplot(food_data, labels=['Zero Adherence', 'Engaged'], patch_artist=True)
    axes3[0, 0].set_title('Food Log Count Comparison')
    axes3[0, 0].set_ylabel('Food Log Count')
    axes3[0, 0].grid(True, alpha=0.3)

# Chat activity comparison
if zero_chat is not None and engaged_chat is not None:
    chat_data = [zero_chat['AVG_DAILY_CHATS'], engaged_chat['AVG_DAILY_CHATS']]
    axes3[0, 1].boxplot(chat_data, labels=['Zero Adherence', 'Engaged'], patch_artist=True)
    axes3[0, 1].set_title('Average Daily Chats Comparison')
    axes3[0, 1].set_ylabel('Avg Daily Chats')
    axes3[0, 1].grid(True, alpha=0.3)

# Visit no-show comparison
if zero_visits is not None and engaged_visits is not None:
    visit_data = [zero_visits['NO_SHOW_RATE'], engaged_visits['NO_SHOW_RATE']]
    axes3[1, 0].boxplot(visit_data, labels=['Zero Adherence', 'Engaged'], patch_artist=True)
    axes3[1, 0].set_title('No-Show Rate Comparison')
    axes3[1, 0].set_ylabel('No-Show Rate')
    axes3[1, 0].grid(True, alpha=0.3)

# Food rating comparison
if zero_food is not None and engaged_food is not None:
    rating_data = [zero_food['FOOD_RATING'], engaged_food['FOOD_RATING']]
    axes3[1, 1].boxplot(rating_data, labels=['Zero Adherence', 'Engaged'], patch_artist=True)
    axes3[1, 1].set_title('Food Rating Comparison')
    axes3[1, 1].set_ylabel('Average Food Rating')
    axes3[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dropout_engagement_patterns.png', dpi=300, bbox_inches='tight')

# ============================================================================
# PART 5: SAVE RESULTS AND SUMMARY
# ============================================================================

print("\n5. SAVING DROPOUT ANALYSIS RESULTS...")

# Save demographic comparison
demo_comparison.to_csv('dropout_demographic_comparison.csv')

# Save clinical outcomes summary
clinical_summary = pd.DataFrame({
    'Group': ['Zero Adherence', 'Engaged'],
    'BP_Change_Mean': [
        zero_bp_outcomes['CHANGE'].mean() if zero_bp_outcomes is not None else np.nan,
        engaged_bp_outcomes['CHANGE'].mean() if engaged_bp_outcomes is not None else np.nan
    ],
    'BP_Change_Std': [
        zero_bp_outcomes['CHANGE'].std() if zero_bp_outcomes is not None else np.nan,
        engaged_bp_outcomes['CHANGE'].std() if engaged_bp_outcomes is not None else np.nan
    ],
    'A1C_Change_Mean': [
        zero_a1c_outcomes['CHANGE'].mean() if zero_a1c_outcomes is not None else np.nan,
        engaged_a1c_outcomes['CHANGE'].mean() if engaged_a1c_outcomes is not None else np.nan
    ],
    'A1C_Change_Std': [
        zero_a1c_outcomes['CHANGE'].std() if zero_a1c_outcomes is not None else np.nan,
        engaged_a1c_outcomes['CHANGE'].std() if engaged_a1c_outcomes is not None else np.nan
    ]
})
clinical_summary.to_csv('dropout_clinical_summary.csv', index=False)

# Print key findings
print("\n=== DROPOUT ANALYSIS KEY FINDINGS ===")
print(f"Dropout Rate: {len(zero_adherence)/len(adherence_df)*100:.1f}% ({len(zero_adherence)} patients)")
print(f"Engagement Rate: {len(engaged_patients)/len(adherence_df)*100:.1f}% ({len(engaged_patients)} patients)")

print(f"\nDemographic Differences:")
print(f"  - Age groups with highest dropout: {zero_age.index[0] if len(zero_age) > 0 else 'N/A'}")
print(f"  - Gender with highest dropout: {zero_gender.index[0] if len(zero_gender) > 0 else 'N/A'}")
print(f"  - BMI category with highest dropout: {zero_bmi.index[0] if len(zero_bmi) > 0 else 'N/A'}")
print(f"  - Treatment type with highest dropout: {zero_treatment.index[0] if len(zero_treatment) > 0 else 'N/A'}")

if zero_bp_outcomes is not None and engaged_bp_outcomes is not None:
    print(f"\nClinical Outcomes:")
    print(f"  - Zero adherence BP change: {zero_bp_outcomes['CHANGE'].mean():.2f} mmHg")
    print(f"  - Engaged BP change: {engaged_bp_outcomes['CHANGE'].mean():.2f} mmHg")
    print(f"  - BP difference: {engaged_bp_outcomes['CHANGE'].mean() - zero_bp_outcomes['CHANGE'].mean():.2f} mmHg")

if zero_food is not None and engaged_food is not None:
    print(f"\nEngagement Patterns:")
    print(f"  - Zero adherence food logs: {zero_food['FOOD_COUNT'].mean():.1f}")
    print(f"  - Engaged food logs: {engaged_food['FOOD_COUNT'].mean():.1f}")
    print(f"  - Food log difference: {engaged_food['FOOD_COUNT'].mean() - zero_food['FOOD_COUNT'].mean():.1f}")

print("\n=== DROPOUT ANALYSIS COMPLETE ===")
print("Files saved:")
print("- dropout_demographics.png")
print("- dropout_clinical_outcomes.png")
print("- dropout_engagement_patterns.png")
print("- dropout_demographic_comparison.csv")
print("- dropout_clinical_summary.csv") 