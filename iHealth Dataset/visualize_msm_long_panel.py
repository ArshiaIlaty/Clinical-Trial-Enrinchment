import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the long-format panel data
df = pd.read_csv('msm_long_panel_with_propensity.csv')

# ---
# Week calculation explanation:
# The WEEK variable is calculated for each patient as:
#   WEEK = ((MEASURMENT_TS - first MEASURMENT_TS for that patient) // 7 days) + 1
# This means WEEK 1 is the first week after a patient's initial BP measurement, WEEK 2 is the second week, etc.
# ---

# Restrict to first 52 weeks (1 year)
df_1yr = df[(df['WEEK'] >= 1) & (df['WEEK'] <= 52)]

# 1. Distribution of adherence over weeks (first year)
adherence_by_week = df_1yr.groupby('WEEK')['adherence'].mean()
plt.figure(figsize=(10,5))
plt.plot(adherence_by_week.index, adherence_by_week.values, marker='o')
plt.title('Mean Adherence by Week (First Year)')
plt.xlabel('Week')
plt.ylabel('Mean Adherence (Proportion with BP Reading)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('msm_adherence_by_week_1yr.png')
plt.show()

# 2. Number of patients per week (first year)
patients_by_week = df_1yr.groupby('WEEK')['HASHED_PATIENT_ID'].nunique()
plt.figure(figsize=(10,5))
plt.plot(patients_by_week.index, patients_by_week.values, marker='o', color='orange')
plt.title('Number of Patients per Week (First Year)')
plt.xlabel('Week')
plt.ylabel('Number of Patients')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('msm_patients_by_week_1yr.png')
plt.show()

# 3. Distribution of propensity scores (first year)
plt.figure(figsize=(8,5))
plt.hist(df_1yr['propensity'], bins=30, color='green', alpha=0.7, edgecolor='black')
plt.title('Distribution of Time-Varying Propensity Scores (First Year)')
plt.xlabel('Propensity Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('msm_propensity_hist_1yr.png')
plt.show()

# 4. Example: Adherence trajectories for a sample of patients (first year)
sample_pids = df_1yr['HASHED_PATIENT_ID'].drop_duplicates().sample(10, random_state=42)
plt.figure(figsize=(12,6))
for pid in sample_pids:
    patient = df_1yr[df_1yr['HASHED_PATIENT_ID'] == pid]
    plt.plot(patient['WEEK'], patient['adherence'], marker='o', label=str(pid)[:6])
plt.title('Adherence Trajectories for Sample Patients (First Year)')
plt.xlabel('Week')
plt.ylabel('Adherence (0/1)')
plt.grid(True, alpha=0.3)
plt.legend(title='Patient', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('msm_adherence_trajectories_1yr.png')
plt.show()

# ---
# Explanation: Difference between previous IPTW and stabilized IPTW weights
'''
# Previous IPTW (used in your earlier scripts):
#   - For each patient, weight = 1/propensity (treated) or 1/(1-propensity) (control)
#   - Used for a single time point (e.g., Q5 vs Q1 adherence at baseline)
#   - Does not account for time-varying exposures/confounders

# Stabilized IPTW for MSMs (to be used next):
#   - For each patient-time, weight = (probability of observed adherence given baseline covariates) / (probability of observed adherence given past adherence and time-varying confounders)
#   - Numerator: marginal probability (stabilizes the weights)
#   - Denominator: conditional probability (from time-varying propensity model)
#   - Allows for unbiased estimation of the effect of sustained adherence over time, even with time-varying confounding
''' 