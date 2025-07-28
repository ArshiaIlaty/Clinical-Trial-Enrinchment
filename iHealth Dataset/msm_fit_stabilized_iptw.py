import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.cov_struct import Exchangeable

# Load long-format panel data (weeks 1-52 only)
df = pd.read_csv('msm_long_panel_with_propensity.csv')
df = df[(df['WEEK'] >= 1) & (df['WEEK'] <= 52)]

# Drop rows with missing required data
df = df.dropna(subset=['adherence','GENDER_AT_BIRTH_M','ASSIGNED_TREATMENT_BP','BMI','mean_bp'])

# Standardize BMI
scaler = StandardScaler()
BMI_scaled = scaler.fit_transform(df[['BMI']])
df['BMI_SCALED'] = BMI_scaled.flatten()

# ---
# For each week, fit propensity models only if both classes are present
# If only one class (all 0 or all 1), set stabilized_weight=1 for that week

def assign_weights_per_week(df):
    df = df.copy()
    df['stabilized_weight'] = np.nan
    for week, sub in df.groupby('WEEK'):
        if sub['adherence'].nunique() == 1:
            # Only one class, set weight=1
            df.loc[sub.index, 'stabilized_weight'] = 1.0
        else:
            # Fit numerator model: P(adherence | baseline covariates)
            X_num = sub[['GENDER_AT_BIRTH_M','ASSIGNED_TREATMENT_BP','BMI_SCALED']]
            y = sub['adherence']
            ps_num = LogisticRegression(max_iter=1000)
            ps_num.fit(X_num, y)
            ps_num_pred = ps_num.predict_proba(X_num)[:,1]
            # Fit denominator model: P(adherence | baseline covariates + prev_adherence)
            X_den = X_num.copy()
            X_den['prev_adherence'] = sub['prev_adherence']
            ps_den = LogisticRegression(max_iter=1000)
            ps_den.fit(X_den, y)
            ps_den_pred = ps_den.predict_proba(X_den)[:,1]
            # Calculate stabilized weights
            weights = ps_num_pred / ps_den_pred
            weights = np.clip(weights, 0, 10)
            df.loc[sub.index, 'stabilized_weight'] = weights
    return df

# Diagnostic: write adherence class counts per week to a file
with open('msm_adherence_by_week_debug.txt', 'w') as f:
    for week, sub in df.groupby('WEEK'):
        f.write(f'Week {week}: {sub["adherence"].value_counts().to_dict()}\n')

df = assign_weights_per_week(df)

print('Stabilized IPTW weights calculated. Mean:', df['stabilized_weight'].mean(), 'Std:', df['stabilized_weight'].std())

# ---
# 4. MSM: Estimate effect of adherence on BP (mean_bp) using GEE with stabilized weights
#   - Outcome: mean_bp (per week)
#   - Exposure: adherence (per week)
#   - Cluster: patient
#   - Weights: stabilized IPTW

df = df.sort_values(['HASHED_PATIENT_ID','WEEK'])

# MSM formula: mean_bp ~ adherence
model = GEE(
    endog=df['mean_bp'],
    exog=sm.add_constant(df['adherence']),
    groups=df['HASHED_PATIENT_ID'],
    family=Gaussian(),
    cov_struct=Exchangeable(),
    weights=df['stabilized_weight']
)
result = model.fit()

print('\n=== MSM (GEE) Weighted Regression Results ===')
print(result.summary())
print(f'Estimated effect of adherence on mean BP (per week): {result.params["adherence"]:.2f} mmHg')

# Save results
df[['HASHED_PATIENT_ID','WEEK','adherence','mean_bp','stabilized_weight']].to_csv('msm_panel_with_stabilized_weights.csv', index=False)
with open('msm_gee_results.txt','w') as f:
    f.write(result.summary().as_text())
    f.write(f'\nEstimated effect of adherence on mean BP (per week): {result.params["adherence"]:.2f} mmHg\n')

print('\nFiles saved:')
print('- msm_panel_with_stabilized_weights.csv')
print('- msm_gee_results.txt')

# ---
# Notes:
# - For weeks with only one adherence class, stabilized_weight=1
# - For other weeks, weights are estimated as usual
# - Numerator: P(adherence | gender, BMI, treatment)
# - Denominator: P(adherence | gender, BMI, treatment, previous adherence)
# - Outcome: mean BP per week
# - Exposure: adherence per week
# - Model: GEE (accounts for repeated measures per patient) 

# Diagnostic: write adherence class counts per week to a file
with open('msm_adherence_by_week_debug.txt', 'w') as f:
    for week, sub in df.groupby('WEEK'):
        f.write(f'Week {week}: {sub["adherence"].value_counts().to_dict()}\n') 