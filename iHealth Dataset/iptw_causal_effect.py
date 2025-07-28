import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Load data
adherence = pd.read_csv('bp_adherence_scores.csv')
demo = pd.read_csv('demographics.csv')

# Load BP measurements and calculate baseline BP
bp = pd.read_csv('bp_measurements.csv', usecols=['HASHED_PATIENT_ID','MEASURMENT_TS','SYSTOLIC_BLOOD_PRESSURE'])
bp['MEASURMENT_TS'] = pd.to_datetime(bp['MEASURMENT_TS'], errors='coerce')
bp['SYSTOLIC_BLOOD_PRESSURE'] = pd.to_numeric(bp['SYSTOLIC_BLOOD_PRESSURE'], errors='coerce')

# Calculate baseline BP (first measurement per patient)
baseline_bp = bp.sort_values('MEASURMENT_TS').groupby('HASHED_PATIENT_ID').first().reset_index()
baseline_bp = baseline_bp.rename(columns={'SYSTOLIC_BLOOD_PRESSURE':'BASELINE_BP'})

# Calculate follow-up BP (last measurement per patient)
followup_bp = bp.sort_values('MEASURMENT_TS').groupby('HASHED_PATIENT_ID').last().reset_index()
followup_bp = followup_bp.rename(columns={'SYSTOLIC_BLOOD_PRESSURE':'FOLLOWUP_BP'})

# Merge baseline and follow-up
bp_change = pd.merge(baseline_bp[['HASHED_PATIENT_ID','BASELINE_BP']], followup_bp[['HASHED_PATIENT_ID','FOLLOWUP_BP']], on='HASHED_PATIENT_ID')
bp_change['BP_CHANGE'] = bp_change['FOLLOWUP_BP'] - bp_change['BASELINE_BP']

# Load A1C and calculate change
try:
    a1c = pd.read_csv('a1c_measurements.csv', usecols=['HASHED_PATIENT_ID','COLLECTED_DATE','A1C_VALUE'])
    a1c['COLLECTED_DATE'] = pd.to_datetime(a1c['COLLECTED_DATE'], errors='coerce')
    a1c['A1C_VALUE'] = pd.to_numeric(a1c['A1C_VALUE'], errors='coerce')
    baseline_a1c = a1c.sort_values('COLLECTED_DATE').groupby('HASHED_PATIENT_ID').first().reset_index().rename(columns={'A1C_VALUE':'BASELINE_A1C'})
    followup_a1c = a1c.sort_values('COLLECTED_DATE').groupby('HASHED_PATIENT_ID').last().reset_index().rename(columns={'A1C_VALUE':'FOLLOWUP_A1C'})
    a1c_change = pd.merge(baseline_a1c[['HASHED_PATIENT_ID','BASELINE_A1C']], followup_a1c[['HASHED_PATIENT_ID','FOLLOWUP_A1C']], on='HASHED_PATIENT_ID')
    a1c_change['A1C_CHANGE'] = a1c_change['FOLLOWUP_A1C'] - a1c_change['BASELINE_A1C']
except Exception as e:
    print('A1C calculation failed:', e)
    a1c_change = None

# Load weight and calculate change
try:
    weight = pd.read_csv('weight_measurements.csv', usecols=['HASHED_PATIENT_ID','MEASURMENT_TS','BODY_WEIGHT_VALUE'])
    weight['MEASURMENT_TS'] = pd.to_datetime(weight['MEASURMENT_TS'], errors='coerce')
    weight['BODY_WEIGHT_VALUE'] = pd.to_numeric(weight['BODY_WEIGHT_VALUE'], errors='coerce')
    baseline_weight = weight.sort_values('MEASURMENT_TS').groupby('HASHED_PATIENT_ID').first().reset_index().rename(columns={'BODY_WEIGHT_VALUE':'BASELINE_WEIGHT'})
    followup_weight = weight.sort_values('MEASURMENT_TS').groupby('HASHED_PATIENT_ID').last().reset_index().rename(columns={'BODY_WEIGHT_VALUE':'FOLLOWUP_WEIGHT'})
    weight_change = pd.merge(baseline_weight[['HASHED_PATIENT_ID','BASELINE_WEIGHT']], followup_weight[['HASHED_PATIENT_ID','FOLLOWUP_WEIGHT']], on='HASHED_PATIENT_ID')
    weight_change['WEIGHT_CHANGE'] = weight_change['FOLLOWUP_WEIGHT'] - weight_change['BASELINE_WEIGHT']
except Exception as e:
    print('Weight calculation failed:', e)
    weight_change = None

# Merge all data
full = adherence.merge(demo[['HASHED_PATIENT_ID','GENDER_AT_BIRTH','BMI','BMI_CATEGORY','CALCULATED_AGE_GROUP','ASSIGNED_TREATMENT']], on='HASHED_PATIENT_ID', how='left')
full = full.merge(bp_change[['HASHED_PATIENT_ID','BP_CHANGE','BASELINE_BP']], on='HASHED_PATIENT_ID', how='left')
if a1c_change is not None:
    full = full.merge(a1c_change[['HASHED_PATIENT_ID','A1C_CHANGE','BASELINE_A1C']], on='HASHED_PATIENT_ID', how='left')
if weight_change is not None:
    full = full.merge(weight_change[['HASHED_PATIENT_ID','WEIGHT_CHANGE','BASELINE_WEIGHT']], on='HASHED_PATIENT_ID', how='left')

# Drop rows with missing key data
full = full.dropna(subset=['ADHERENCE_QUINTILE','BP_CHANGE','BASELINE_BP','BMI','GENDER_AT_BIRTH','ASSIGNED_TREATMENT'])

# Define treatment: High adherence (Q5) vs. Low adherence (Q1)
full = full[full['ADHERENCE_QUINTILE'].isin([1,5])]
full['TREATED'] = (full['ADHERENCE_QUINTILE'] == 5).astype(int)

# Covariate selection and encoding
covariates = ['BASELINE_BP','BMI','GENDER_AT_BIRTH','ASSIGNED_TREATMENT']
X = full[covariates].copy()
X = pd.get_dummies(X, columns=['GENDER_AT_BIRTH','ASSIGNED_TREATMENT'], drop_first=True)
scaler = StandardScaler()
X[['BASELINE_BP','BMI']] = scaler.fit_transform(X[['BASELINE_BP','BMI']])

# Propensity score estimation
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X, full['TREATED'])
full['propensity'] = ps_model.predict_proba(X)[:,1]

# IPTW weights
full['weight'] = np.where(full['TREATED']==1, 1/full['propensity'], 1/(1-full['propensity']))

# Weighted regression for BP change
X_reg = sm.add_constant(full['TREATED'])
wls_model = sm.WLS(full['BP_CHANGE'], X_reg, weights=full['weight'])
results = wls_model.fit()
print('=== IPTW Causal Effect: High vs Low Adherence on BP Change ===')
print(results.summary())
print(f'Estimated causal effect (ATE) of high adherence (Q5 vs Q1) on BP change: {results.params["TREATED"]:.2f} mmHg')

# Weighted regression for A1C change (if available)
if 'A1C_CHANGE' in full.columns:
    y = full.dropna(subset=['A1C_CHANGE'])
    X_reg_a1c = sm.add_constant(y['TREATED'])
    wls_model_a1c = sm.WLS(y['A1C_CHANGE'], X_reg_a1c, weights=y['weight'])
    results_a1c = wls_model_a1c.fit()
    print('\n=== IPTW Causal Effect: High vs Low Adherence on A1C Change ===')
    print(results_a1c.summary())
    print(f'Estimated causal effect (ATE) of high adherence (Q5 vs Q1) on A1C change: {results_a1c.params["TREATED"]:.2f}')

# Weighted regression for weight change (if available)
if 'WEIGHT_CHANGE' in full.columns:
    y = full.dropna(subset=['WEIGHT_CHANGE'])
    X_reg_weight = sm.add_constant(y['TREATED'])
    wls_model_weight = sm.WLS(y['WEIGHT_CHANGE'], X_reg_weight, weights=y['weight'])
    results_weight = wls_model_weight.fit()
    print('\n=== IPTW Causal Effect: High vs Low Adherence on Weight Change ===')
    print(results_weight.summary())
    print(f'Estimated causal effect (ATE) of high adherence (Q5 vs Q1) on weight change: {results_weight.params["TREATED"]:.2f} kg') 