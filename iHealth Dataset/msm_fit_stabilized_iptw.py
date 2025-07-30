import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan

# Load long-format panel data (weeks 1-52 only)
df = pd.read_csv('msm_long_panel_with_propensity.csv')
df = df[(df['WEEK'] >= 1) & (df['WEEK'] <= 52)]

# Focus on patients with BP measurements (address data quality issue)
df = df[df['n_bp'] > 0].copy()

# Drop rows with missing required data
df = df.dropna(subset=['ADHERENCE_SCORE','GENDER_AT_BIRTH_M','ASSIGNED_TREATMENT_BP','BMI','mean_bp'])
# Reset index to ensure contiguous indices for GEE model
df = df.reset_index(drop=True)

print(f"=== DATA QUALITY SUMMARY ===")
print(f"Total observations after filtering: {len(df)}")
print(f"Unique patients: {df['HASHED_PATIENT_ID'].nunique()}")
print(f"Weeks range: {df['WEEK'].min()} to {df['WEEK'].max()}")
print(f"Mean BP range: {df['mean_bp'].min():.1f} to {df['mean_bp'].max():.1f} mmHg")
print(f"Adherence score range: {df['ADHERENCE_SCORE'].min():.3f} to {df['ADHERENCE_SCORE'].max():.3f}")

# Standardize continuous variables
scaler = StandardScaler()
df[['BMI_SCALED', 'ADHERENCE_SCORE_SCALED']] = scaler.fit_transform(df[['BMI', 'ADHERENCE_SCORE']])

# ---
# Time Series Diagnostics
print(f"\n=== TIME SERIES DIAGNOSTICS ===")

# Check for stationarity in BP over time
bp_by_week = df.groupby('WEEK')['mean_bp'].mean()
adf_result = adfuller(bp_by_week.dropna())
print(f"BP stationarity test (ADF): p-value = {adf_result[1]:.4f}")
print(f"BP is {'stationary' if adf_result[1] < 0.05 else 'non-stationary'}")

# Check for autocorrelation in residuals (will do after model fitting)
print(f"Mean BP by week trend: {bp_by_week.mean():.1f} mmHg")

# ---
# Modified IPTW for continuous adherence score
def assign_weights_continuous(df):
    df = df.copy()
    df['stabilized_weight'] = np.nan
    
    for week, sub in df.groupby('WEEK'):
        if len(sub) < 10:  # Too few observations for reliable modeling
            df.loc[sub.index, 'stabilized_weight'] = 1.0
        else:
            # Numerator model: P(adherence_score | baseline covariates)
            X_num = sub[['GENDER_AT_BIRTH_M','ASSIGNED_TREATMENT_BP','BMI_SCALED']]
            y = sub['ADHERENCE_SCORE_SCALED']
            
            # Use linear regression for continuous outcome
            ps_num = LinearRegression()
            ps_num.fit(X_num, y)
            ps_num_pred = ps_num.predict(X_num)
            
            # Denominator model: P(adherence_score | baseline covariates + prev_adherence)
            X_den = X_num.copy()
            X_den['prev_adherence_score'] = sub['prev_adherence']
            
            ps_den = LinearRegression()
            ps_den.fit(X_den, y)
            ps_den_pred = ps_den.predict(X_den)
            
            # Calculate stabilized weights (avoid division by zero)
            weights = np.where(ps_den_pred != 0, ps_num_pred / ps_den_pred, 1.0)
            weights = np.clip(weights, 0.1, 10)  # Trim extreme weights
            df.loc[sub.index, 'stabilized_weight'] = weights
    
    return df

df = assign_weights_continuous(df)

print(f"\nStabilized IPTW weights calculated. Mean: {df['stabilized_weight'].mean():.3f}, Std: {df['stabilized_weight'].std():.3f}")

# ---
# Multiple Analysis Approaches

# 1. GEE with time-varying adherence score
print(f"\n=== ANALYSIS 1: GEE WITH CONTINUOUS ADHERENCE SCORE ===")
df = df.sort_values(['HASHED_PATIENT_ID','WEEK'])

model_gee = GEE(
    endog=df['mean_bp'],
    exog=sm.add_constant(df[['ADHERENCE_SCORE_SCALED']]),
    groups=df['HASHED_PATIENT_ID'],
    family=Gaussian(),
    cov_struct=Exchangeable(),
    weights=df['stabilized_weight']
)
result_gee = model_gee.fit()

print(result_gee.summary())
print(f"Estimated effect of adherence score on mean BP: {result_gee.params['ADHERENCE_SCORE_SCALED']:.2f} mmHg per SD increase")

# 2. Time series aware analysis with lagged effects
print(f"\n=== ANALYSIS 2: TIME SERIES WITH LAGGED EFFECTS ===")

# Add lagged adherence score
df['adherence_score_lag1'] = df.groupby('HASHED_PATIENT_ID')['ADHERENCE_SCORE_SCALED'].shift(1).fillna(0)

model_lagged = GEE(
    endog=df['mean_bp'],
    exog=sm.add_constant(df[['ADHERENCE_SCORE_SCALED', 'adherence_score_lag1']]),
    groups=df['HASHED_PATIENT_ID'],
    family=Gaussian(),
    cov_struct=Exchangeable(),
    weights=df['stabilized_weight']
)
result_lagged = model_lagged.fit()

print(result_lagged.summary())
print(f"Current adherence effect: {result_lagged.params['ADHERENCE_SCORE_SCALED']:.2f} mmHg per SD")
print(f"Lagged adherence effect: {result_lagged.params['adherence_score_lag1']:.2f} mmHg per SD")

# 3. Quintile-based analysis for robustness
print(f"\n=== ANALYSIS 3: QUINTILE-BASED ANALYSIS ===")

# Create adherence quintiles
df['adherence_quintile_scaled'] = pd.qcut(df['ADHERENCE_SCORE'], q=5, labels=[1,2,3,4,5], duplicates='drop')
df['adherence_quintile_scaled'] = df['adherence_quintile_scaled'].astype(float)

model_quintile = GEE(
    endog=df['mean_bp'],
    exog=sm.add_constant(df[['adherence_quintile_scaled']]),
    groups=df['HASHED_PATIENT_ID'],
    family=Gaussian(),
    cov_struct=Exchangeable(),
    weights=df['stabilized_weight']
)
result_quintile = model_quintile.fit()

print(result_quintile.summary())
print(f"Effect per quintile increase: {result_quintile.params['adherence_quintile_scaled']:.2f} mmHg")

# ---
# Model Diagnostics
print(f"\n=== MODEL DIAGNOSTICS ===")

# Residual analysis
residuals = result_gee.resid
print(f"Residual mean: {residuals.mean():.3f}")
print(f"Residual std: {residuals.std():.3f}")

# Heteroscedasticity test
bp_test = het_breuschpagan(residuals, result_gee.model.exog)
print(f"Heteroscedasticity test (Breusch-Pagan): p-value = {bp_test[1]:.4f}")

# Save comprehensive results
df[['HASHED_PATIENT_ID','WEEK','ADHERENCE_SCORE','mean_bp','stabilized_weight','adherence_quintile_scaled']].to_csv('msm_panel_continuous_analysis.csv', index=False)

with open('msm_continuous_analysis_results.txt','w') as f:
    f.write("=== CONTINUOUS ADHERENCE SCORE MSM ANALYSIS ===\n\n")
    f.write("1. GEE WITH CONTINUOUS ADHERENCE SCORE:\n")
    f.write(result_gee.summary().as_text())
    f.write(f"\nEstimated effect: {result_gee.params['ADHERENCE_SCORE_SCALED']:.2f} mmHg per SD increase\n\n")
    
    f.write("2. TIME SERIES WITH LAGGED EFFECTS:\n")
    f.write(result_lagged.summary().as_text())
    f.write(f"\nCurrent effect: {result_lagged.params['ADHERENCE_SCORE_SCALED']:.2f} mmHg per SD\n")
    f.write(f"Lagged effect: {result_lagged.params['adherence_score_lag1']:.2f} mmHg per SD\n\n")
    
    f.write("3. QUINTILE-BASED ANALYSIS:\n")
    f.write(result_quintile.summary().as_text())
    f.write(f"\nEffect per quintile: {result_quintile.params['adherence_quintile_scaled']:.2f} mmHg\n\n")
    
    f.write("=== DIAGNOSTICS ===\n")
    f.write(f"Residual mean: {residuals.mean():.3f}\n")
    f.write(f"Residual std: {residuals.std():.3f}\n")
    f.write(f"Heteroscedasticity p-value: {bp_test[1]:.4f}\n")
    f.write(f"BP stationarity p-value: {adf_result[1]:.4f}\n")

print(f"\nFiles saved:")
print(f"- msm_panel_continuous_analysis.csv")
print(f"- msm_continuous_analysis_results.txt")

# ---
# Notes:
# - Uses continuous ADHERENCE_SCORE instead of binary adherence
# - Addresses data quality by focusing on patients with BP measurements
# - Includes time series diagnostics and lagged effects
# - Provides multiple analysis approaches for robustness
# - Includes model diagnostics for validity assessment 