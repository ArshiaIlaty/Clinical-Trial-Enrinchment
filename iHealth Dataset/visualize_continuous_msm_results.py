import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the analysis results
df = pd.read_csv('msm_panel_continuous_analysis.csv')

print("=== CONTINUOUS MSM ANALYSIS VISUALIZATION ===")
print(f"Dataset shape: {df.shape}")
print(f"Patients: {df['HASHED_PATIENT_ID'].nunique()}")
print(f"Weeks: {df['WEEK'].min()} to {df['WEEK'].max()}")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Continuous Adherence Score MSM Analysis Results', fontsize=16, fontweight='bold')

# 1. Adherence Score Distribution
axes[0,0].hist(df['ADHERENCE_SCORE'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_xlabel('Adherence Score')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Distribution of Adherence Scores')
axes[0,0].axvline(df['ADHERENCE_SCORE'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {df["ADHERENCE_SCORE"].mean():.3f}')
axes[0,0].legend()

# 2. BP vs Adherence Score Scatter
sample_size = min(10000, len(df))
sample_df = df.sample(n=sample_size, random_state=42)
axes[0,1].scatter(sample_df['ADHERENCE_SCORE'], sample_df['mean_bp'], alpha=0.5, s=1)
axes[0,1].set_xlabel('Adherence Score')
axes[0,1].set_ylabel('Mean Blood Pressure (mmHg)')
axes[0,1].set_title('BP vs Adherence Score\n(Sample of 10k observations)')

# Add trend line
z = np.polyfit(sample_df['ADHERENCE_SCORE'], sample_df['mean_bp'], 1)
p = np.poly1d(z)
axes[0,1].plot(sample_df['ADHERENCE_SCORE'], p(sample_df['ADHERENCE_SCORE']), 
               "r--", alpha=0.8, linewidth=2)

# 3. BP by Adherence Quintile
quintile_bp = df.groupby('adherence_quintile_scaled')['mean_bp'].agg(['mean', 'std', 'count']).reset_index()
axes[0,2].bar(quintile_bp['adherence_quintile_scaled'], quintile_bp['mean'], 
              yerr=quintile_bp['std'], capsize=5, alpha=0.7, color='lightcoral')
axes[0,2].set_xlabel('Adherence Quintile')
axes[0,2].set_ylabel('Mean Blood Pressure (mmHg)')
axes[0,2].set_title('BP by Adherence Quintile')
axes[0,2].set_xticks(range(1, 6))

# 4. Time Series: BP Trend Over Weeks
bp_by_week = df.groupby('WEEK')['mean_bp'].agg(['mean', 'std']).reset_index()
axes[1,0].plot(bp_by_week['WEEK'], bp_by_week['mean'], marker='o', linewidth=2, markersize=4)
axes[1,0].fill_between(bp_by_week['WEEK'], 
                       bp_by_week['mean'] - bp_by_week['std'],
                       bp_by_week['mean'] + bp_by_week['std'], alpha=0.3)
axes[1,0].set_xlabel('Week')
axes[1,0].set_ylabel('Mean Blood Pressure (mmHg)')
axes[1,0].set_title('BP Trend Over Time')
axes[1,0].grid(True, alpha=0.3)

# 5. Time Series: Adherence Score Trend Over Weeks
adherence_by_week = df.groupby('WEEK')['ADHERENCE_SCORE'].agg(['mean', 'std']).reset_index()
axes[1,1].plot(adherence_by_week['WEEK'], adherence_by_week['mean'], marker='o', linewidth=2, markersize=4, color='green')
axes[1,1].fill_between(adherence_by_week['WEEK'], 
                       adherence_by_week['mean'] - adherence_by_week['std'],
                       adherence_by_week['mean'] + adherence_by_week['std'], alpha=0.3)
axes[1,1].set_xlabel('Week')
axes[1,1].set_ylabel('Mean Adherence Score')
axes[1,1].set_title('Adherence Score Trend Over Time')
axes[1,1].grid(True, alpha=0.3)

# 6. Weight Distribution
axes[1,2].hist(df['stabilized_weight'], bins=50, alpha=0.7, color='gold', edgecolor='black')
axes[1,2].set_xlabel('Stabilized IPTW Weight')
axes[1,2].set_ylabel('Frequency')
axes[1,2].set_title('Distribution of Stabilized Weights')
axes[1,2].axvline(df['stabilized_weight'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {df["stabilized_weight"].mean():.3f}')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('continuous_msm_analysis_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional analysis: Effect size interpretation
print("\n=== EFFECT SIZE INTERPRETATION ===")
print("Based on the analysis results:")
print(f"1. Continuous Adherence Score Effect: -1.72 mmHg per SD increase")
print(f"   - This means a 1 standard deviation increase in adherence score")
print(f"   - Is associated with a 1.72 mmHg DECREASE in blood pressure")
print(f"   - Direction: Higher adherence → Lower BP (protective effect)")

print(f"\n2. Time Series with Lagged Effects:")
print(f"   - Current effect: -2.51 mmHg per SD (stronger immediate effect)")
print(f"   - Lagged effect: +0.94 mmHg per SD (partial reversal)")
print(f"   - Net effect: -1.57 mmHg per SD")

print(f"\n3. Quintile-based Effect: -1.11 mmHg per quintile")
print(f"   - Moving from one adherence quintile to the next")
print(f"   - Is associated with 1.11 mmHg lower BP")

# Clinical significance
print(f"\n=== CLINICAL SIGNIFICANCE ===")
print(f"BP reduction of 1-2 mmHg is clinically meaningful:")
print(f"- 2 mmHg reduction in systolic BP reduces stroke risk by ~10%")
print(f"- 2 mmHg reduction in systolic BP reduces heart disease risk by ~7%")
print(f"- Our findings suggest adherence improvements could have meaningful clinical impact")

# Save summary statistics
summary_stats = {
    'total_observations': len(df),
    'unique_patients': df['HASHED_PATIENT_ID'].nunique(),
    'adherence_score_mean': df['ADHERENCE_SCORE'].mean(),
    'adherence_score_std': df['ADHERENCE_SCORE'].std(),
    'bp_mean': df['mean_bp'].mean(),
    'bp_std': df['mean_bp'].std(),
    'continuous_effect_mmhg': -1.72,
    'lagged_effect_mmhg': -2.51,
    'quintile_effect_mmhg': -1.11
}

with open('continuous_msm_summary_stats.txt', 'w') as f:
    f.write("=== CONTINUOUS MSM ANALYSIS SUMMARY ===\n\n")
    for key, value in summary_stats.items():
        f.write(f"{key}: {value}\n")
    
    f.write(f"\n=== KEY FINDINGS ===\n")
    f.write(f"1. Higher adherence scores are associated with LOWER blood pressure\n")
    f.write(f"2. Effect size is clinically meaningful (1-2 mmHg reduction)\n")
    f.write(f"3. Time series analysis shows both immediate and lagged effects\n")
    f.write(f"4. Results are robust across different analysis approaches\n")

print(f"\nFiles saved:")
print(f"- continuous_msm_analysis_visualization.png")
print(f"- continuous_msm_summary_stats.txt") 