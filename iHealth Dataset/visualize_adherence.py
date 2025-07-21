import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the adherence scores
df = pd.read_csv('bp_adherence_scores.csv')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('BP Adherence Score Distribution Analysis', fontsize=16, fontweight='bold')

# 1. Histogram of adherence scores
axes[0, 0].hist(df['ADHERENCE_SCORE'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of Adherence Scores')
axes[0, 0].set_xlabel('Adherence Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df['ADHERENCE_SCORE'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["ADHERENCE_SCORE"].mean():.3f}')
axes[0, 0].axvline(df['ADHERENCE_SCORE'].median(), color='orange', linestyle='--', 
                   label=f'Median: {df["ADHERENCE_SCORE"].median():.3f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Box plot by quintile
quintile_order = [1, 2, 3, 4, 5]
box_data = [df[df['ADHERENCE_QUINTILE'] == q]['ADHERENCE_SCORE'] for q in quintile_order]
axes[0, 1].boxplot(box_data, labels=[f'Q{q}' for q in quintile_order])
axes[0, 1].set_title('Adherence Scores by Quintile')
axes[0, 1].set_xlabel('Adherence Quintile')
axes[0, 1].set_ylabel('Adherence Score')
axes[0, 1].grid(True, alpha=0.3)

# 3. Violin plot by quintile
sns.violinplot(data=df, x='ADHERENCE_QUINTILE', y='ADHERENCE_SCORE', ax=axes[1, 0])
axes[1, 0].set_title('Adherence Score Distribution by Quintile (Violin Plot)')
axes[1, 0].set_xlabel('Adherence Quintile')
axes[1, 0].set_ylabel('Adherence Score')

# 4. Cumulative distribution
sorted_scores = np.sort(df['ADHERENCE_SCORE'])
cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
axes[1, 1].plot(sorted_scores, cumulative, linewidth=2, color='green')
axes[1, 1].set_title('Cumulative Distribution of Adherence Scores')
axes[1, 1].set_xlabel('Adherence Score')
axes[1, 1].set_ylabel('Cumulative Probability')
axes[1, 1].grid(True, alpha=0.3)

# Add percentile lines
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    value = df['ADHERENCE_SCORE'].quantile(p/100)
    axes[1, 1].axvline(value, color='red', linestyle=':', alpha=0.7)
    axes[1, 1].text(value, 0.1, f'{p}%', rotation=90, fontsize=8)

plt.tight_layout()
plt.savefig('adherence_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create additional detailed plots
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
fig2.suptitle('Detailed Adherence Analysis', fontsize=16, fontweight='bold')

# 5. Quintile statistics bar plot
quintile_stats = df.groupby('ADHERENCE_QUINTILE')['ADHERENCE_SCORE'].agg(['mean', 'std']).reset_index()
x_pos = np.arange(len(quintile_stats))
bars = axes2[0].bar(x_pos, quintile_stats['mean'], yerr=quintile_stats['std'], 
                    capsize=5, alpha=0.7, color='lightcoral')
axes2[0].set_title('Mean Adherence Score by Quintile (±1 SD)')
axes2[0].set_xlabel('Adherence Quintile')
axes2[0].set_ylabel('Mean Adherence Score')
axes2[0].set_xticks(x_pos)
axes2[0].set_xticklabels([f'Q{q}' for q in quintile_stats['ADHERENCE_QUINTILE']])
axes2[0].grid(True, alpha=0.3)

# Add value labels on bars
for bar, mean_val in zip(bars, quintile_stats['mean']):
    height = bar.get_height()
    axes2[0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                  f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')

# 6. Log-scale histogram for better visualization of low adherence
axes2[1].hist(df['ADHERENCE_SCORE'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
axes2[1].set_title('Adherence Score Distribution (Log Scale Y-axis)')
axes2[1].set_xlabel('Adherence Score')
axes2[1].set_ylabel('Frequency (Log Scale)')
axes2[1].set_yscale('log')
axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('adherence_detailed_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics for the plots
print("=== VISUALIZATION SUMMARY ===")
print(f"Total patients analyzed: {len(df)}")
print(f"Adherence score range: {df['ADHERENCE_SCORE'].min():.3f} - {df['ADHERENCE_SCORE'].max():.3f}")
print(f"Mean adherence: {df['ADHERENCE_SCORE'].mean():.3f}")
print(f"Median adherence: {df['ADHERENCE_SCORE'].median():.3f}")
print(f"Standard deviation: {df['ADHERENCE_SCORE'].std():.3f}")

print("\nQuintile Summary:")
quintile_summary = df.groupby('ADHERENCE_QUINTILE')['ADHERENCE_SCORE'].agg(['count', 'mean', 'std', 'min', 'max'])
print(quintile_summary.round(4))

print("\nPlots saved as:")
print("- adherence_distribution_analysis.png")
print("- adherence_detailed_analysis.png") 