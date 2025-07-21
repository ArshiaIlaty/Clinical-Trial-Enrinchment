import pandas as pd
import numpy as np

# Load the adherence scores
df = pd.read_csv('bp_adherence_scores.csv')

print("=== BP ADHERENCE SCORES ANALYSIS ===\n")

# Basic statistics
print("1. BASIC STATISTICS:")
print(f"Total patients with adherence scores: {len(df)}")
print(f"Adherence score range: {df['ADHERENCE_SCORE'].min():.3f} - {df['ADHERENCE_SCORE'].max():.3f}")
print(f"Mean adherence score: {df['ADHERENCE_SCORE'].mean():.3f}")
print(f"Median adherence score: {df['ADHERENCE_SCORE'].median():.3f}")
print(f"Standard deviation: {df['ADHERENCE_SCORE'].std():.3f}")

print("\n2. QUINTILE DISTRIBUTION:")
quintile_counts = df['ADHERENCE_QUINTILE'].value_counts().sort_index()
for quintile, count in quintile_counts.items():
    percentage = (count / len(df)) * 100
    print(f"Quintile {quintile}: {count} patients ({percentage:.1f}%)")

print("\n3. ADHERENCE SCORE BY QUINTILE:")
quintile_stats = df.groupby('ADHERENCE_QUINTILE')['ADHERENCE_SCORE'].agg(['mean', 'std', 'min', 'max'])
print(quintile_stats)

print("\n4. PERCENTILE BREAKDOWN:")
percentiles = [10, 25, 50, 75, 90, 95, 99]
print("Adherence score percentiles:")
for p in percentiles:
    value = df['ADHERENCE_SCORE'].quantile(p/100)
    print(f"  {p}th percentile: {value:.3f}")

print("\n5. DATA QUALITY CHECK:")
print(f"Missing adherence scores: {df['ADHERENCE_SCORE'].isnull().sum()}")
print(f"Missing quintiles: {df['ADHERENCE_QUINTILE'].isnull().sum()}")

# Save detailed statistics
print("\n6. SAVING DETAILED STATISTICS...")
detailed_stats = df.describe()
detailed_stats.to_csv('adherence_detailed_stats.csv')
print("Saved adherence_detailed_stats.csv")

# Quintile summary
quintile_summary = df.groupby('ADHERENCE_QUINTILE').agg({
    'ADHERENCE_SCORE': ['count', 'mean', 'std', 'min', 'max']
}).round(3)
quintile_summary.to_csv('adherence_quintile_summary.csv')
print("Saved adherence_quintile_summary.csv")

print("\n=== ANALYSIS COMPLETE ===") 