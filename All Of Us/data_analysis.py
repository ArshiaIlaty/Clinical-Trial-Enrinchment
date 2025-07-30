import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== PERSON DATA FILTERING SCRIPT (REFINED) ===\n")

# Load the large dataset in chunks for memory efficiency
print("Loading Person Data dataset...")
chunk_size = 100000  # Process 100k rows at a time

# Initialize counters
total_rows = 0
filtered_rows = 0
filtered_data = []

# Define exclusion criteria for each demographic variable
exclusion_criteria = {
    'gender': [
        'Not man only, not woman only, prefer not to answer, or skipped',
        'No matching concept'
    ],
    'race': [
        'None Indicated',
        'PMI: Skip',
        'I prefer not to answer',
        'No matching concept'
    ],
    'ethnicity': [
        'PMI: Skip',
        'What Race Ethnicity: Race Ethnicity None Of These',
        'PMI: Prefer Not To Answer',
        'No matching concept'
    ],
    'sex_at_birth': [
        'Not male, not female, prefer not to answer, or skipped',
        'No matching concept'
    ],
    'self_reported_category': [
        'PMI: Skip',
        'None of these',
        'I prefer not to answer',
        'No matching concept'
    ]
}

# Read the dataset in chunks
for chunk_num, chunk in enumerate(pd.read_csv('Person Data.csv', chunksize=chunk_size)):
    total_rows += len(chunk)
    print(f"Processing chunk {chunk_num + 1}: {len(chunk):,} rows (Total processed: {total_rows:,})")
    
    # Filter for patients with all required demographic variables declared
    required_columns = ['gender', 'race', 'ethnicity', 'sex_at_birth', 'self_reported_category']
    
    # Create mask for rows where all required columns have valid values
    mask = chunk[required_columns].notna().all(axis=1) & \
           (chunk[required_columns] != '').all(axis=1)
    
    # Apply exclusion criteria for each column
    for col, exclude_values in exclusion_criteria.items():
        for exclude_val in exclude_values:
            mask = mask & (chunk[col] != exclude_val)
    
    # Apply the filter
    filtered_chunk = chunk[mask]
    filtered_rows += len(filtered_chunk)
    
    # Add to our filtered data list
    if len(filtered_chunk) > 0:
        filtered_data.append(filtered_chunk)
    
    # Print progress
    print(f"  - Chunk {chunk_num + 1} filtered: {len(filtered_chunk):,} rows kept")
    print(f"  - Running total filtered: {filtered_rows:,} rows")

print(f"\n=== FILTERING COMPLETE ===")
print(f"Total rows processed: {total_rows:,}")
print(f"Rows after filtering: {filtered_rows:,}")
print(f"Reduction: {((total_rows - filtered_rows) / total_rows * 100):.1f}%")

# Combine all filtered chunks
if filtered_data:
    print("\nCombining filtered data...")
    final_filtered_df = pd.concat(filtered_data, ignore_index=True)
    
    print(f"Final dataset shape: {final_filtered_df.shape}")
    
    # Save the filtered dataset
    output_filename = 'Person_Data_Refined.csv'
    print(f"\nSaving refined dataset to: {output_filename}")
    final_filtered_df.to_csv(output_filename, index=False)
    
    # Display summary statistics
    print("\n=== REFINED DATASET SUMMARY ===")
    print(f"Total patients: {len(final_filtered_df):,}")
    
    # Show distribution of each demographic variable
    print("\nDemographic Variable Distributions:")
    for col in ['gender', 'race', 'ethnicity', 'sex_at_birth', 'self_reported_category']:
        print(f"\n{col.upper()}:")
        value_counts = final_filtered_df[col].value_counts()
        for value, count in value_counts.items():
            percentage = (count / len(final_filtered_df)) * 100
            print(f"  {value}: {count:,} ({percentage:.1f}%)")
    
    # Check for any remaining excluded values
    print("\n=== EXCLUSION CHECK ===")
    print("Checking for any remaining excluded values:")
    for col, exclude_values in exclusion_criteria.items():
        for exclude_val in exclude_values:
            count = (final_filtered_df[col] == exclude_val).sum()
            if count > 0:
                print(f"  {col}: {count:,} instances of '{exclude_val}' found")
            else:
                print(f"  {col}: No instances of '{exclude_val}' ✓")
    
    # Check for any remaining missing values
    print("\n=== MISSING VALUE CHECK ===")
    missing_summary = final_filtered_df[required_columns].isnull().sum()
    print("Missing values in required columns:")
    for col, missing_count in missing_summary.items():
        if missing_count > 0:
            print(f"  {col}: {missing_count:,} missing values")
        else:
            print(f"  {col}: No missing values ✓")
    
    # Save summary statistics
    summary_filename = 'refined_filtering_summary.txt'
    with open(summary_filename, 'w') as f:
        f.write("=== PERSON DATA REFINED FILTERING SUMMARY ===\n\n")
        f.write(f"Original dataset size: {total_rows:,} rows\n")
        f.write(f"Refined dataset size: {filtered_rows:,} rows\n")
        f.write(f"Reduction percentage: {((total_rows - filtered_rows) / total_rows * 100):.1f}%\n\n")
        
        f.write("=== EXCLUSION CRITERIA APPLIED ===\n")
        for col, exclude_values in exclusion_criteria.items():
            f.write(f"\n{col.upper()} excluded values:\n")
            for val in exclude_values:
                f.write(f"  - {val}\n")
        
        f.write("\n=== DEMOGRAPHIC DISTRIBUTIONS ===\n")
        for col in ['gender', 'race', 'ethnicity', 'sex_at_birth', 'self_reported_category']:
            f.write(f"\n{col.upper()}:\n")
            value_counts = final_filtered_df[col].value_counts()
            for value, count in value_counts.items():
                percentage = (count / len(final_filtered_df)) * 100
                f.write(f"  {value}: {count:,} ({percentage:.1f}%)\n")
    
    print(f"\nSummary saved to: {summary_filename}")
    print(f"Refined dataset saved to: {output_filename}")
    
else:
    print("\nNo data passed the filtering criteria!")
    print("Check if the column names match exactly or if the data format is as expected.")

print("\n=== SCRIPT COMPLETE ===")
