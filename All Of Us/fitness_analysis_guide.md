# Fitness Data Analysis Guide

## Overview
This guide provides comprehensive analysis tools for large fitness datasets (40M+ rows) with the following columns:
- `person_id` - Unique user identifier
- `date` - Date of activity
- `activity_calories` - Calories burned from activity
- `calories_bmr` - Basal metabolic rate calories
- `calories_out` - Total calories burned
- `elevation` - Elevation data
- `fairly_active_minutes` - Minutes of fairly active exercise
- `floors` - Floors climbed
- `lightly_active_minutes` - Minutes of light activity
- `marginal_calories` - Marginal calories burned
- `sedentary_minutes` - Minutes spent sedentary
- `steps` - Step count
- `very_active_minutes` - Minutes of very active exercise

## Files Included

### 1. `fitness_data_analysis.py` (Comprehensive Version)
- **Full-featured analysis** with 12 different visualizations
- **User-level insights** and segmentation
- **Time series analysis** with trends
- **Detailed summary reports**
- **Memory-efficient chunked processing**

### 2. `fitness_analysis_notebook.py` (Simplified Version)
- **Streamlined analysis** with 6 key visualizations
- **Quick insights** generation
- **Easy to modify** and adapt
- **Perfect for initial exploration**

## Usage Instructions

### Step 1: Prepare Your Data
Ensure your CSV file has the exact column names listed above. The `date` column should be in a format pandas can parse (e.g., 'YYYY-MM-DD').

### Step 2: Update File Path
In both scripts, update the file path:
```python
file_path = 'your_fitness_data.csv'  # Change this to your actual filename
```

### Step 3: Run the Analysis

#### Option A: Comprehensive Analysis
```bash
python fitness_data_analysis.py
```

#### Option B: Quick Analysis
```bash
python fitness_analysis_notebook.py
```

## Output Files Generated

### Visualizations
- **`fitness_data_overview.png`** - 12-panel comprehensive overview
- **`fitness_user_insights.png`** - User-level analysis and segmentation
- **`fitness_time_series.png`** - Time-based trends and patterns
- **`fitness_dashboard.png`** - 6-panel quick dashboard

### Reports
- **`fitness_analysis_summary.txt`** - Detailed analysis summary
- **`fitness_insights.txt`** - Key insights and statistics

## Key Features

### 🚀 Memory Efficient
- **Chunked processing** handles 40M+ rows without memory issues
- **Sample-based analysis** for detailed visualizations
- **Progressive loading** with progress updates

### 📊 Comprehensive Visualizations
1. **Steps Distribution** - Activity level patterns
2. **Calories vs Steps** - Energy expenditure correlation
3. **Activity Minutes** - Exercise intensity breakdown
4. **Daily Patterns** - Weekly activity cycles
5. **Correlation Matrix** - Variable relationships
6. **User Engagement** - Participation patterns
7. **Time Series Trends** - Longitudinal patterns
8. **User Segmentation** - Activity level classification

### 🎯 User Insights
- **Activity Level Classification**: Low, Moderate, High, Very High
- **Consistency Scoring**: User engagement patterns
- **Demographic Analysis**: User behavior segmentation
- **Engagement Metrics**: Participation tracking

### 📈 Time Series Analysis
- **Daily Trends**: Step and calorie patterns
- **Weekly Cycles**: Day-of-week effects
- **Monthly Patterns**: Seasonal variations
- **User Engagement**: Participation over time

## Customization Options

### Adjust Chunk Size
For different memory constraints:
```python
chunk_size = 50000  # Smaller chunks for less memory
chunk_size = 200000  # Larger chunks for faster processing
```

### Modify Activity Levels
Customize user segmentation:
```python
user_stats['activity_level'] = pd.cut(user_stats['steps_mean'], 
                                     bins=[0, 3000, 6000, 9000, float('inf')],  # Custom thresholds
                                     labels=['Sedentary', 'Low', 'Moderate', 'Active'])
```

### Add Custom Visualizations
Extend the analysis with additional plots:
```python
# Example: Add heart rate analysis if available
if 'heart_rate' in df.columns:
    axes[2,3].hist(df['heart_rate'].dropna(), bins=30, alpha=0.7)
    axes[2,3].set_title('Heart Rate Distribution')
```

## Performance Tips

### For Very Large Datasets (>100M rows)
1. **Increase chunk size** to 200,000-500,000
2. **Reduce sample size** for visualizations
3. **Use sampling** for detailed analysis
4. **Consider database storage** for frequent queries

### For Real-time Analysis
1. **Pre-aggregate data** by day/user
2. **Use incremental processing**
3. **Cache intermediate results**
4. **Implement streaming analysis**

## Troubleshooting

### Common Issues

**Memory Error**: Reduce chunk size or use sampling
```python
chunk_size = 50000  # Smaller chunks
```

**File Not Found**: Check file path and permissions
```python
file_path = '/full/path/to/your/data.csv'
```

**Date Parsing Error**: Ensure consistent date format
```python
# Add date parsing options if needed
parse_dates=['date'], date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
```

**Missing Columns**: Verify column names match exactly
```python
# Check available columns
print(df.columns.tolist())
```

## Advanced Analysis Ideas

### 1. Predictive Modeling
- **Dropout prediction** based on activity patterns
- **Goal achievement** probability
- **Health risk assessment**

### 2. Clustering Analysis
- **User behavior clusters**
- **Activity pattern groups**
- **Engagement segments**

### 3. Anomaly Detection
- **Unusual activity patterns**
- **Data quality issues**
- **Device malfunction detection**

### 4. Cohort Analysis
- **User retention by activity level**
- **Seasonal engagement patterns**
- **Feature adoption rates**

## Support

For questions or customizations:
1. Check the code comments for detailed explanations
2. Modify parameters based on your specific needs
3. Add custom visualizations for your use case
4. Scale processing based on your dataset size

---

**Happy Analyzing! 🏃‍♂️📊** 