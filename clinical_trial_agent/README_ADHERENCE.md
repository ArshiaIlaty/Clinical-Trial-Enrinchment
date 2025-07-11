# Clinical Trial Adherence Analysis System

## Overview

This system provides comprehensive adherence labeling and analysis capabilities for clinical trial data, specifically designed to work with the All of Us Registered Tier Dataset v8. The system implements both multi-class and state-based labeling strategies based on research from blood pressure monitoring studies and clinical trial best practices.

## Features

### 🎯 Adherence Labeling Strategies

#### Multi-Class Labels (High/Medium/Low)
- **High Adherence**: ≥80% Fitbit coverage, ≥3 surveys completed, regular lab visits
- **Medium Adherence**: 40-79% Fitbit coverage, 1-2 surveys, some missing visits
- **Low Adherence**: <40% Fitbit coverage, few/no surveys, large data gaps

#### State-Based Labels (Active/Inactive/Exit)
- **Active**: Daily Fitbit data, recent surveys (≤30d), recent labs (≤90d)
- **Inactive**: No Fitbit for ≥7 days, no labs/surveys for ≥90 days
- **Exit**: No data for >180 days, stopped syncing, device issues

### 📊 Data Integration

#### All of Us Dataset Support
- **Person Domain**: Demographics, age, gender, race, ethnicity
- **Fitbit Activity**: Steps, calories, active minutes, sedentary time
- **Fitbit Device**: Battery status, sync patterns, device type
- **Fitbit Heart Rate**: Daily averages, variability patterns
- **Fitbit Sleep**: Duration, efficiency, sleep stages
- **Survey Data**: Completion rates, response patterns, survey types

### 🔍 Analysis Capabilities

#### Pattern Analysis
- Demographic analysis by adherence level
- Behavioral pattern identification
- Risk factor analysis
- Time-series state transitions

#### Visualization
- Adherence distribution charts
- Activity pattern comparisons
- Sleep quality analysis
- Survey completion trends
- Demographic breakdowns

## Installation

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn asyncio
```

### System Requirements
- Python 3.8+
- 8GB+ RAM (for large datasets)
- 2GB+ disk space for outputs

## Quick Start

### 1. Basic Usage

```python
import asyncio
from clinical_trial_agent.data_integration.all_of_us_connector import AllOfUsConnector
from clinical_trial_agent.models.adherence_labeler import AllOfUsAdherenceLabeler

async def basic_analysis():
    # Initialize connectors
    connector = AllOfUsConnector()
    labeler = AllOfUsAdherenceLabeler()
    
    # Load data
    data = await connector.load_all_domains(
        person_ids=[1000, 1001, 1002],  # Sample participants
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31)
    )
    
    # Create labels
    multi_class_labels = labeler.create_multi_class_labels(data, study_period_days=365)
    state_labels = labeler.create_state_based_labels(
        data, 
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31)
    )
    
    # Analyze patterns
    analysis = labeler.analyze_adherence_patterns(multi_class_labels, data)
    
    return multi_class_labels, state_labels, analysis

# Run analysis
results = asyncio.run(basic_analysis())
```

### 2. Complete Analysis Example

Run the comprehensive example:

```bash
cd clinical_trial_agent/examples
python adherence_analysis_example.py
```

This will:
- Load sample All of Us data
- Create both labeling strategies
- Generate comprehensive visualizations
- Export results and analysis reports

## Configuration

### AdherenceConfig Parameters

```python
from clinical_trial_agent.models.adherence_labeler import AdherenceConfig

config = AdherenceConfig(
    # Fitbit coverage thresholds
    fitbit_coverage_high=0.8,    # 80% days with data
    fitbit_coverage_medium=0.4,  # 40% days with data
    fitbit_coverage_low=0.2,     # 20% days with data
    
    # Survey completion thresholds
    survey_completion_high=3,    # 3+ surveys
    survey_completion_medium=1,  # 1-2 surveys
    
    # Time gap thresholds (days)
    inactive_gap=7,              # 7+ days without Fitbit
    exit_gap=180,                # 180+ days without any data
    
    # Lab visit expectations
    lab_visit_expected_months=6, # Every 6 months
    lab_visit_tolerance_days=30, # 30-day tolerance
)

labeler = AllOfUsAdherenceLabeler(config)
```

## Data Schema

### All of Us Dataset v8 Schema

#### Person Domain
```python
{
    'person_id': int,           # Unique participant ID
    'gender': str,              # Male/Female
    'date_of_birth': datetime,  # Birth date
    'race': str,                # Race category
    'ethnicity': str,           # Ethnicity category
    'sex_at_birth': str         # Sex at birth
}
```

#### Fitbit Activity Domain
```python
{
    'person_id': int,           # Participant ID
    'date': datetime,           # Activity date
    'steps': int,               # Daily step count
    'calories_out': float,      # Calories burned
    'very_active_minutes': int, # Vigorous activity
    'fairly_active_minutes': int, # Moderate activity
    'lightly_active_minutes': int, # Light activity
    'sedentary_minutes': int    # Inactive time
}
```

#### Fitbit Sleep Domain
```python
{
    'person_id': int,           # Participant ID
    'sleep_date': datetime,     # Sleep date
    'minute_in_bed': int,       # Total time in bed
    'minute_asleep': int,       # Time actually asleep
    'minute_deep': int,         # Deep sleep time
    'minute_light': int,        # Light sleep time
    'minute_rem': int,          # REM sleep time
    'minute_awake': int         # Awake time
}
```

#### Survey Domain
```python
{
    'person_id': int,           # Participant ID
    'survey_datetime': datetime, # Survey completion time
    'survey': str,              # Survey name (e.g., 'The Basics')
    'question': str,            # Question text
    'answer': str               # Participant response
}
```

## Adherence Scoring Algorithm

### Multi-Class Score Calculation

The adherence score is calculated as a weighted average of four components:

1. **Fitbit Activity Coverage (40% weight)**
   - Daily data availability
   - Device sync patterns
   - Battery status considerations

2. **Survey Completion Rate (25% weight)**
   - Number of surveys completed
   - Survey frequency patterns
   - Time gaps between surveys

3. **Lab Visit Adherence (20% weight)**
   - Expected vs. actual lab visits
   - Visit timing compliance
   - Follow-up completion

4. **Behavioral Consistency (15% weight)**
   - Activity pattern stability
   - Sleep consistency
   - Heart rate variability

### State Transition Logic

```python
def determine_state(fitbit_data, survey_data, lab_data, date):
    has_recent_fitbit = check_fitbit_recent(fitbit_data, date, 7)
    has_recent_surveys = check_survey_recent(survey_data, date, 90)
    has_recent_labs = check_lab_recent(lab_data, date, 90)
    
    if has_recent_fitbit and (has_recent_surveys or has_recent_labs):
        return "active"
    elif has_recent_fitbit or has_recent_surveys or has_recent_labs:
        return "inactive"
    else:
        # Check for exit condition (180+ days without data)
        if no_data_for_180_days(data, date):
            return "exit"
        else:
            return "inactive"
```

## Output Files

### Generated Files

Running the analysis creates the following outputs in `adherence_analysis_output/`:

#### Data Files
- `multi_class_labels.csv` - High/Medium/Low adherence labels
- `state_based_labels.csv` - Active/Inactive/Exit state labels
- `analysis_results.json` - Comprehensive analysis results
- `labeling_config.json` - Configuration parameters used

#### Visualizations
- `adherence_distribution.png` - Pie chart of adherence levels
- `adherence_scores.png` - Score distribution histograms
- `state_transitions.png` - State changes over time
- `activity_patterns.png` - Activity by adherence level
- `sleep_patterns.png` - Sleep patterns by adherence level
- `survey_patterns.png` - Survey completion patterns
- `demographic_analysis.png` - Demographics by adherence

#### Reports
- `summary_report.txt` - Text summary of key findings
- `data_summary.json` - Dataset statistics

## Advanced Usage

### Custom Data Loading

```python
# Load specific domains only
data = await connector.load_all_domains(
    domains=['person', 'fitbit_activity', 'survey'],
    person_ids=[1000, 1001, 1002],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)

# Load single domain
fitbit_data = await connector.load_domain(
    'fitbit_activity',
    person_ids=[1000, 1001, 1002]
)
```

### Custom Analysis

```python
# Analyze specific patterns
analysis = labeler.analyze_adherence_patterns(labels_df, data)

# Access specific analysis components
demographics = analysis['demographic_analysis']
behavioral_patterns = analysis['behavioral_patterns']
risk_factors = analysis['risk_factors']

# Export specific results
labeler.export_labels(
    labels_df, 
    'custom_labels.csv', 
    format='csv'
)
```

### Custom Visualizations

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create custom plots
plt.figure(figsize=(12, 8))
sns.boxplot(data=merged_data, x='adherence_level', y='steps')
plt.title('Daily Steps by Adherence Level')
plt.savefig('custom_plot.png')
plt.close()
```

## Research Background

### Previous Research Example (Blood Pressure Monitoring)

- **Study Duration**: 4-year remote patient monitoring program (Mar 2019 - Mar 2024)
- **Adherence Components**:
  - Number of readings in 6-month period
  - Average weekly measurement gaps
  - Duration of participation
  - Standard deviation of measurements
- **Key Finding**: Clear gradient showing higher adherence led to better BP outcomes
- **Analysis Method**: Inverse propensity score treatment weighting for causal analysis
- **Limitation**: Did not account for time-varying confounding

### Current Implementation Improvements

- **Multi-modal Data Integration**: Combines wearable, survey, and clinical data
- **Time-varying Analysis**: Tracks state transitions over time
- **Behavioral Clustering**: Identifies patient behavior patterns
- **Risk Factor Identification**: Focuses on changeable features vs. static demographics

## Validation and Quality Assurance

### Data Quality Checks

1. **Schema Validation**: Ensures required columns are present
2. **Data Type Validation**: Validates date formats and numeric values
3. **Range Validation**: Checks for reasonable value ranges
4. **Missing Data Analysis**: Identifies and reports data gaps

### Label Validation

1. **Cross-validation**: Compares multi-class vs. state-based labels
2. **Demographic Analysis**: Validates against known demographic patterns
3. **Behavioral Consistency**: Checks for logical adherence patterns
4. **Temporal Consistency**: Validates state transition logic

## Performance Considerations

### Memory Usage
- Large datasets (>10K participants) may require chunked processing
- Consider using `dask` for very large datasets
- Monitor memory usage during analysis

### Processing Time
- Multi-class labeling: ~1-2 seconds per 1000 participants
- State-based labeling: ~5-10 seconds per 1000 participants
- Visualization generation: ~30-60 seconds for full analysis

### Optimization Tips
- Use specific person_ids to limit data loading
- Process domains separately for large datasets
- Cache intermediate results for repeated analysis

## Troubleshooting

### Common Issues

1. **Memory Errors**
   ```python
   # Reduce batch size
   data = await connector.load_all_domains(
       person_ids=person_ids[:1000]  # Process in smaller batches
   )
   ```

2. **Date Format Errors**
   ```python
   # Ensure proper date formatting
   start_date = pd.to_datetime('2023-01-01')
   end_date = pd.to_datetime('2023-12-31')
   ```

3. **Missing Data Warnings**
   ```python
   # Check data availability
   summary = connector.get_data_summary(data)
   print(summary)
   ```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Development Setup

1. Clone the repository
2. Install development dependencies
3. Run tests: `python -m pytest tests/`
4. Follow PEP 8 style guidelines

### Adding New Features

1. Create feature branch
2. Add tests for new functionality
3. Update documentation
4. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{clinical_trial_adherence_2024,
  title={Clinical Trial Adherence Analysis System},
  author={Clinical Trial Agent Team},
  year={2024},
  url={https://github.com/your-repo/clinical-trial-agent}
}
```

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review example scripts

---

**Note**: This system is designed for research purposes. Always ensure compliance with data privacy regulations and institutional review board requirements when working with clinical data. 