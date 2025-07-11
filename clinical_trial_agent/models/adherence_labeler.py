"""
Adherence Labeling System for Clinical Trial Data

This module implements comprehensive adherence labeling strategies based on:
- Multi-class labels (High/Medium/Low adherence)
- State-based labels (Active/Inactive/Exit)
- All of Us Registered Tier Dataset v8 schema
- Previous research on blood pressure monitoring adherence
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

class AdherenceLevel(Enum):
    """Enumeration for adherence levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class AdherenceState(Enum):
    """Enumeration for adherence states."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXIT = "exit"

@dataclass
class AdherenceConfig:
    """Configuration for adherence labeling."""
    # Fitbit activity thresholds
    fitbit_coverage_high: float = 0.8  # 80% days with Fitbit data
    fitbit_coverage_medium: float = 0.4  # 40% days with Fitbit data
    fitbit_coverage_low: float = 0.2  # 20% days with Fitbit data
    
    # Survey completion thresholds
    survey_completion_high: int = 3  # 3+ surveys completed
    survey_completion_medium: int = 1  # 1-2 surveys completed
    
    # Time gap thresholds (in days)
    inactive_gap: int = 7  # 7+ days without Fitbit data
    exit_gap: int = 180  # 180+ days without any data
    
    # Lab visit thresholds
    lab_visit_expected_months: int = 6  # Expected lab visits every 6 months
    lab_visit_tolerance_days: int = 30  # Tolerance for lab visit timing
    
    # Sleep quality thresholds
    sleep_fragmentation_threshold: float = 0.3  # High fragmentation threshold
    sleep_efficiency_threshold: float = 0.8  # Low efficiency threshold
    
    # Activity consistency thresholds
    activity_std_threshold: float = 0.4  # High variability threshold
    min_steps_threshold: int = 5000  # Minimum daily steps
    
    # Heart rate variability thresholds
    hr_variability_low: float = 30.0  # Low HRV threshold
    hr_variability_high: float = 60.0  # High HRV threshold

class AllOfUsAdherenceLabeler:
    """
    Comprehensive adherence labeling system for All of Us dataset.
    
    Implements both multi-class and state-based labeling strategies based on:
    - Fitbit activity patterns and frequency
    - Survey completion rates
    - Lab visit attendance
    - Behavioral regularity and time gaps
    """
    
    def __init__(self, config: Optional[AdherenceConfig] = None):
        """Initialize the adherence labeler."""
        self.config = config or AdherenceConfig()
        self.label_history = {}
        
    def create_multi_class_labels(
        self,
        person_data: Dict[str, pd.DataFrame],
        study_period_days: int = 365
    ) -> pd.DataFrame:
        """
        Create multi-class adherence labels (High/Medium/Low).
        
        Args:
            person_data: Dictionary containing DataFrames for each domain
            study_period_days: Total study period in days
            
        Returns:
            DataFrame with person_id and adherence_level
        """
        logger.info("Creating multi-class adherence labels...")
        
        labels = []
        
        for person_id in person_data.get('person', pd.DataFrame()).get('person_id', []):
            adherence_score = self._calculate_multi_class_score(
                person_id, person_data, study_period_days
            )
            
            # Determine adherence level
            if adherence_score >= 0.8:
                level = AdherenceLevel.HIGH
            elif adherence_score >= 0.4:
                level = AdherenceLevel.MEDIUM
            else:
                level = AdherenceLevel.LOW
            
            labels.append({
                'person_id': person_id,
                'adherence_level': level.value,
                'adherence_score': adherence_score
            })
        
        return pd.DataFrame(labels)
    
    def create_state_based_labels(
        self,
        person_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Create state-based adherence labels (Active/Inactive/Exit).
        
        Args:
            person_data: Dictionary containing DataFrames for each domain
            start_date: Study start date
            end_date: Study end date
            
        Returns:
            DataFrame with person_id, date, and adherence_state
        """
        logger.info("Creating state-based adherence labels...")
        
        state_labels = []
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        for person_id in person_data.get('person', pd.DataFrame()).get('person_id', []):
            person_states = self._calculate_person_states(
                person_id, person_data, date_range
            )
            state_labels.extend(person_states)
        
        return pd.DataFrame(state_labels)
    
    def _calculate_multi_class_score(
        self,
        person_id: int,
        person_data: Dict[str, pd.DataFrame],
        study_period_days: int
    ) -> float:
        """Calculate adherence score for multi-class labeling."""
        scores = []
        
        # 1. Fitbit Activity Coverage (40% weight)
        fitbit_score = self._calculate_fitbit_coverage_score(person_id, person_data, study_period_days)
        scores.append(('fitbit_coverage', fitbit_score, 0.4))
        
        # 2. Survey Completion Rate (25% weight)
        survey_score = self._calculate_survey_completion_score(person_id, person_data)
        scores.append(('survey_completion', survey_score, 0.25))
        
        # 3. Lab Visit Adherence (20% weight)
        lab_score = self._calculate_lab_visit_score(person_id, person_data, study_period_days)
        scores.append(('lab_visits', lab_score, 0.2))
        
        # 4. Behavioral Consistency (15% weight)
        consistency_score = self._calculate_behavioral_consistency_score(person_id, person_data)
        scores.append(('behavioral_consistency', consistency_score, 0.15))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in scores)
        
        logger.debug(f"Person {person_id} adherence scores: {dict([(name, score) for name, score, _ in scores])}")
        
        return total_score
    
    def _calculate_fitbit_coverage_score(
        self,
        person_id: int,
        person_data: Dict[str, pd.DataFrame],
        study_period_days: int
    ) -> float:
        """Calculate Fitbit data coverage score."""
        try:
            # Get Fitbit activity data
            fitbit_activity = person_data.get('fitbit_activity', pd.DataFrame())
            person_activity = fitbit_activity[fitbit_activity['person_id'] == person_id]
            
            if person_activity.empty:
                return 0.0
            
            # Calculate coverage
            unique_days = person_activity['date'].nunique()
            coverage_rate = unique_days / study_period_days
            
            # Get device sync data
            fitbit_device = person_data.get('fitbit_device', pd.DataFrame())
            person_device = fitbit_device[fitbit_device['person_id'] == person_id]
            
            # Check for device issues
            device_score = 1.0
            if not person_device.empty:
                # Check battery status and sync issues
                low_battery_days = person_device[person_device['battery_level'] < 20].shape[0]
                device_score = max(0.5, 1.0 - (low_battery_days / len(person_device)))
            
            # Combine coverage and device score
            final_score = coverage_rate * device_score
            
            logger.debug(f"Person {person_id} Fitbit coverage: {coverage_rate:.3f}, device score: {device_score:.3f}")
            
            return min(1.0, final_score)
            
        except Exception as e:
            logger.error(f"Error calculating Fitbit coverage score: {str(e)}")
            return 0.0
    
    def _calculate_survey_completion_score(
        self,
        person_id: int,
        person_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Calculate survey completion score."""
        try:
            # Get survey data
            survey_data = person_data.get('survey', pd.DataFrame())
            person_surveys = survey_data[survey_data['person_id'] == person_id]
            
            if person_surveys.empty:
                return 0.0
            
            # Count unique surveys completed
            unique_surveys = person_surveys['survey'].nunique()
            
            # Calculate completion rate based on expected surveys
            # Assuming key surveys: Basics, COPE, SDOH, etc.
            expected_surveys = 5  # Adjust based on study requirements
            completion_rate = min(1.0, unique_surveys / expected_surveys)
            
            # Check survey frequency (time between surveys)
            if len(person_surveys) > 1:
                person_surveys_sorted = person_surveys.sort_values('survey_datetime')
                time_gaps = person_surveys_sorted['survey_datetime'].diff().dt.days
                avg_gap = time_gaps.mean()
                
                # Penalize very long gaps (>90 days)
                gap_penalty = max(0.5, 1.0 - (avg_gap / 90.0))
                completion_rate *= gap_penalty
            
            logger.debug(f"Person {person_id} survey completion: {completion_rate:.3f}")
            
            return completion_rate
            
        except Exception as e:
            logger.error(f"Error calculating survey completion score: {str(e)}")
            return 0.0
    
    def _calculate_lab_visit_score(
        self,
        person_id: int,
        person_data: Dict[str, pd.DataFrame],
        study_period_days: int
    ) -> float:
        """Calculate lab visit adherence score."""
        try:
            # This would typically come from observations or visits domain
            # For now, we'll use a placeholder implementation
            
            # Expected lab visits based on study period
            expected_visits = max(1, study_period_days // (self.config.lab_visit_expected_months * 30))
            
            # Placeholder: assume 70% of participants have regular lab visits
            # In real implementation, this would query actual lab/visit data
            lab_score = 0.7
            
            logger.debug(f"Person {person_id} lab visit score: {lab_score:.3f}")
            
            return lab_score
            
        except Exception as e:
            logger.error(f"Error calculating lab visit score: {str(e)}")
            return 0.0
    
    def _calculate_behavioral_consistency_score(
        self,
        person_id: int,
        person_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Calculate behavioral consistency score."""
        try:
            consistency_scores = []
            
            # 1. Activity consistency (steps variability)
            fitbit_activity = person_data.get('fitbit_activity', pd.DataFrame())
            person_activity = fitbit_activity[fitbit_activity['person_id'] == person_id]
            
            if not person_activity.empty and 'steps' in person_activity.columns:
                steps = person_activity['steps'].dropna()
                if len(steps) > 1:
                    steps_cv = steps.std() / steps.mean()  # Coefficient of variation
                    activity_consistency = max(0.0, 1.0 - steps_cv)
                    consistency_scores.append(activity_consistency)
            
            # 2. Sleep consistency
            fitbit_sleep = person_data.get('fitbit_sleep_daily_summary', pd.DataFrame())
            person_sleep = fitbit_sleep[fitbit_sleep['person_id'] == person_id]
            
            if not person_sleep.empty and 'minute_asleep' in person_sleep.columns:
                sleep_duration = person_sleep['minute_asleep'].dropna()
                if len(sleep_duration) > 1:
                    sleep_cv = sleep_duration.std() / sleep_duration.mean()
                    sleep_consistency = max(0.0, 1.0 - sleep_cv)
                    consistency_scores.append(sleep_consistency)
            
            # 3. Heart rate variability
            fitbit_hr = person_data.get('fitbit_heart_rate_level', pd.DataFrame())
            person_hr = fitbit_hr[fitbit_hr['person_id'] == person_id]
            
            if not person_hr.empty and 'avg_rate' in person_hr.columns:
                hr_values = person_hr['avg_rate'].dropna()
                if len(hr_values) > 1:
                    hr_cv = hr_values.std() / hr_values.mean()
                    hr_consistency = max(0.0, 1.0 - hr_cv)
                    consistency_scores.append(hr_consistency)
            
            # Calculate average consistency score
            if consistency_scores:
                avg_consistency = np.mean(consistency_scores)
            else:
                avg_consistency = 0.5  # Default score if no data available
            
            logger.debug(f"Person {person_id} behavioral consistency: {avg_consistency:.3f}")
            
            return avg_consistency
            
        except Exception as e:
            logger.error(f"Error calculating behavioral consistency score: {str(e)}")
            return 0.5
    
    def _calculate_person_states(
        self,
        person_id: int,
        person_data: Dict[str, pd.DataFrame],
        date_range: pd.DatetimeIndex
    ) -> List[Dict[str, Union[int, str, datetime]]]:
        """Calculate state transitions for a person over time."""
        states = []
        
        for date in date_range:
            state = self._determine_daily_state(person_id, person_data, date)
            states.append({
                'person_id': person_id,
                'date': date,
                'adherence_state': state.value
            })
        
        return states
    
    def _determine_daily_state(
        self,
        person_id: int,
        person_data: Dict[str, pd.DataFrame],
        date: datetime
    ) -> AdherenceState:
        """Determine adherence state for a specific date."""
        try:
            # Check for recent Fitbit data
            fitbit_activity = person_data.get('fitbit_activity', pd.DataFrame())
            person_activity = fitbit_activity[fitbit_activity['person_id'] == person_id]
            
            # Check if there's Fitbit data within the last 7 days
            recent_activity = person_activity[
                person_activity['date'] >= (date - timedelta(days=self.config.inactive_gap))
            ]
            
            has_recent_fitbit = not recent_activity.empty
            
            # Check for recent surveys
            survey_data = person_data.get('survey', pd.DataFrame())
            person_surveys = survey_data[survey_data['person_id'] == person_id]
            
            recent_surveys = person_surveys[
                person_surveys['survey_datetime'] >= (date - timedelta(days=90))
            ]
            
            has_recent_surveys = not recent_surveys.empty
            
            # Check for recent lab visits (placeholder)
            has_recent_labs = True  # Placeholder - would check actual lab data
            
            # Determine state
            if has_recent_fitbit and (has_recent_surveys or has_recent_labs):
                return AdherenceState.ACTIVE
            elif has_recent_fitbit or has_recent_surveys or has_recent_labs:
                return AdherenceState.INACTIVE
            else:
                # Check for exit condition (no data for 180+ days)
                old_activity = person_activity[
                    person_activity['date'] >= (date - timedelta(days=self.config.exit_gap))
                ]
                old_surveys = person_surveys[
                    person_surveys['survey_datetime'] >= (date - timedelta(days=self.config.exit_gap))
                ]
                
                if old_activity.empty and old_surveys.empty:
                    return AdherenceState.EXIT
                else:
                    return AdherenceState.INACTIVE
                    
        except Exception as e:
            logger.error(f"Error determining daily state: {str(e)}")
            return AdherenceState.INACTIVE
    
    def analyze_adherence_patterns(
        self,
        labels_df: pd.DataFrame,
        person_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Analyze adherence patterns and generate insights.
        
        Args:
            labels_df: DataFrame with adherence labels
            person_data: Dictionary containing DataFrames for each domain
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing adherence patterns...")
        
        analysis = {
            'summary_stats': self._calculate_summary_stats(labels_df),
            'demographic_analysis': self._analyze_demographics(labels_df, person_data),
            'behavioral_patterns': self._analyze_behavioral_patterns(labels_df, person_data),
            'risk_factors': self._identify_risk_factors(labels_df, person_data)
        }
        
        return analysis
    
    def _calculate_summary_stats(self, labels_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for adherence labels."""
        if 'adherence_level' in labels_df.columns:
            # Multi-class labels
            level_counts = labels_df['adherence_level'].value_counts()
            level_percentages = (level_counts / len(labels_df)) * 100
            
            return {
                'total_participants': len(labels_df),
                'level_distribution': level_counts.to_dict(),
                'level_percentages': level_percentages.to_dict(),
                'mean_adherence_score': labels_df.get('adherence_score', pd.Series()).mean()
            }
        elif 'adherence_state' in labels_df.columns:
            # State-based labels
            state_counts = labels_df['adherence_state'].value_counts()
            state_percentages = (state_counts / len(labels_df)) * 100
            
            return {
                'total_observations': len(labels_df),
                'state_distribution': state_counts.to_dict(),
                'state_percentages': state_percentages.to_dict()
            }
        else:
            return {'error': 'Unknown label format'}
    
    def _analyze_demographics(
        self,
        labels_df: pd.DataFrame,
        person_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Analyze adherence by demographic factors."""
        try:
            person_df = person_data.get('person', pd.DataFrame())
            if person_df.empty:
                return {'error': 'No person data available'}
            
            # Merge labels with person data
            merged_df = labels_df.merge(person_df, on='person_id', how='left')
            
            analysis = {}
            
            # Analyze by age
            if 'date_of_birth' in merged_df.columns:
                merged_df['age'] = (pd.Timestamp.now() - pd.to_datetime(merged_df['date_of_birth'])).dt.days / 365.25
                age_bins = [0, 45, 65, 100]
                age_labels = ['<45', '45-65', '>65']
                merged_df['age_group'] = pd.cut(merged_df['age'], bins=age_bins, labels=age_labels)
                
                age_analysis = merged_df.groupby('age_group')['adherence_level'].value_counts(normalize=True)
                analysis['age_analysis'] = age_analysis.to_dict()
            
            # Analyze by gender
            if 'gender' in merged_df.columns:
                gender_analysis = merged_df.groupby('gender')['adherence_level'].value_counts(normalize=True)
                analysis['gender_analysis'] = gender_analysis.to_dict()
            
            # Analyze by race/ethnicity
            if 'race' in merged_df.columns:
                race_analysis = merged_df.groupby('race')['adherence_level'].value_counts(normalize=True)
                analysis['race_analysis'] = race_analysis.to_dict()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing demographics: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_behavioral_patterns(
        self,
        labels_df: pd.DataFrame,
        person_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Analyze behavioral patterns associated with adherence."""
        try:
            patterns = {}
            
            # Analyze Fitbit activity patterns
            fitbit_activity = person_data.get('fitbit_activity', pd.DataFrame())
            if not fitbit_activity.empty:
                # Merge with labels
                merged_activity = labels_df.merge(fitbit_activity, on='person_id', how='left')
                
                # Activity patterns by adherence level
                if 'adherence_level' in merged_activity.columns and 'steps' in merged_activity.columns:
                    activity_by_level = merged_activity.groupby('adherence_level')['steps'].agg(['mean', 'std', 'count'])
                    patterns['activity_by_adherence'] = activity_by_level.to_dict()
            
            # Analyze sleep patterns
            fitbit_sleep = person_data.get('fitbit_sleep_daily_summary', pd.DataFrame())
            if not fitbit_sleep.empty:
                merged_sleep = labels_df.merge(fitbit_sleep, on='person_id', how='left')
                
                if 'adherence_level' in merged_sleep.columns and 'minute_asleep' in merged_sleep.columns:
                    sleep_by_level = merged_sleep.groupby('adherence_level')['minute_asleep'].agg(['mean', 'std', 'count'])
                    patterns['sleep_by_adherence'] = sleep_by_level.to_dict()
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing behavioral patterns: {str(e)}")
            return {'error': str(e)}
    
    def _identify_risk_factors(
        self,
        labels_df: pd.DataFrame,
        person_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Identify risk factors for low adherence."""
        try:
            risk_factors = {}
            
            # Merge all relevant data
            person_df = person_data.get('person', pd.DataFrame())
            fitbit_activity = person_data.get('fitbit_activity', pd.DataFrame())
            fitbit_device = person_data.get('fitbit_device', pd.DataFrame())
            
            # Create comprehensive dataset
            merged_df = labels_df.merge(person_df, on='person_id', how='left')
            
            # Identify low adherence participants
            low_adherence = merged_df[merged_df['adherence_level'] == 'low']
            high_adherence = merged_df[merged_df['adherence_level'] == 'high']
            
            # Compare characteristics
            if not low_adherence.empty and not high_adherence.empty:
                # Age comparison
                if 'date_of_birth' in merged_df.columns:
                    low_adherence['age'] = (pd.Timestamp.now() - pd.to_datetime(low_adherence['date_of_birth'])).dt.days / 365.25
                    high_adherence['age'] = (pd.Timestamp.now() - pd.to_datetime(high_adherence['date_of_birth'])).dt.days / 365.25
                    
                    risk_factors['age_comparison'] = {
                        'low_adherence_mean_age': low_adherence['age'].mean(),
                        'high_adherence_mean_age': high_adherence['age'].mean(),
                        'age_difference': high_adherence['age'].mean() - low_adherence['age'].mean()
                    }
                
                # Gender comparison
                if 'gender' in merged_df.columns:
                    gender_risk = low_adherence['gender'].value_counts(normalize=True)
                    risk_factors['gender_risk'] = gender_risk.to_dict()
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {str(e)}")
            return {'error': str(e)}
    
    def export_labels(
        self,
        labels_df: pd.DataFrame,
        output_path: str,
        format: str = 'csv'
    ) -> None:
        """Export adherence labels to file."""
        try:
            if format.lower() == 'csv':
                labels_df.to_csv(output_path, index=False)
            elif format.lower() == 'parquet':
                labels_df.to_parquet(output_path, index=False)
            elif format.lower() == 'json':
                labels_df.to_json(output_path, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Adherence labels exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting labels: {str(e)}")
            raise
    
    def get_labeling_summary(self) -> Dict[str, Any]:
        """Get summary of labeling configuration and history."""
        return {
            'config': {
                'fitbit_coverage_thresholds': {
                    'high': self.config.fitbit_coverage_high,
                    'medium': self.config.fitbit_coverage_medium,
                    'low': self.config.fitbit_coverage_low
                },
                'survey_completion_thresholds': {
                    'high': self.config.survey_completion_high,
                    'medium': self.config.survey_completion_medium
                },
                'time_gap_thresholds': {
                    'inactive': self.config.inactive_gap,
                    'exit': self.config.exit_gap
                }
            },
            'label_history': self.label_history
        } 