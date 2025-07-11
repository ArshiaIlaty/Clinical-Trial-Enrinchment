"""
All of Us Dataset Connector

This module provides specialized data connectors for the All of Us Registered Tier Dataset v8,
handling the specific schema and data formats for clinical trial adherence analysis.

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from pathlib import Path
import json

from .base_connector import BaseDataConnector

logger = logging.getLogger(__name__)

class AllOfUsConnector(BaseDataConnector):
    """
    Connector for All of Us Registered Tier Dataset v8.
    
    Handles data loading and preprocessing for:
    - Person demographics
    - Fitbit activity, device, heart rate, and sleep data
    - Survey responses
    - Lab results and clinical observations
    """
    
    def __init__(self, data_directory: Optional[str] = None):
        """Initialize the All of Us connector."""
        super().__init__(
            name="AllOfUsConnector",
            description="Connector for All of Us Registered Tier Dataset v8"
        )
        self.data_directory = Path(data_directory) if data_directory else None
        self.data_cache = {}
        self.schema_info = self._load_schema_info()
    
    def _load_schema_info(self) -> Dict[str, Any]:
        """Load schema information for All of Us dataset."""
        return {
            "person": {
                "required_columns": ["person_id", "gender", "date_of_birth", "race", "ethnicity"],
                "date_columns": ["date_of_birth"],
                "categorical_columns": ["gender", "race", "ethnicity", "sex_at_birth"]
            },
            "fitbit_activity": {
                "required_columns": ["person_id", "date", "steps", "calories_out"],
                "date_columns": ["date"],
                "numeric_columns": ["steps", "calories_out", "activity_calories", "sedentary_minutes", "lightly_active_minutes", "fairly_active_minutes", "very_active_minutes"]
            },
            "fitbit_device": {
                "required_columns": ["person_id", "device_id", "device_date"],
                "date_columns": ["device_date", "last_sync_time"],
                "categorical_columns": ["battery", "device_type", "device_version"]
            },
            "fitbit_heart_rate_level": {
                "required_columns": ["person_id", "date", "avg_rate"],
                "date_columns": ["date"],
                "numeric_columns": ["avg_rate"]
            },
            "fitbit_heart_rate_summary": {
                "required_columns": ["person_id", "date", "zone_name", "minute_in_zone"],
                "date_columns": ["date"],
                "categorical_columns": ["zone_name"],
                "numeric_columns": ["min_heart_rate", "max_heart_rate", "minute_in_zone", "calorie_count"]
            },
            "fitbit_sleep_daily_summary": {
                "required_columns": ["person_id", "sleep_date", "minute_in_bed", "minute_asleep"],
                "date_columns": ["sleep_date"],
                "numeric_columns": ["minute_in_bed", "minute_asleep", "minute_awake", "minute_restless", "minute_deep", "minute_light", "minute_rem"]
            },
            "fitbit_sleep_level": {
                "required_columns": ["person_id", "sleep_date", "level", "duration_in_min"],
                "date_columns": ["sleep_date", "date"],
                "categorical_columns": ["level"],
                "numeric_columns": ["duration_in_min"]
            },
            "survey": {
                "required_columns": ["person_id", "survey_datetime", "survey", "question", "answer"],
                "date_columns": ["survey_datetime"],
                "categorical_columns": ["survey", "question", "answer"]
            }
        }
    
    async def load_all_domains(
        self,
        person_ids: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        domains: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data from multiple All of Us domains.
        
        Args:
            person_ids: List of person IDs to filter by (None for all)
            start_date: Start date for filtering (None for all)
            end_date: End date for filtering (None for all)
            domains: List of domains to load (None for all)
            
        Returns:
            Dictionary with domain names as keys and DataFrames as values
        """
        try:
            if domains is None:
                domains = list(self.schema_info.keys())
            
            data = {}
            
            for domain in domains:
                logger.info(f"Loading {domain} domain...")
                domain_data = await self.load_domain(
                    domain, person_ids, start_date, end_date
                )
                if domain_data is not None and not domain_data.empty:
                    data[domain] = domain_data
                    logger.info(f"Loaded {len(domain_data)} records from {domain}")
                else:
                    logger.warning(f"No data loaded for {domain}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading domains: {str(e)}")
            return {}
    
    async def load_domain(
        self,
        domain: str,
        person_ids: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load data from a specific domain.
        
        Args:
            domain: Domain name (e.g., 'person', 'fitbit_activity')
            person_ids: List of person IDs to filter by
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            DataFrame with domain data or None if error
        """
        try:
            # Check if domain is supported
            if domain not in self.schema_info:
                logger.error(f"Unsupported domain: {domain}")
                return None
            
            # Load data (placeholder implementation)
            # In real implementation, this would load from actual data files
            data = await self._load_domain_data(domain, person_ids, start_date, end_date)
            
            if data is not None and not data.empty:
                # Apply filters
                data = self._apply_filters(data, domain, person_ids, start_date, end_date)
                
                # Validate data
                if self._validate_domain_data(data, domain):
                    # Preprocess data
                    data = self._preprocess_domain_data(data, domain)
                    return data
                else:
                    logger.error(f"Data validation failed for {domain}")
                    return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading domain {domain}: {str(e)}")
            return None
    
    async def _load_domain_data(
        self,
        domain: str,
        person_ids: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load raw data for a domain.
        
        This is a placeholder implementation. In practice, this would:
        1. Load from CSV/Parquet files in the data directory
        2. Query from a database
        3. Call API endpoints
        """
        try:
            # Placeholder: generate sample data based on schema
            if domain == "person":
                return self._generate_sample_person_data(person_ids)
            elif domain == "fitbit_activity":
                return self._generate_sample_fitbit_activity_data(person_ids, start_date, end_date)
            elif domain == "fitbit_device":
                return self._generate_sample_fitbit_device_data(person_ids)
            elif domain == "fitbit_heart_rate_level":
                return self._generate_sample_heart_rate_data(person_ids, start_date, end_date)
            elif domain == "fitbit_sleep_daily_summary":
                return self._generate_sample_sleep_data(person_ids, start_date, end_date)
            elif domain == "survey":
                return self._generate_sample_survey_data(person_ids, start_date, end_date)
            else:
                logger.warning(f"No sample data generator for domain: {domain}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading domain data: {str(e)}")
            return None
    
    def _generate_sample_person_data(self, person_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Generate sample person data."""
        if person_ids is None:
            person_ids = list(range(1000, 1010))  # Sample 10 participants
        
        data = []
        for person_id in person_ids:
            data.append({
                'person_id': person_id,
                'gender_concept_id': np.random.choice([8507, 8532]),  # Male/Female
                'gender': np.random.choice(['Male', 'Female']),
                'date_of_birth': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(6570, 25550)),  # 18-70 years
                'race_concept_id': np.random.choice([8515, 8516, 8527, 8657, 0]),  # White, Black, Asian, etc.
                'race': np.random.choice(['White', 'Black or African American', 'Asian', 'Other', 'Unknown']),
                'ethnicity_concept_id': np.random.choice([38003563, 38003564, 0]),  # Hispanic, Non-Hispanic, Unknown
                'ethnicity': np.random.choice(['Hispanic or Latino', 'Not Hispanic or Latino', 'Unknown']),
                'sex_at_birth_concept_id': np.random.choice([8507, 8532]),
                'sex_at_birth': np.random.choice(['Male', 'Female']),
                'self_reported_category_concept_id': np.random.choice([8515, 8516, 8527, 8657, 0]),
                'self_reported_category': np.random.choice(['White', 'Black or African American', 'Asian', 'Other', 'Unknown'])
            })
        
        return pd.DataFrame(data)
    
    def _generate_sample_fitbit_activity_data(
        self,
        person_ids: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Generate sample Fitbit activity data."""
        if person_ids is None:
            person_ids = list(range(1000, 1010))
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        data = []
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        for person_id in person_ids:
            # Simulate adherence patterns
            adherence_level = np.random.choice(['high', 'medium', 'low'], p=[0.3, 0.5, 0.2])
            
            for date in date_range:
                # Skip some days based on adherence level
                if adherence_level == 'high' and np.random.random() < 0.1:  # 10% missing
                    continue
                elif adherence_level == 'medium' and np.random.random() < 0.3:  # 30% missing
                    continue
                elif adherence_level == 'low' and np.random.random() < 0.6:  # 60% missing
                    continue
                
                # Generate activity data
                base_steps = np.random.normal(8000, 2000)
                if adherence_level == 'high':
                    base_steps = np.random.normal(10000, 1500)
                elif adherence_level == 'low':
                    base_steps = np.random.normal(4000, 1500)
                
                data.append({
                    'person_id': person_id,
                    'date': date,
                    'activity_calories': max(0, np.random.normal(300, 100)),
                    'calories_bmr': np.random.normal(1500, 200),
                    'calories_out': max(0, np.random.normal(2000, 300)),
                    'elevation': max(0, np.random.normal(10, 5)),
                    'fairly_active_minutes': max(0, np.random.normal(20, 10)),
                    'floors': max(0, np.random.normal(8, 4)),
                    'lightly_active_minutes': max(0, np.random.normal(150, 50)),
                    'marginal_calories': max(0, np.random.normal(50, 20)),
                    'sedentary_minutes': max(0, np.random.normal(600, 100)),
                    'steps': max(0, int(base_steps)),
                    'very_active_minutes': max(0, np.random.normal(30, 15))
                })
        
        return pd.DataFrame(data)
    
    def _generate_sample_fitbit_device_data(self, person_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Generate sample Fitbit device data."""
        if person_ids is None:
            person_ids = list(range(1000, 1010))
        
        data = []
        for person_id in person_ids:
            # Generate device data
            device_types = ['Versa 2', 'Charge 4', 'Inspire 2', 'Sense']
            device_type = np.random.choice(device_types)
            
            data.append({
                'person_id': person_id,
                'device_id': f"device_{person_id}",
                'device_date': datetime.now() - timedelta(days=np.random.randint(1, 365)),
                'battery': np.random.choice(['Full', 'Good', 'Low', 'Critical']),
                'battery_level': np.random.randint(10, 100),
                'device_version': f"v{np.random.randint(1, 5)}.{np.random.randint(0, 10)}",
                'device_type': device_type,
                'last_sync_time': datetime.now() - timedelta(hours=np.random.randint(1, 72)),
                'src_id': np.random.randint(1, 100)
            })
        
        return pd.DataFrame(data)
    
    def _generate_sample_heart_rate_data(
        self,
        person_ids: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Generate sample heart rate data."""
        if person_ids is None:
            person_ids = list(range(1000, 1010))
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        data = []
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        for person_id in person_ids:
            # Simulate adherence patterns
            adherence_level = np.random.choice(['high', 'medium', 'low'], p=[0.3, 0.5, 0.2])
            
            for date in date_range:
                # Skip some days based on adherence level
                if adherence_level == 'high' and np.random.random() < 0.15:  # 15% missing
                    continue
                elif adherence_level == 'medium' and np.random.random() < 0.4:  # 40% missing
                    continue
                elif adherence_level == 'low' and np.random.random() < 0.7:  # 70% missing
                    continue
                
                # Generate heart rate data
                base_hr = np.random.normal(70, 10)
                if adherence_level == 'high':
                    base_hr = np.random.normal(65, 8)  # Lower, more consistent HR
                elif adherence_level == 'low':
                    base_hr = np.random.normal(75, 15)  # Higher, more variable HR
                
                data.append({
                    'person_id': person_id,
                    'date': date,
                    'avg_rate': max(40, min(120, int(base_hr)))
                })
        
        return pd.DataFrame(data)
    
    def _generate_sample_sleep_data(
        self,
        person_ids: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Generate sample sleep data."""
        if person_ids is None:
            person_ids = list(range(1000, 1010))
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        data = []
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        for person_id in person_ids:
            # Simulate adherence patterns
            adherence_level = np.random.choice(['high', 'medium', 'low'], p=[0.3, 0.5, 0.2])
            
            for date in date_range:
                # Skip some days based on adherence level
                if adherence_level == 'high' and np.random.random() < 0.2:  # 20% missing
                    continue
                elif adherence_level == 'medium' and np.random.random() < 0.45:  # 45% missing
                    continue
                elif adherence_level == 'low' and np.random.random() < 0.75:  # 75% missing
                    continue
                
                # Generate sleep data
                if adherence_level == 'high':
                    total_sleep = np.random.normal(420, 30)  # 7 hours ± 30 min
                    sleep_efficiency = np.random.normal(0.85, 0.05)  # 85% efficiency
                elif adherence_level == 'medium':
                    total_sleep = np.random.normal(360, 60)  # 6 hours ± 1 hour
                    sleep_efficiency = np.random.normal(0.75, 0.1)  # 75% efficiency
                else:  # low
                    total_sleep = np.random.normal(300, 90)  # 5 hours ± 1.5 hours
                    sleep_efficiency = np.random.normal(0.65, 0.15)  # 65% efficiency
                
                total_sleep = max(180, min(600, int(total_sleep)))  # 3-10 hours
                sleep_efficiency = max(0.3, min(1.0, sleep_efficiency))
                
                time_in_bed = int(total_sleep / sleep_efficiency)
                time_asleep = int(total_sleep)
                time_awake = time_in_bed - time_asleep
                
                # Distribute sleep stages
                deep_sleep = int(time_asleep * 0.2)  # 20% deep sleep
                light_sleep = int(time_asleep * 0.5)  # 50% light sleep
                rem_sleep = int(time_asleep * 0.25)  # 25% REM sleep
                restless = int(time_asleep * 0.05)  # 5% restless
                
                data.append({
                    'person_id': person_id,
                    'sleep_date': date,
                    'is_main_sleep': True,
                    'minute_in_bed': time_in_bed,
                    'minute_asleep': time_asleep,
                    'minute_after_wakeup': 0,
                    'minute_awake': time_awake,
                    'minute_restless': restless,
                    'minute_deep': deep_sleep,
                    'minute_light': light_sleep,
                    'minute_rem': rem_sleep,
                    'minute_wake': time_awake
                })
        
        return pd.DataFrame(data)
    
    def _generate_sample_survey_data(
        self,
        person_ids: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Generate sample survey data."""
        if person_ids is None:
            person_ids = list(range(1000, 1010))
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        data = []
        
        # Define survey types and questions
        surveys = {
            'The Basics': [
                'What is your current age?',
                'What is your gender?',
                'What is your race?',
                'What is your ethnicity?'
            ],
            'COPE': [
                'How often do you feel nervous, anxious, or on edge?',
                'How often do you have trouble relaxing?',
                'How often do you worry too much about different things?'
            ],
            'SDOH': [
                'What is your highest level of education?',
                'What is your current employment status?',
                'What is your household income?'
            ],
            'Lifestyle': [
                'How often do you exercise?',
                'How many hours do you sleep per night?',
                'Do you smoke cigarettes?'
            ]
        }
        
        for person_id in person_ids:
            # Simulate adherence patterns
            adherence_level = np.random.choice(['high', 'medium', 'low'], p=[0.3, 0.5, 0.2])
            
            # Determine number of surveys completed
            if adherence_level == 'high':
                num_surveys = np.random.randint(3, 6)
            elif adherence_level == 'medium':
                num_surveys = np.random.randint(1, 4)
            else:  # low
                num_surveys = np.random.randint(0, 2)
            
            # Generate survey responses
            survey_dates = np.random.choice(
                pd.date_range(start_date, end_date, freq='D'),
                size=num_surveys,
                replace=False
            )
            
            for i, survey_date in enumerate(survey_dates):
                survey_name = np.random.choice(list(surveys.keys()))
                questions = surveys[survey_name]
                
                for question in questions:
                    data.append({
                        'person_id': person_id,
                        'survey_datetime': survey_date,
                        'survey': survey_name,
                        'question_concept_id': f"q_{hash(question) % 10000}",
                        'question': question,
                        'answer_concept_id': f"a_{hash(question) % 10000}",
                        'answer': self._generate_survey_answer(question),
                        'survey_version_concept_id': 1,
                        'survey_version_name': 'v1.0'
                    })
        
        return pd.DataFrame(data)
    
    def _generate_survey_answer(self, question: str) -> str:
        """Generate appropriate survey answers based on question."""
        if 'age' in question.lower():
            return str(np.random.randint(18, 80))
        elif 'gender' in question.lower():
            return np.random.choice(['Male', 'Female', 'Other'])
        elif 'race' in question.lower():
            return np.random.choice(['White', 'Black or African American', 'Asian', 'Other'])
        elif 'ethnicity' in question.lower():
            return np.random.choice(['Hispanic or Latino', 'Not Hispanic or Latino'])
        elif 'education' in question.lower():
            return np.random.choice(['High School', 'Some College', 'Bachelor\'s Degree', 'Graduate Degree'])
        elif 'employment' in question.lower():
            return np.random.choice(['Employed', 'Unemployed', 'Retired', 'Student'])
        elif 'income' in question.lower():
            return np.random.choice(['<$25,000', '$25,000-$50,000', '$50,000-$100,000', '>$100,000'])
        elif 'exercise' in question.lower():
            return np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])
        elif 'sleep' in question.lower():
            return str(np.random.randint(4, 10))
        elif 'smoke' in question.lower():
            return np.random.choice(['Yes', 'No', 'Former smoker'])
        elif 'nervous' in question.lower() or 'anxious' in question.lower():
            return np.random.choice(['Not at all', 'Several days', 'More than half the days', 'Nearly every day'])
        else:
            return np.random.choice(['Yes', 'No', 'Sometimes', 'Not sure'])
    
    def _apply_filters(
        self,
        data: pd.DataFrame,
        domain: str,
        person_ids: Optional[List[int]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Apply filters to the data."""
        filtered_data = data.copy()
        
        # Filter by person IDs
        if person_ids is not None:
            filtered_data = filtered_data[filtered_data['person_id'].isin(person_ids)]
        
        # Filter by date range
        if start_date is not None or end_date is not None:
            date_col = self.schema_info[domain].get('date_columns', [None])[0]
            if date_col and date_col in filtered_data.columns:
                if start_date is not None:
                    filtered_data = filtered_data[filtered_data[date_col] >= start_date]
                if end_date is not None:
                    filtered_data = filtered_data[filtered_data[date_col] <= end_date]
        
        return filtered_data
    
    def _validate_domain_data(self, data: pd.DataFrame, domain: str) -> bool:
        """Validate data for a specific domain."""
        try:
            if data.empty:
                return True  # Empty data is valid
            
            schema = self.schema_info[domain]
            
            # Check required columns
            for col in schema['required_columns']:
                if col not in data.columns:
                    logger.error(f"Missing required column {col} in {domain}")
                    return False
            
            # Check data types for date columns
            for col in schema.get('date_columns', []):
                if col in data.columns:
                    try:
                        pd.to_datetime(data[col])
                    except:
                        logger.error(f"Invalid date format in column {col} of {domain}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating {domain} data: {str(e)}")
            return False
    
    def _preprocess_domain_data(self, data: pd.DataFrame, domain: str) -> pd.DataFrame:
        """Preprocess data for a specific domain."""
        try:
            processed_data = data.copy()
            schema = self.schema_info[domain]
            
            # Convert date columns
            for col in schema.get('date_columns', []):
                if col in processed_data.columns:
                    processed_data[col] = pd.to_datetime(processed_data[col])
            
            # Handle missing values
            for col in schema.get('numeric_columns', []):
                if col in processed_data.columns:
                    processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
            
            # Sort by date if available
            date_cols = schema.get('date_columns', [])
            if date_cols:
                processed_data = processed_data.sort_values(date_cols[0])
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing {domain} data: {str(e)}")
            return data
    
    def get_data_summary(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get summary statistics for loaded data."""
        summary = {}
        
        for domain, df in data.items():
            if df is not None and not df.empty:
                summary[domain] = {
                    'record_count': len(df),
                    'person_count': df['person_id'].nunique() if 'person_id' in df.columns else 0,
                    'date_range': self._get_date_range(df, domain),
                    'missing_values': df.isnull().sum().to_dict()
                }
            else:
                summary[domain] = {
                    'record_count': 0,
                    'person_count': 0,
                    'date_range': None,
                    'missing_values': {}
                }
        
        return summary
    
    def _get_date_range(self, df: pd.DataFrame, domain: str) -> Optional[Dict[str, str]]:
        """Get date range for a domain."""
        date_cols = self.schema_info[domain].get('date_columns', [])
        
        for col in date_cols:
            if col in df.columns:
                try:
                    dates = pd.to_datetime(df[col])
                    return {
                        'start': dates.min().strftime('%Y-%m-%d'),
                        'end': dates.max().strftime('%Y-%m-%d')
                    }
                except:
                    continue
        
        return None
    
    def export_data_summary(self, data: Dict[str, pd.DataFrame], output_path: str) -> None:
        """Export data summary to JSON file."""
        try:
            summary = self.get_data_summary(data)
            
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Data summary exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting data summary: {str(e)}")
            raise 