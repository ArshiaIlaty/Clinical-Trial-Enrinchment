#!/usr/bin/env python3
"""
Test Dataset Generator for Clinical Trial Adherence Agent

This script generates realistic test data for the clinical trial adherence prediction system.
It creates multimodal data including wearable device data, EHR data, survey responses,
and adherence labels for testing the agent's capabilities.
"""

import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import os

class ClinicalTrialDatasetGenerator:
    """Generates realistic clinical trial data for testing adherence prediction."""
    
    def __init__(self, num_patients: int = 100, trial_duration_days: int = 90):
        self.num_patients = num_patients
        self.trial_duration_days = trial_duration_days
        self.start_date = datetime(2024, 1, 1)
        self.end_date = self.start_date + timedelta(days=trial_duration_days)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Adherence categories
        self.adherence_categories = {
            'high': {'min_score': 0.8, 'max_score': 1.0, 'probability': 0.3},
            'medium': {'min_score': 0.6, 'max_score': 0.79, 'probability': 0.4},
            'low': {'min_score': 0.0, 'max_score': 0.59, 'probability': 0.3}
        }
        
        # Patient demographics
        self.age_ranges = [(18, 30), (31, 45), (46, 60), (61, 75), (76, 90)]
        self.genders = ['M', 'F', 'Other']
        self.conditions = [
            'Type 2 Diabetes', 'Hypertension', 'Depression', 'Anxiety',
            'Chronic Pain', 'Asthma', 'Heart Disease', 'Obesity'
        ]
        
    def generate_patient_demographics(self) -> pd.DataFrame:
        """Generate patient demographic information."""
        patients = []
        
        for i in range(self.num_patients):
            patient = {
                'patient_id': f'P{i+1:04d}',
                'age': random.randint(*random.choice(self.age_ranges)),
                'gender': random.choice(self.genders),
                'primary_condition': random.choice(self.conditions),
                'enrollment_date': self.start_date + timedelta(days=random.randint(0, 30)),
                'adherence_category': self._assign_adherence_category(),
                'true_adherence_score': self._generate_true_adherence_score()
            }
            patients.append(patient)
        
        return pd.DataFrame(patients)
    
    def _assign_adherence_category(self) -> str:
        """Assign adherence category based on probabilities."""
        rand = random.random()
        cumulative = 0
        for category, params in self.adherence_categories.items():
            cumulative += params['probability']
            if rand <= cumulative:
                return category
        return 'medium'  # fallback
    
    def _generate_true_adherence_score(self) -> float:
        """Generate true adherence score based on category."""
        category = self._assign_adherence_category()
        params = self.adherence_categories[category]
        return random.uniform(params['min_score'], params['max_score'])
    
    def generate_wearable_data(self, patient_id: str, adherence_score: float) -> Dict[str, Any]:
        """Generate realistic wearable device data."""
        # Base values that correlate with adherence
        base_activity = 8000 + (adherence_score * 4000)  # More adherent = more active
        base_sleep_quality = 0.7 + (adherence_score * 0.2)  # More adherent = better sleep
        
        # Add some noise and temporal patterns
        daily_data = []
        current_date = self.start_date
        
        while current_date <= self.end_date:
            # Weekend effect
            weekend_factor = 0.8 if current_date.weekday() >= 5 else 1.0
            
            # Random daily variation
            daily_variation = random.uniform(0.8, 1.2)
            
            # Generate daily metrics
            steps = int(base_activity * weekend_factor * daily_variation + random.uniform(-500, 500))
            calories = steps * 0.04 + random.uniform(-100, 100)
            heart_rate_resting = 65 + random.uniform(-5, 5) + (1 - adherence_score) * 10
            heart_rate_variability = 45 + random.uniform(-10, 10) + adherence_score * 15
            
            sleep_duration = 7.5 + random.uniform(-1, 1)
            sleep_quality = min(1.0, base_sleep_quality * daily_variation + random.uniform(-0.1, 0.1))
            
            daily_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'steps': max(0, steps),
                'calories': max(0, calories),
                'heart_rate_resting': round(heart_rate_resting, 1),
                'heart_rate_variability': round(heart_rate_variability, 1),
                'sleep_duration': round(sleep_duration, 1),
                'sleep_quality': round(sleep_quality, 2),
                'active_minutes': int(steps / 100 + random.uniform(-10, 10))
            })
            
            current_date += timedelta(days=1)
        
        return {
            'patient_id': patient_id,
            'device_type': random.choice(['fitbit', 'apple_watch', 'garmin']),
            'daily_data': daily_data,
            'summary_stats': {
                'avg_steps': round(np.mean([d['steps'] for d in daily_data])),
                'avg_sleep_duration': round(np.mean([d['sleep_duration'] for d in daily_data]), 1),
                'avg_sleep_quality': round(np.mean([d['sleep_quality'] for d in daily_data]), 2),
                'avg_heart_rate': round(np.mean([d['heart_rate_resting'] for d in daily_data]), 1),
                'avg_heart_rate_variability': round(np.mean([d['heart_rate_variability'] for d in daily_data]), 1)
            }
        }
    
    def generate_ehr_data(self, patient_id: str, adherence_score: float) -> Dict[str, Any]:
        """Generate realistic EHR data."""
        # Number of visits correlates with adherence
        num_visits = max(1, int(3 + adherence_score * 4 + random.uniform(-1, 1)))
        
        visits = []
        medications = []
        lab_results = []
        vitals = []
        clinical_notes = []
        
        # Generate visits
        for i in range(num_visits):
            visit_date = self.start_date + timedelta(days=random.randint(0, self.trial_duration_days))
            attended = random.random() < (0.7 + adherence_score * 0.2)  # More adherent = more likely to attend
            
            visits.append({
                'visit_id': f'V{i+1:03d}',
                'date': visit_date.strftime('%Y-%m-%d'),
                'type': random.choice(['screening', 'baseline', 'follow_up', 'final']),
                'attended': attended,
                'notes': f"Patient {'attended' if attended else 'missed'} {visit_date.strftime('%B %d, %Y')} visit."
            })
        
        # Generate medications
        num_medications = random.randint(1, 4)
        for i in range(num_medications):
            med_names = ['Lisinopril', 'Metformin', 'Atorvastatin', 'Amlodipine', 'Sertraline', 'Ibuprofen']
            medication = {
                'medication_id': f'M{i+1:03d}',
                'name': random.choice(med_names),
                'dosage': f"{random.randint(5, 50)}mg",
                'frequency': random.choice(['daily', 'twice daily', 'as needed']),
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'status': 'active',
                'adherence_rate': adherence_score + random.uniform(-0.1, 0.1)
            }
            medications.append(medication)
        
        # Generate lab results
        num_labs = random.randint(2, 6)
        for i in range(num_labs):
            lab_date = self.start_date + timedelta(days=random.randint(0, self.trial_duration_days))
            lab_tests = [
                {'test': 'Hemoglobin A1c', 'unit': '%', 'normal_range': '4.0-5.6'},
                {'test': 'Blood Glucose', 'unit': 'mg/dL', 'normal_range': '70-100'},
                {'test': 'Cholesterol', 'unit': 'mg/dL', 'normal_range': '<200'},
                {'test': 'Blood Pressure', 'unit': 'mmHg', 'normal_range': '<120/80'}
            ]
            test = random.choice(lab_tests)
            
            lab_results.append({
                'lab_id': f'L{i+1:03d}',
                'test': test['test'],
                'value': self._generate_lab_value(test['test']),
                'unit': test['unit'],
                'date': lab_date.strftime('%Y-%m-%d'),
                'reference_range': test['normal_range']
            })
        
        # Generate vitals
        num_vitals = random.randint(3, 8)
        for i in range(num_vitals):
            vital_date = self.start_date + timedelta(days=random.randint(0, self.trial_duration_days))
            vitals.append({
                'vital_id': f'VT{i+1:03d}',
                'type': random.choice(['Blood Pressure', 'Heart Rate', 'Temperature', 'Weight']),
                'value': self._generate_vital_value(),
                'date': vital_date.strftime('%Y-%m-%d')
            })
        
        # Generate clinical notes
        num_notes = random.randint(1, 3)
        for i in range(num_notes):
            note_date = self.start_date + timedelta(days=random.randint(0, self.trial_duration_days))
            clinical_notes.append({
                'note_id': f'N{i+1:03d}',
                'type': 'Progress Note',
                'date': note_date.strftime('%Y-%m-%d'),
                'content': self._generate_clinical_note(adherence_score),
                'author': f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'])}"
            })
        
        return {
            'patient_id': patient_id,
            'visits': visits,
            'medications': medications,
            'lab_results': lab_results,
            'vitals': vitals,
            'clinical_notes': clinical_notes,
            'summary_stats': {
                'total_visits': len(visits),
                'attended_visits': sum(1 for v in visits if v['attended']),
                'total_medications': len(medications),
                'avg_medication_adherence': round(np.mean([m['adherence_rate'] for m in medications]), 2)
            }
        }
    
    def _generate_lab_value(self, test_name: str) -> Any:
        """Generate realistic lab values."""
        if 'A1c' in test_name:
            return round(random.uniform(5.0, 9.0), 1)
        elif 'Glucose' in test_name:
            return random.randint(70, 200)
        elif 'Cholesterol' in test_name:
            return random.randint(150, 300)
        elif 'Blood Pressure' in test_name:
            systolic = random.randint(110, 160)
            diastolic = random.randint(70, 100)
            return f"{systolic}/{diastolic}"
        else:
            return random.randint(50, 150)
    
    def _generate_vital_value(self) -> Any:
        """Generate realistic vital signs."""
        vital_type = random.choice(['Blood Pressure', 'Heart Rate', 'Temperature', 'Weight'])
        
        if vital_type == 'Blood Pressure':
            systolic = random.randint(110, 160)
            diastolic = random.randint(70, 100)
            return f"{systolic}/{diastolic}"
        elif vital_type == 'Heart Rate':
            return random.randint(60, 100)
        elif vital_type == 'Temperature':
            return round(random.uniform(97.0, 99.5), 1)
        elif vital_type == 'Weight':
            return random.randint(120, 250)
        else:
            return random.randint(50, 150)
    
    def _generate_clinical_note(self, adherence_score: float) -> str:
        """Generate realistic clinical notes."""
        if adherence_score > 0.8:
            notes = [
                "Patient reports excellent adherence to medication regimen.",
                "Patient is highly motivated and following treatment plan well.",
                "No concerns about medication compliance. Patient is doing well.",
                "Patient demonstrates strong commitment to health goals."
            ]
        elif adherence_score > 0.6:
            notes = [
                "Patient reports mostly good adherence with occasional missed doses.",
                "Patient is generally compliant but could improve consistency.",
                "Some challenges with medication timing but overall good progress.",
                "Patient shows improvement in adherence over time."
            ]
        else:
            notes = [
                "Patient reports difficulty maintaining medication schedule.",
                "Concerns about medication adherence. Patient needs support.",
                "Patient has missed several doses. Discussed strategies for improvement.",
                "Adherence issues identified. Patient education provided."
            ]
        
        return random.choice(notes)
    
    def generate_survey_data(self, patient_id: str, adherence_score: float) -> Dict[str, Any]:
        """Generate realistic survey response data."""
        surveys = []
        
        # Generate multiple survey responses over time
        num_surveys = random.randint(3, 8)
        for i in range(num_surveys):
            survey_date = self.start_date + timedelta(days=random.randint(0, self.trial_duration_days))
            
            # Adherence survey
            adherence_responses = [
                {
                    'question_id': 'adherence_1',
                    'question': 'How often did you take your medication as prescribed?',
                    'response': self._get_adherence_response(adherence_score),
                    'score': self._get_adherence_score(adherence_score),
                    'date': survey_date.strftime('%Y-%m-%d')
                },
                {
                    'question_id': 'adherence_2',
                    'question': 'Did you miss any doses in the past week?',
                    'response': 'No' if adherence_score > 0.7 else 'Yes, 1-2 doses',
                    'score': 5 if adherence_score > 0.7 else 3,
                    'date': survey_date.strftime('%Y-%m-%d')
                }
            ]
            
            # Symptom survey
            symptom_responses = [
                {
                    'question_id': 'symptom_1',
                    'question': 'How would you rate your overall symptoms?',
                    'response': self._get_symptom_response(adherence_score),
                    'score': self._get_symptom_score(adherence_score),
                    'date': survey_date.strftime('%Y-%m-%d')
                },
                {
                    'question_id': 'symptom_2',
                    'question': 'Are you experiencing any side effects?',
                    'response': 'None' if adherence_score > 0.6 else 'Mild symptoms',
                    'score': 5 if adherence_score > 0.6 else 3,
                    'date': survey_date.strftime('%Y-%m-%d')
                }
            ]
            
            # Quality of life survey
            qol_responses = [
                {
                    'question_id': 'qol_1',
                    'question': 'How would you rate your physical health?',
                    'response': self._get_qol_response(adherence_score),
                    'score': self._get_qol_score(adherence_score),
                    'date': survey_date.strftime('%Y-%m-%d')
                },
                {
                    'question_id': 'qol_2',
                    'question': 'How would you rate your mental well-being?',
                    'response': self._get_qol_response(adherence_score),
                    'score': self._get_qol_score(adherence_score),
                    'date': survey_date.strftime('%Y-%m-%d')
                }
            ]
            
            surveys.append({
                'survey_id': f'S{i+1:03d}',
                'survey_type': 'comprehensive_assessment',
                'date': survey_date.strftime('%Y-%m-%d'),
                'responses': adherence_responses + symptom_responses + qol_responses,
                'completion_rate': 0.9 + random.uniform(-0.1, 0.1)
            })
        
        return {
            'patient_id': patient_id,
            'surveys': surveys,
            'summary_stats': {
                'total_surveys': len(surveys),
                'avg_completion_rate': round(np.mean([s['completion_rate'] for s in surveys]), 2),
                'avg_adherence_score': round(np.mean([
                    r['score'] for s in surveys for r in s['responses'] 
                    if 'adherence' in r['question_id']
                ]), 2)
            }
        }
    
    def _get_adherence_response(self, adherence_score: float) -> str:
        """Get adherence response based on score."""
        if adherence_score > 0.8:
            return 'Always'
        elif adherence_score > 0.6:
            return 'Usually'
        elif adherence_score > 0.4:
            return 'Sometimes'
        else:
            return 'Rarely'
    
    def _get_adherence_score(self, adherence_score: float) -> int:
        """Get adherence score (1-5 scale)."""
        if adherence_score > 0.8:
            return 5
        elif adherence_score > 0.6:
            return 4
        elif adherence_score > 0.4:
            return 3
        elif adherence_score > 0.2:
            return 2
        else:
            return 1
    
    def _get_symptom_response(self, adherence_score: float) -> str:
        """Get symptom response based on adherence."""
        if adherence_score > 0.7:
            return 'Very mild'
        elif adherence_score > 0.5:
            return 'Mild'
        elif adherence_score > 0.3:
            return 'Moderate'
        else:
            return 'Severe'
    
    def _get_symptom_score(self, adherence_score: float) -> int:
        """Get symptom score (1-5 scale, higher is better)."""
        if adherence_score > 0.7:
            return 5
        elif adherence_score > 0.5:
            return 4
        elif adherence_score > 0.3:
            return 3
        elif adherence_score > 0.1:
            return 2
        else:
            return 1
    
    def _get_qol_response(self, adherence_score: float) -> str:
        """Get quality of life response."""
        if adherence_score > 0.7:
            return 'Excellent'
        elif adherence_score > 0.5:
            return 'Good'
        elif adherence_score > 0.3:
            return 'Fair'
        else:
            return 'Poor'
    
    def _get_qol_score(self, adherence_score: float) -> int:
        """Get quality of life score (1-5 scale)."""
        if adherence_score > 0.7:
            return 5
        elif adherence_score > 0.5:
            return 4
        elif adherence_score > 0.3:
            return 3
        elif adherence_score > 0.1:
            return 2
        else:
            return 1
    
    def generate_complete_dataset(self) -> Dict[str, Any]:
        """Generate complete dataset with all modalities."""
        print(f"Generating dataset for {self.num_patients} patients...")
        
        # Generate patient demographics
        patients_df = self.generate_patient_demographics()
        
        # Generate data for each patient
        wearable_data = []
        ehr_data = []
        survey_data = []
        
        for _, patient in patients_df.iterrows():
            print(f"Generating data for patient {patient['patient_id']}...")
            
            # Generate wearable data
            wearable = self.generate_wearable_data(patient['patient_id'], patient['true_adherence_score'])
            wearable_data.append(wearable)
            
            # Generate EHR data
            ehr = self.generate_ehr_data(patient['patient_id'], patient['true_adherence_score'])
            ehr_data.append(ehr)
            
            # Generate survey data
            survey = self.generate_survey_data(patient['patient_id'], patient['true_adherence_score'])
            survey_data.append(survey)
        
        # Create complete dataset
        dataset = {
            'metadata': {
                'num_patients': self.num_patients,
                'trial_duration_days': self.trial_duration_days,
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'end_date': self.end_date.strftime('%Y-%m-%d'),
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'patients': patients_df.to_dict('records'),
            'wearable_data': wearable_data,
            'ehr_data': ehr_data,
            'survey_data': survey_data
        }
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, Any], output_dir: str = 'test_data') -> None:
        """Save dataset to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save complete dataset as JSON
        with open(os.path.join(output_dir, 'complete_dataset.json'), 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Save individual CSV files for easy analysis
        patients_df = pd.DataFrame(dataset['patients'])
        patients_df.to_csv(os.path.join(output_dir, 'patients.csv'), index=False)
        
        # Save wearable data summary
        wearable_summary = []
        for data in dataset['wearable_data']:
            summary = {
                'patient_id': data['patient_id'],
                'device_type': data['device_type'],
                **data['summary_stats']
            }
            wearable_summary.append(summary)
        
        wearable_df = pd.DataFrame(wearable_summary)
        wearable_df.to_csv(os.path.join(output_dir, 'wearable_summary.csv'), index=False)
        
        # Save EHR data summary
        ehr_summary = []
        for data in dataset['ehr_data']:
            summary = {
                'patient_id': data['patient_id'],
                **data['summary_stats']
            }
            ehr_summary.append(summary)
        
        ehr_df = pd.DataFrame(ehr_summary)
        ehr_df.to_csv(os.path.join(output_dir, 'ehr_summary.csv'), index=False)
        
        # Save survey data summary
        survey_summary = []
        for data in dataset['survey_data']:
            summary = {
                'patient_id': data['patient_id'],
                **data['summary_stats']
            }
            survey_summary.append(summary)
        
        survey_df = pd.DataFrame(survey_summary)
        survey_df.to_csv(os.path.join(output_dir, 'survey_summary.csv'), index=False)
        
        print(f"Dataset saved to {output_dir}/")
        print(f"Files created:")
        print(f"  - complete_dataset.json (full dataset)")
        print(f"  - patients.csv (patient demographics)")
        print(f"  - wearable_summary.csv (wearable data summary)")
        print(f"  - ehr_summary.csv (EHR data summary)")
        print(f"  - survey_summary.csv (survey data summary)")

def main():
    """Generate and save test dataset."""
    # Create dataset generator
    generator = ClinicalTrialDatasetGenerator(num_patients=50, trial_duration_days=90)
    
    # Generate complete dataset
    dataset = generator.generate_complete_dataset()
    
    # Save dataset
    generator.save_dataset(dataset, 'test_data')
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total patients: {len(dataset['patients'])}")
    print(f"Trial duration: {dataset['metadata']['trial_duration_days']} days")
    
    # Adherence distribution
    adherence_counts = {}
    for patient in dataset['patients']:
        category = patient['adherence_category']
        adherence_counts[category] = adherence_counts.get(category, 0) + 1
    
    print("\nAdherence Distribution:")
    for category, count in adherence_counts.items():
        percentage = (count / len(dataset['patients'])) * 100
        print(f"  {category}: {count} patients ({percentage:.1f}%)")
    
    print("\nDataset generation complete!")

if __name__ == "__main__":
    main() 