from typing import Dict, Any, List, Optional
import logging
import numpy as np
import pandas as pd
import shap
from tensorflow.keras.models import load_model
from datetime import datetime

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """Generates explanations for adherence predictions using SHAP values."""
    
    def __init__(self):
        self.explainer = None
        self.feature_names = []
        self.feature_categories = {
            "wearable": [
                "activity_level",
                "sleep_quality",
                "heart_rate",
                "steps",
                "calories"
            ],
            "ehr": [
                "medication_adherence",
                "visit_attendance",
                "lab_results",
                "vitals",
                "clinical_notes"
            ],
            "survey": [
                "symptom_severity",
                "quality_of_life",
                "side_effects",
                "adherence_self_report",
                "satisfaction"
            ],
            "temporal": [
                "data_frequency",
                "consistency",
                "trends",
                "patterns"
            ],
            "behavioral": [
                "activity_patterns",
                "medication_patterns",
                "engagement_level",
                "response_time"
            ]
        }
    
    def load_model(self, model: Any, feature_names: List[str]) -> None:
        """Load the model and feature names."""
        try:
            self.feature_names = feature_names
            self.explainer = shap.DeepExplainer(model, np.zeros((1, len(feature_names))))
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def explain(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Generate SHAP explanations for the prediction."""
        try:
            # Convert data to feature array
            features = self._convert_to_features(data)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(features)
            
            # Get feature importance
            feature_importance = self._calculate_feature_importance(shap_values)
            
            # Categorize features
            categorized_importance = self._categorize_features(feature_importance)
            
            # Generate detailed explanations
            explanations = self._generate_explanations(categorized_importance)
            
            return {
                "feature_importance": feature_importance,
                "categorized_importance": categorized_importance,
                "explanations": explanations,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating explanations: {str(e)}")
            raise
    
    def _convert_to_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Convert input data to feature array."""
        try:
            features = []
            
            # Extract wearable features
            if "wearable" in data:
                features.extend(self._extract_wearable_features(data["wearable"]))
            
            # Extract EHR features
            if "ehr" in data:
                features.extend(self._extract_ehr_features(data["ehr"]))
            
            # Extract survey features
            if "survey" in data:
                features.extend(self._extract_survey_features(data["survey"]))
            
            # Extract temporal features
            features.extend(self._extract_temporal_features(data))
            
            # Extract behavioral features
            features.extend(self._extract_behavioral_features(data))
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error converting data to features: {str(e)}")
            raise
    
    def _extract_wearable_features(self, wearable_data: Dict[str, Any]) -> List[float]:
        """Extract features from wearable device data."""
        features = []
        
        # Activity level
        if "activity" in wearable_data:
            activity = wearable_data["activity"]
            features.extend([
                np.mean(activity),
                np.std(activity),
                np.max(activity)
            ])
        
        # Sleep quality
        if "sleep" in wearable_data:
            sleep = wearable_data["sleep"]
            features.extend([
                sleep.get("duration", 0),
                sleep.get("quality", 0),
                sleep.get("efficiency", 0)
            ])
        
        return features
    
    def _extract_ehr_features(self, ehr_data: Dict[str, Any]) -> List[float]:
        """Extract features from EHR data."""
        features = []
        
        # Medication adherence
        if "medications" in ehr_data:
            meds = ehr_data["medications"]
            features.extend([
                len(meds),
                sum(1 for m in meds if m.get("adherence", 0) > 0.8)
            ])
        
        # Visit attendance
        if "visits" in ehr_data:
            visits = ehr_data["visits"]
            features.extend([
                len(visits),
                sum(1 for v in visits if v.get("attended", False))
            ])
        
        return features
    
    def _extract_survey_features(self, survey_data: Dict[str, Any]) -> List[float]:
        """Extract features from survey data."""
        features = []
        
        # Symptom severity
        if "symptoms" in survey_data:
            symptoms = survey_data["symptoms"]
            features.extend([
                np.mean(symptoms),
                np.std(symptoms),
                np.max(symptoms)
            ])
        
        # Quality of life
        if "quality_of_life" in survey_data:
            qol = survey_data["quality_of_life"]
            features.extend([
                qol.get("physical", 0),
                qol.get("mental", 0),
                qol.get("social", 0)
            ])
        
        return features
    
    def _extract_temporal_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract temporal features from the data."""
        features = []
        
        # Data frequency
        timestamps = []
        for source in ["wearable", "ehr", "survey"]:
            if source in data and "timestamp" in data[source]:
                timestamps.append(pd.to_datetime(data[source]["timestamp"]))
        
        if timestamps:
            time_diffs = np.diff(sorted(timestamps))
            features.extend([
                np.mean(time_diffs),
                np.std(time_diffs)
            ])
        
        return features
    
    def _extract_behavioral_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract behavioral features from the data."""
        features = []
        
        # Activity patterns
        if "wearable" in data and "activity" in data["wearable"]:
            activity = data["wearable"]["activity"]
            features.append(np.std(activity))
        
        # Medication patterns
        if "ehr" in data and "medications" in data["ehr"]:
            meds = data["ehr"]["medications"]
            adherence_rates = [m.get("adherence", 0) for m in meds]
            features.append(np.mean(adherence_rates))
        
        return features
    
    def _calculate_feature_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance from SHAP values."""
        try:
            # Get absolute SHAP values
            abs_shap = np.abs(shap_values[0])
            
            # Calculate importance scores
            importance = abs_shap / np.sum(abs_shap)
            
            # Create feature importance dictionary
            feature_importance = dict(zip(self.feature_names, importance))
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise
    
    def _categorize_features(self, feature_importance: Dict[str, float]) -> Dict[str, float]:
        """Categorize features and calculate category importance."""
        try:
            categorized_importance = {
                "wearable": 0.0,
                "ehr": 0.0,
                "survey": 0.0,
                "temporal": 0.0,
                "behavioral": 0.0
            }
            
            # Calculate category importance
            for category, features in self.feature_categories.items():
                category_importance = 0.0
                for feature in features:
                    if feature in feature_importance:
                        category_importance += feature_importance[feature]
                categorized_importance[category] = category_importance
            
            # Normalize category importance
            total = sum(categorized_importance.values())
            if total > 0:
                categorized_importance = {
                    k: v/total for k, v in categorized_importance.items()
                }
            
            return categorized_importance
            
        except Exception as e:
            logger.error(f"Error categorizing features: {str(e)}")
            raise
    
    def _generate_explanations(self, categorized_importance: Dict[str, float]) -> Dict[str, str]:
        """Generate detailed explanations for each category."""
        try:
            explanations = {}
            
            # Generate explanations for each category
            for category, importance in categorized_importance.items():
                if importance > 0:
                    explanations[category] = self._get_category_explanation(
                        category, importance
                    )
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating explanations: {str(e)}")
            raise
    
    def _get_category_explanation(self, category: str, importance: float) -> str:
        """Get explanation for a specific category."""
        explanations = {
            "wearable": "Wearable device data indicates {:.1%} influence on adherence prediction. This includes activity levels, sleep quality, and heart rate patterns.",
            "ehr": "Electronic Health Records contribute {:.1%} to the prediction. This includes medication adherence, visit attendance, and clinical notes.",
            "survey": "Patient-reported data from surveys accounts for {:.1%} of the prediction. This includes symptom severity, quality of life, and side effects.",
            "temporal": "Temporal patterns in the data show {:.1%} influence. This includes data frequency, consistency, and trends over time.",
            "behavioral": "Behavioral patterns contribute {:.1%} to the prediction. This includes activity patterns, medication patterns, and engagement level."
        }
        
        return explanations.get(category, "").format(importance)
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get human-readable descriptions of features."""
        return {
            # Wearable features
            "heart_rate_resting": "Average resting heart rate",
            "heart_rate_variability": "Heart rate variability (HRV) measure",
            "sleep_duration": "Average sleep duration in hours",
            "sleep_quality": "Sleep quality score (0-1)",
            "activity_steps": "Daily step count",
            "activity_calories": "Daily calorie burn",
            
            # EHR features
            "medication_count": "Number of active medications",
            "lab_result_count": "Number of recent lab results",
            "vital_sign_count": "Number of vital sign measurements",
            "clinical_note_count": "Number of clinical notes",
            
            # Survey features
            "adherence_score": "Self-reported medication adherence score",
            "symptom_score": "Reported symptom severity score",
            "quality_of_life_score": "Quality of life assessment score",
            "side_effect_score": "Reported side effect severity score"
        }
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """Get feature categories for grouping explanations."""
        return {
            "Wearable Data": [
                "heart_rate_resting",
                "heart_rate_variability",
                "sleep_duration",
                "sleep_quality",
                "activity_steps",
                "activity_calories"
            ],
            "EHR Data": [
                "medication_count",
                "lab_result_count",
                "vital_sign_count",
                "clinical_note_count"
            ],
            "Patient Reports": [
                "adherence_score",
                "symptom_score",
                "quality_of_life_score",
                "side_effect_score"
            ]
        }
    
    def format_explanation(
        self,
        feature_importance: Dict[str, float],
        threshold: float = 0.05
    ) -> str:
        """Format feature importance into a human-readable explanation."""
        try:
            # Sort features by importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Get feature descriptions
            descriptions = self.get_feature_descriptions()
            
            # Format explanation
            explanation = "Key factors influencing the prediction:\n\n"
            
            for feature, importance in sorted_features:
                if importance >= threshold:
                    explanation += f"- {descriptions[feature]}: {importance:.1%}\n"
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error formatting explanation: {str(e)}")
            raise
    
    def get_category_importance(
        self,
        feature_importance: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate importance scores for each feature category."""
        try:
            categories = self.get_feature_categories()
            category_importance = {}
            
            for category, features in categories.items():
                importance = sum(
                    feature_importance[feature]
                    for feature in features
                    if feature in feature_importance
                )
                category_importance[category] = importance
            
            return category_importance
            
        except Exception as e:
            logger.error(f"Error calculating category importance: {str(e)}")
            raise 