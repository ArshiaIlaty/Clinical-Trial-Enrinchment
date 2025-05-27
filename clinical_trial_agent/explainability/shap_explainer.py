from typing import Dict, Any, List
import logging
import numpy as np
import shap
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """SHAP-based explainer for adherence predictions."""
    
    def __init__(self, model_path: str = None):
        self.model = None
        if model_path:
            self.load_model(model_path)
        
        self.feature_names = [
            # Wearable features
            "heart_rate_resting",
            "heart_rate_variability",
            "sleep_duration",
            "sleep_quality",
            "activity_steps",
            "activity_calories",
            
            # EHR features
            "medication_count",
            "lab_result_count",
            "vital_sign_count",
            "clinical_note_count",
            
            # Survey features
            "adherence_score",
            "symptom_score",
            "quality_of_life_score",
            "side_effect_score"
        ]
    
    def load_model(self, model_path: str) -> None:
        """Load the model for explanation."""
        try:
            self.model = load_model(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    async def explain(self, prediction: Dict[str, Any]) -> Dict[str, float]:
        """Generate SHAP explanations for a prediction."""
        try:
            if not self.model:
                raise ValueError("Model not loaded. Call load_model first.")
            
            # Extract features from prediction
            features = np.array(list(prediction["features"].values())).reshape(1, -1)
            
            # Create explainer
            explainer = shap.DeepExplainer(self.model, features)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(features)
            
            # Convert to feature importance dictionary
            feature_importance = dict(zip(
                self.feature_names,
                np.abs(shap_values[0])
            ))
            
            # Normalize importance scores
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                feature_importance = {
                    k: v / total_importance
                    for k, v in feature_importance.items()
                }
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {str(e)}")
            raise
    
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