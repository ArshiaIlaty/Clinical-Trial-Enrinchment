from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta

from .base_agent import BaseClinicalAgent
from ..data_integration.wearable import WearableDataConnector
from ..data_integration.ehr import EHRConnector
from ..data_integration.survey import SurveyConnector
from ..models.adherence_predictor import AdherencePredictor
from ..explainability.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)

class AdherencePredictionAgent(BaseClinicalAgent):
    """Agent for predicting clinical trial adherence."""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
    ):
        # Initialize data connectors
        self.wearable_connector = WearableDataConnector()
        self.ehr_connector = EHRConnector()
        self.survey_connector = SurveyConnector()
        
        # Initialize prediction model
        self.predictor = AdherencePredictor()
        
        # Initialize explainability component
        self.explainer = SHAPExplainer()
        
        # Define tools for the agent
        tools = [
            self.wearable_connector.get_tool(),
            self.ehr_connector.get_tool(),
            self.survey_connector.get_tool(),
        ]
        
        super().__init__(
            name="AdherencePredictionAgent",
            description="Predicts clinical trial adherence and provides explanations",
            tools=tools,
            model_name=model_name,
            temperature=temperature
        )
    
    def _get_system_prompt(self) -> str:
        return """You are an AI agent specialized in predicting clinical trial adherence.
        Your role is to analyze patient data from multiple sources and predict the likelihood
        of trial adherence. You should:
        1. Analyze patterns in wearable device data
        2. Review EHR records for relevant medical history
        3. Consider patient-reported outcomes
        4. Generate predictions with clear explanations
        5. Highlight risk factors and potential interventions
        
        Always maintain patient privacy and follow HIPAA guidelines."""
    
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and integrate data from multiple sources."""
        try:
            # Get time window for analysis
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Analyze last 30 days
            
            # Collect data from all sources
            wearable_data = await self.wearable_connector.get_data(
                patient_id=data["patient_id"],
                start_date=start_date,
                end_date=end_date
            )
            
            ehr_data = await self.ehr_connector.get_data(
                patient_id=data["patient_id"],
                start_date=start_date,
                end_date=end_date
            )
            
            survey_data = await self.survey_connector.get_data(
                patient_id=data["patient_id"],
                start_date=start_date,
                end_date=end_date
            )
            
            # Integrate all data sources
            integrated_data = {
                "wearable": wearable_data,
                "ehr": ehr_data,
                "survey": survey_data,
                "metadata": {
                    "patient_id": data["patient_id"],
                    "analysis_period": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    }
                }
            }
            
            return integrated_data
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
    
    async def generate_explanation(self, prediction: Any) -> str:
        """Generate explanation for the prediction using SHAP values."""
        try:
            # Get feature importance using SHAP
            feature_importance = await self.explainer.explain(prediction)
            
            # Format explanation
            explanation = "Based on the analysis of patient data:\n\n"
            
            # Add key factors
            explanation += "Key factors influencing adherence prediction:\n"
            for feature, importance in feature_importance.items():
                explanation += f"- {feature}: {importance:.2f}\n"
            
            # Add recommendations
            explanation += "\nRecommendations:\n"
            if prediction["risk_score"] > 0.7:
                explanation += "- High risk of non-adherence detected\n"
                explanation += "- Consider immediate intervention\n"
                explanation += "- Schedule additional follow-up\n"
            elif prediction["risk_score"] > 0.4:
                explanation += "- Moderate risk of non-adherence\n"
                explanation += "- Monitor closely\n"
                explanation += "- Consider preventive measures\n"
            else:
                explanation += "- Low risk of non-adherence\n"
                explanation += "- Continue current monitoring\n"
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise
    
    def _calculate_confidence(self, prediction: Any) -> float:
        """Calculate confidence score based on data quality and model certainty."""
        try:
            # Get data quality scores
            wearable_quality = self.wearable_connector.get_data_quality()
            ehr_quality = self.ehr_connector.get_data_quality()
            survey_quality = self.survey_connector.get_data_quality()
            
            # Calculate weighted average of data quality
            data_quality = (
                wearable_quality * 0.4 +
                ehr_quality * 0.4 +
                survey_quality * 0.2
            )
            
            # Combine with model confidence
            model_confidence = prediction.get("confidence", 0.5)
            
            # Final confidence score
            confidence = (data_quality * 0.3 + model_confidence * 0.7)
            
            return min(max(confidence, 0.0), 1.0)  # Ensure between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0 