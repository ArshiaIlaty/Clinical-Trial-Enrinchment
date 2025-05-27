from typing import Dict, Any, List, ClassVar
from datetime import datetime
import logging
import aiohttp
import asyncio
from pydantic import Field

from langchain.tools import BaseTool
from .base_connector import BaseDataConnector

logger = logging.getLogger(__name__)

class SurveyDataTool(BaseTool):
    """Tool for getting survey data."""
    
    name: ClassVar[str] = "survey_data"
    description: ClassVar[str] = "Get patient-reported survey data"
    connector: 'SurveyConnector' = Field(description="The survey data connector")
    
    def __init__(self, connector: 'SurveyConnector'):
        super().__init__(connector=connector)
    
    async def _arun(self, patient_id: str, start_date: datetime, end_date: datetime, **kwargs) -> Dict[str, Any]:
        """Run the tool asynchronously."""
        return await self.connector.get_data(patient_id, start_date, end_date, **kwargs)
    
    def _run(self, patient_id: str, start_date: datetime, end_date: datetime, **kwargs) -> Dict[str, Any]:
        """Run the tool synchronously."""
        return asyncio.run(self._arun(patient_id, start_date, end_date, **kwargs))

class SurveyConnector(BaseDataConnector):
    """Connector for patient-reported survey data."""
    
    def __init__(self):
        super().__init__(
            name="SurveyConnector",
            description="Connector for patient-reported survey data"
        )
        self._session = None
        self.survey_types = {
            "adherence": "medication_adherence_survey",
            "symptoms": "symptom_tracking",
            "quality_of_life": "quality_of_life_assessment",
            "side_effects": "side_effects_reporting"
        }
    
    async def get_data(
        self,
        patient_id: str,
        start_date: datetime,
        end_date: datetime,
        survey_type: str = "adherence",
        **kwargs
    ) -> Dict[str, Any]:
        """Retrieve survey data for the specified patient and time range."""
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            # Get survey data based on type
            if survey_type not in self.survey_types:
                raise ValueError(f"Unsupported survey type: {survey_type}")
            
            data = await self._get_survey_data(
                patient_id,
                start_date,
                end_date,
                self.survey_types[survey_type]
            )
            
            # Validate and preprocess data
            if await self.validate_data(data):
                data = await self.preprocess_data(data)
                return data
            else:
                raise ValueError("Invalid survey data")
            
        except Exception as e:
            logger.error(f"Error getting survey data: {str(e)}")
            raise
        finally:
            if self._session:
                await self._session.close()
                self._session = None
    
    async def _get_survey_data(
        self,
        patient_id: str,
        start_date: datetime,
        end_date: datetime,
        survey_id: str
    ) -> Dict[str, Any]:
        """Get survey data from the survey system."""
        # Implement survey system API calls
        # This is a placeholder implementation
        return {
            "survey_id": survey_id,
            "responses": [
                {
                    "question_id": "q1",
                    "question": "How often did you take your medication as prescribed?",
                    "response": "Always",
                    "score": 5,
                    "date": "2024-01-15"
                },
                {
                    "question_id": "q2",
                    "question": "Did you experience any side effects?",
                    "response": "Mild nausea",
                    "score": 2,
                    "date": "2024-01-15"
                },
                {
                    "question_id": "q3",
                    "question": "How would you rate your quality of life?",
                    "response": "Good",
                    "score": 4,
                    "date": "2024-01-15"
                }
            ],
            "metadata": {
                "completion_rate": 0.95,
                "last_updated": "2024-01-15T10:30:00Z"
            }
        }
    
    def get_tool(self) -> BaseTool:
        """Get the LangChain tool for this connector."""
        return SurveyDataTool(self)
    
    def _get_required_fields(self) -> List[str]:
        """Get list of required fields for survey data."""
        return [
            "survey_id",
            "responses",
            "metadata"
        ]
    
    async def _validate_data_specific(self, data: Dict[str, Any]) -> bool:
        """Validate survey-specific data."""
        try:
            # Check survey ID
            if not isinstance(data["survey_id"], str):
                return False
            
            # Check responses
            if not isinstance(data["responses"], list):
                return False
            
            # Validate each response
            for response in data["responses"]:
                if not all(key in response for key in ["question_id", "response", "score", "date"]):
                    return False
                if not isinstance(response["score"], (int, float)):
                    return False
            
            # Check metadata
            if not isinstance(data["metadata"], dict):
                return False
            if "completion_rate" not in data["metadata"]:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating survey data: {str(e)}")
            return False
    
    def _get_sensitive_fields(self) -> List[str]:
        """Get list of sensitive fields for survey data."""
        return [
            "patient_id",
            "name",
            "contact_info",
            "free_text_responses"
        ]
    
    async def _preprocess_data_specific(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess survey-specific data."""
        try:
            processed_data = data.copy()
            
            # Normalize scores
            for response in processed_data["responses"]:
                if "score" in response:
                    response["score"] = round(float(response["score"]), 2)
            
            # Calculate aggregate scores
            processed_data["aggregate_scores"] = self._calculate_aggregate_scores(
                processed_data["responses"]
            )
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing survey data: {str(e)}")
            return data
    
    def _calculate_aggregate_scores(self, responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate scores from survey responses."""
        try:
            scores = {}
            for response in responses:
                question_type = response["question_id"].split("_")[0]
                if question_type not in scores:
                    scores[question_type] = []
                scores[question_type].append(response["score"])
            
            # Calculate averages
            return {
                question_type: sum(scores_list) / len(scores_list)
                for question_type, scores_list in scores.items()
            }
            
        except Exception as e:
            logger.error(f"Error calculating aggregate scores: {str(e)}")
            return {} 