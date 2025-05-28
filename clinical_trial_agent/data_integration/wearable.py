from typing import Dict, Any, List, ClassVar
from datetime import datetime
import logging
import aiohttp
import asyncio
from pydantic import Field

from langchain.tools import BaseTool
from .base_connector import BaseDataConnector

logger = logging.getLogger(__name__)

class WearableDataTool(BaseTool):
    """Tool for getting wearable device data."""
    
    name: ClassVar[str] = "wearable_data"
    description: ClassVar[str] = "Get wearable device data for a patient"
    connector: 'WearableDataConnector' = Field(description="The wearable data connector")
    
    def __init__(self, connector: 'WearableDataConnector'):
        super().__init__(connector=connector)
    
    async def _arun(self, patient_id: str, start_date: datetime, end_date: datetime, **kwargs) -> Dict[str, Any]:
        """Run the tool asynchronously."""
        return await self.connector.get_data(patient_id, start_date, end_date, **kwargs)
    
    def _run(self, patient_id: str, start_date: datetime, end_date: datetime, **kwargs) -> Dict[str, Any]:
        """Run the tool synchronously."""
        return asyncio.run(self._arun(patient_id, start_date, end_date, **kwargs))

class WearableDataConnector(BaseDataConnector):
    """Connector for wearable device data."""
    
    def __init__(self):
        super().__init__(
            name="WearableDataConnector",
            description="Connector for wearable device data (e.g., Fitbit, Apple Watch)"
        )
        self.api_endpoints = {
            "fitbit": "https://api.fitbit.com/1/user/-/",
            "apple_health": "https://api.apple.com/health/",
        }
        self._session = None
    
    async def get_data(
        self,
        patient_id: str,
        start_date: datetime,
        end_date: datetime,
        device_type: str = "fitbit",
        **kwargs
    ) -> Dict[str, Any]:
        """Retrieve wearable data for the specified patient and time range."""
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            # Get device-specific data
            if device_type == "fitbit":
                data = await self._get_fitbit_data(patient_id, start_date, end_date)
            elif device_type == "apple_health":
                data = await self._get_apple_health_data(patient_id, start_date, end_date)
            else:
                raise ValueError(f"Unsupported device type: {device_type}")
            
            # Validate data
            if self.validate_data(data):
                # Preprocess data
                processed_data = await self.preprocess_data(data)
                return processed_data
            else:
                logger.error("Invalid wearable data")
                return {}
            
        except Exception as e:
            logger.error(f"Error getting wearable data: {str(e)}")
            return {}
        finally:
            if self._session:
                await self._session.close()
                self._session = None
    
    async def _get_fitbit_data(
        self,
        patient_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get data from Fitbit API."""
        # Implement Fitbit API calls
        # This is a placeholder implementation
        return {
            "heart_rate": {
                "resting": 65,
                "variability": 45,
                "trend": "stable"
            },
            "sleep": {
                "duration": 7.5,
                "quality": 0.85,
                "stages": {
                    "deep": 1.5,
                    "light": 4.0,
                    "rem": 2.0
                }
            },
            "activity": {
                "steps": 8500,
                "calories": 2100,
                "active_minutes": 45
            }
        }
    
    async def _get_apple_health_data(
        self,
        patient_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get data from Apple Health API."""
        # Implement Apple Health API calls
        # This is a placeholder implementation
        return {
            "heart_rate": {
                "resting": 68,
                "variability": 42,
                "trend": "stable"
            },
            "sleep": {
                "duration": 7.2,
                "quality": 0.82,
                "stages": {
                    "deep": 1.3,
                    "light": 4.2,
                    "rem": 1.7
                }
            },
            "activity": {
                "steps": 9200,
                "calories": 2300,
                "active_minutes": 50
            }
        }
    
    def get_tool(self) -> BaseTool:
        """Get the LangChain tool for this connector."""
        return WearableDataTool(self)
    
    def _get_required_fields(self) -> List[str]:
        """Get list of required fields for wearable data."""
        return [
            "heart_rate",
            "sleep",
            "activity"
        ]
    
    async def _validate_data_specific(self, data: Dict[str, Any]) -> bool:
        """Validate wearable-specific data."""
        try:
            # Check heart rate data
            if not isinstance(data["heart_rate"]["resting"], (int, float)):
                return False
            if not isinstance(data["heart_rate"]["variability"], (int, float)):
                return False
            
            # Check sleep data
            if not isinstance(data["sleep"]["duration"], (int, float)):
                return False
            if not isinstance(data["sleep"]["quality"], (int, float)):
                return False
            
            # Check activity data
            if not isinstance(data["activity"]["steps"], int):
                return False
            if not isinstance(data["activity"]["calories"], (int, float)):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating wearable data: {str(e)}")
            return False
    
    def _get_sensitive_fields(self) -> List[str]:
        """Get list of sensitive fields for wearable data."""
        return [
            "patient_id",
            "device_id",
            "location_data"
        ]
    
    async def _preprocess_data_specific(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess wearable-specific data."""
        try:
            processed_data = data.copy()
            
            # Normalize heart rate data
            processed_data["heart_rate"]["resting"] = round(
                processed_data["heart_rate"]["resting"], 1
            )
            processed_data["heart_rate"]["variability"] = round(
                processed_data["heart_rate"]["variability"], 1
            )
            
            # Normalize sleep data
            processed_data["sleep"]["duration"] = round(
                processed_data["sleep"]["duration"], 1
            )
            processed_data["sleep"]["quality"] = round(
                processed_data["sleep"]["quality"], 2
            )
            
            # Normalize activity data
            processed_data["activity"]["calories"] = round(
                processed_data["activity"]["calories"], 0
            )
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing wearable data: {str(e)}")
            return data 