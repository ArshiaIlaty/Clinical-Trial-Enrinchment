from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import logging

from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

class BaseDataConnector(ABC):
    """Base class for data connectors."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._data_quality = 1.0  # Default data quality score
    
    @abstractmethod
    async def get_data(
        self,
        patient_id: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Dict[str, Any]:
        """Retrieve data for the specified patient and time range."""
        pass
    
    @abstractmethod
    def get_tool(self) -> BaseTool:
        """Get the LangChain tool for this connector."""
        pass
    
    def get_data_quality(self) -> float:
        """Get the current data quality score."""
        return self._data_quality
    
    def update_data_quality(self, score: float) -> None:
        """Update the data quality score."""
        self._data_quality = max(0.0, min(1.0, score))
    
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate the retrieved data."""
        try:
            # Basic validation
            if not data:
                return False
            
            # Check for required fields
            required_fields = self._get_required_fields()
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Perform specific validation
            return await self._validate_data_specific(data)
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False
    
    @abstractmethod
    def _get_required_fields(self) -> list:
        """Get list of required fields for this data source."""
        pass
    
    @abstractmethod
    async def _validate_data_specific(self, data: Dict[str, Any]) -> bool:
        """Perform source-specific data validation."""
        pass
    
    async def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the data before returning it."""
        try:
            # Basic preprocessing
            processed_data = data.copy()
            
            # Remove sensitive information
            processed_data = self._remove_sensitive_data(processed_data)
            
            # Perform specific preprocessing
            processed_data = await self._preprocess_data_specific(processed_data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return data
    
    def _remove_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from the data."""
        # Implement basic data anonymization
        sensitive_fields = self._get_sensitive_fields()
        for field in sensitive_fields:
            if field in data:
                data[field] = "[REDACTED]"
        return data
    
    @abstractmethod
    def _get_sensitive_fields(self) -> list:
        """Get list of sensitive fields for this data source."""
        pass
    
    @abstractmethod
    async def _preprocess_data_specific(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform source-specific data preprocessing."""
        pass 