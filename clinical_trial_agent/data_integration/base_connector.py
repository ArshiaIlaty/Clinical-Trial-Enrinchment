from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

class BaseDataConnector(ABC):
    """Base class for data connectors."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.data_quality_metrics = {
            "completeness": 0.0,
            "timeliness": 0.0,
            "consistency": 0.0,
            "accuracy": 0.0
        }
    
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
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate the data structure and content."""
        try:
            # Check required fields
            required_fields = self._get_required_fields()
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate data types
            if not self._validate_data_types(data):
                return False
            
            # Check for data quality
            self._update_quality_metrics(data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False
    
    def _get_required_fields(self) -> List[str]:
        """Get list of required fields for the data type."""
        return ["timestamp", "patient_id"]
    
    def _validate_data_types(self, data: Dict[str, Any]) -> bool:
        """Validate data types of fields."""
        try:
            # Validate timestamp
            if not isinstance(data["timestamp"], (str, datetime)):
                logger.error("Invalid timestamp format")
                return False
            
            # Validate patient_id
            if not isinstance(data["patient_id"], str):
                logger.error("Invalid patient_id format")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data types: {str(e)}")
            return False
    
    def _update_quality_metrics(self, data: Dict[str, Any]) -> None:
        """Update data quality metrics."""
        try:
            # Calculate completeness
            total_fields = len(self._get_required_fields())
            non_null_fields = sum(1 for field in self._get_required_fields() 
                                if data.get(field) is not None)
            self.data_quality_metrics["completeness"] = non_null_fields / total_fields
            
            # Calculate timeliness
            if "timestamp" in data:
                current_time = datetime.now()
                data_time = pd.to_datetime(data["timestamp"])
                time_diff = (current_time - data_time).total_seconds()
                self.data_quality_metrics["timeliness"] = max(0, 1 - (time_diff / (24 * 3600)))  # 24-hour window
            
            # Calculate consistency
            self.data_quality_metrics["consistency"] = self._calculate_consistency(data)
            
            # Calculate accuracy
            self.data_quality_metrics["accuracy"] = self._calculate_accuracy(data)
            
        except Exception as e:
            logger.error(f"Error updating quality metrics: {str(e)}")
    
    def _calculate_consistency(self, data: Dict[str, Any]) -> float:
        """Calculate data consistency score."""
        # Implement consistency checks based on data type
        return 1.0  # Placeholder
    
    def _calculate_accuracy(self, data: Dict[str, Any]) -> float:
        """Calculate data accuracy score."""
        # Implement accuracy checks based on data type
        return 1.0  # Placeholder
    
    def get_data_quality(self) -> float:
        """Get overall data quality score."""
        weights = {
            "completeness": 0.3,
            "timeliness": 0.3,
            "consistency": 0.2,
            "accuracy": 0.2
        }
        
        return sum(score * weights[metric] 
                  for metric, score in self.data_quality_metrics.items())
    
    async def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the data before returning it."""
        try:
            # Convert timestamps to datetime
            if "timestamp" in data:
                data["timestamp"] = pd.to_datetime(data["timestamp"])
            
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Normalize data
            data = self._normalize_data(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return data
    
    def _handle_missing_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle missing values in the data."""
        # Implement missing value handling based on data type
        return data
    
    def _normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize the data for analysis."""
        # Implement data normalization based on data type
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