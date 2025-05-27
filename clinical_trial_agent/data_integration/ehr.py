from typing import Dict, Any, List, ClassVar
from datetime import datetime
import logging
import aiohttp
import asyncio
from pydantic import Field

from fhirclient import client
from langchain.tools import BaseTool
from .base_connector import BaseDataConnector

logger = logging.getLogger(__name__)

class EHRDataTool(BaseTool):
    """Tool for getting EHR data."""
    
    name: ClassVar[str] = "ehr_data"
    description: ClassVar[str] = "Get Electronic Health Record data for a patient"
    connector: 'EHRConnector' = Field(description="The EHR data connector")
    
    def __init__(self, connector: 'EHRConnector'):
        super().__init__(connector=connector)
    
    async def _arun(self, patient_id: str, start_date: datetime, end_date: datetime, **kwargs) -> Dict[str, Any]:
        """Run the tool asynchronously."""
        return await self.connector.get_data(patient_id, start_date, end_date, **kwargs)
    
    def _run(self, patient_id: str, start_date: datetime, end_date: datetime, **kwargs) -> Dict[str, Any]:
        """Run the tool synchronously."""
        return asyncio.run(self._arun(patient_id, start_date, end_date, **kwargs))

class EHRConnector(BaseDataConnector):
    """Connector for Electronic Health Records (EHR) data."""
    
    def __init__(self):
        super().__init__(
            name="EHRConnector",
            description="Connector for Electronic Health Records (EHR) data"
        )
        self._fhir_client = None
        self._session = None
    
    async def get_data(
        self,
        patient_id: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Dict[str, Any]:
        """Retrieve EHR data for the specified patient and time range."""
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            # Initialize FHIR client if needed
            if not self._fhir_client:
                self._fhir_client = self._initialize_fhir_client()
            
            # Get various types of EHR data
            data = {
                "medications": await self._get_medications(patient_id, start_date, end_date),
                "diagnoses": await self._get_diagnoses(patient_id, start_date, end_date),
                "lab_results": await self._get_lab_results(patient_id, start_date, end_date),
                "vitals": await self._get_vitals(patient_id, start_date, end_date),
                "clinical_notes": await self._get_clinical_notes(patient_id, start_date, end_date)
            }
            
            # Validate and preprocess data
            if await self.validate_data(data):
                data = await self.preprocess_data(data)
                return data
            else:
                raise ValueError("Invalid EHR data")
            
        except Exception as e:
            logger.error(f"Error getting EHR data: {str(e)}")
            raise
        finally:
            if self._session:
                await self._session.close()
                self._session = None
    
    def _initialize_fhir_client(self) -> client.FHIRClient:
        """Initialize FHIR client."""
        # This is a placeholder implementation
        settings = {
            'app_id': 'clinical_trial_agent',
            'api_base': 'https://fhir-api-url.com/fhir',
            'auth': {
                'token': 'your-auth-token'
            }
        }
        return client.FHIRClient(settings=settings)
    
    async def _get_medications(
        self,
        patient_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get medication history."""
        # Implement FHIR MedicationRequest search
        # This is a placeholder implementation
        return [
            {
                "name": "Lisinopril",
                "dosage": "10mg",
                "frequency": "daily",
                "start_date": "2024-01-01",
                "status": "active"
            },
            {
                "name": "Metformin",
                "dosage": "500mg",
                "frequency": "twice daily",
                "start_date": "2024-01-15",
                "status": "active"
            }
        ]
    
    async def _get_diagnoses(
        self,
        patient_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get diagnosis history."""
        # Implement FHIR Condition search
        # This is a placeholder implementation
        return [
            {
                "code": "E11.9",
                "description": "Type 2 diabetes mellitus without complications",
                "date": "2024-01-01",
                "status": "active"
            },
            {
                "code": "I10",
                "description": "Essential (primary) hypertension",
                "date": "2024-01-15",
                "status": "active"
            }
        ]
    
    async def _get_lab_results(
        self,
        patient_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get laboratory results."""
        # Implement FHIR Observation search
        # This is a placeholder implementation
        return [
            {
                "test": "Hemoglobin A1c",
                "value": 7.2,
                "unit": "%",
                "date": "2024-01-15",
                "reference_range": "4.0-5.6"
            },
            {
                "test": "Blood Pressure",
                "value": "130/85",
                "unit": "mmHg",
                "date": "2024-01-15",
                "reference_range": "<120/80"
            }
        ]
    
    async def _get_vitals(
        self,
        patient_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get vital signs."""
        # Implement FHIR Observation search for vitals
        # This is a placeholder implementation
        return [
            {
                "type": "Blood Pressure",
                "value": "130/85",
                "unit": "mmHg",
                "date": "2024-01-15"
            },
            {
                "type": "Heart Rate",
                "value": 72,
                "unit": "bpm",
                "date": "2024-01-15"
            }
        ]
    
    async def _get_clinical_notes(
        self,
        patient_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get clinical notes."""
        # Implement FHIR DocumentReference search
        # This is a placeholder implementation
        return [
            {
                "type": "Progress Note",
                "date": "2024-01-15",
                "content": "Patient reports good adherence to medication regimen.",
                "author": "Dr. Smith"
            }
        ]
    
    def get_tool(self) -> BaseTool:
        """Get the LangChain tool for this connector."""
        return EHRDataTool(self)
    
    def _get_required_fields(self) -> List[str]:
        """Get list of required fields for EHR data."""
        return [
            "medications",
            "diagnoses",
            "lab_results",
            "vitals",
            "clinical_notes"
        ]
    
    async def _validate_data_specific(self, data: Dict[str, Any]) -> bool:
        """Validate EHR-specific data."""
        try:
            # Check medications
            if not isinstance(data["medications"], list):
                return False
            
            # Check diagnoses
            if not isinstance(data["diagnoses"], list):
                return False
            
            # Check lab results
            if not isinstance(data["lab_results"], list):
                return False
            
            # Check vitals
            if not isinstance(data["vitals"], list):
                return False
            
            # Check clinical notes
            if not isinstance(data["clinical_notes"], list):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating EHR data: {str(e)}")
            return False
    
    def _get_sensitive_fields(self) -> List[str]:
        """Get list of sensitive fields for EHR data."""
        return [
            "patient_id",
            "name",
            "date_of_birth",
            "address",
            "phone_number",
            "email",
            "insurance_info"
        ]
    
    async def _preprocess_data_specific(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess EHR-specific data."""
        try:
            processed_data = data.copy()
            
            # Anonymize clinical notes
            for note in processed_data["clinical_notes"]:
                if "content" in note:
                    # Implement basic text anonymization
                    note["content"] = self._anonymize_clinical_text(note["content"])
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing EHR data: {str(e)}")
            return data
    
    def _anonymize_clinical_text(self, text: str) -> str:
        """Anonymize sensitive information in clinical text."""
        # Implement text anonymization
        # This is a placeholder implementation
        return text 