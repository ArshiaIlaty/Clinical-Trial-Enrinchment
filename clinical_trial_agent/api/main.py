from typing import Dict, Any, List
from datetime import datetime
import logging
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ..agents.adherence_agent import AdherencePredictionAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Trial Adherence Prediction API",
    description="API for predicting and explaining clinical trial adherence",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize agent
agent = AdherencePredictionAgent()

# Pydantic models
class PredictionRequest(BaseModel):
    patient_id: str = Field(..., description="Patient identifier")
    start_date: datetime = Field(..., description="Start date for data analysis")
    end_date: datetime = Field(..., description="End date for data analysis")
    device_type: str = Field("fitbit", description="Type of wearable device")
    survey_type: str = Field("adherence", description="Type of survey data")

class PredictionResponse(BaseModel):
    risk_score: float = Field(..., description="Predicted risk of non-adherence")
    confidence: float = Field(..., description="Confidence in the prediction")
    explanation: str = Field(..., description="Explanation of the prediction")
    timestamp: datetime = Field(..., description="Timestamp of the prediction")
    features: Dict[str, float] = Field(..., description="Feature importance scores")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")

# Authentication
async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Implement proper authentication
    # This is a placeholder implementation
    return {"username": "test_user"}

# API endpoints
@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}}
)
async def predict_adherence(
    request: PredictionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PredictionResponse:
    """Predict clinical trial adherence for a patient."""
    try:
        # Prepare input data
        input_data = {
            "patient_id": request.patient_id,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "device_type": request.device_type,
            "survey_type": request.survey_type
        }
        
        # Get prediction from agent
        result = await agent.run(input_data)
        
        return PredictionResponse(
            risk_score=result["prediction"]["risk_score"],
            confidence=result["confidence"],
            explanation=result["explanation"],
            timestamp=datetime.fromisoformat(result["timestamp"]),
            features=result["prediction"]["features"]
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 