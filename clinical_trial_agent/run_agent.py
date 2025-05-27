import asyncio
import logging
from datetime import datetime, timedelta
import argparse

from clinical_trial_agent.agents.adherence_agent import AdherencePredictionAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run the adherence prediction agent."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Run Clinical Trial Adherence Agent')
        parser.add_argument('--patient-id', required=True, help='Patient identifier')
        parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
        parser.add_argument('--device-type', default='fitbit', help='Type of wearable device')
        parser.add_argument('--survey-type', default='adherence', help='Type of survey data')
        args = parser.parse_args()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        # Initialize agent
        agent = AdherencePredictionAgent()
        
        # Prepare input data
        input_data = {
            "patient_id": args.patient_id,
            "start_date": start_date,
            "end_date": end_date,
            "device_type": args.device_type,
            "survey_type": args.survey_type
        }
        
        # Run agent
        logger.info(f"Running agent for patient {args.patient_id}")
        result = await agent.run(input_data)
        
        # Print results
        print("\nPrediction Results:")
        print(f"Risk Score: {result['prediction']['risk_score']:.2f}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("\nExplanation:")
        print(result['explanation'])
        
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 