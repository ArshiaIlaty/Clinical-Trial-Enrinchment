# Clinical Trial Adherence Prediction System

An agentic AI system for early prediction and explanation of clinical trial adherence using multimodal patient data.

## Overview

This system integrates multiple data sources to predict and explain clinical trial adherence:
- Wearable device data (heart rate, sleep patterns, activity levels)
- Electronic Health Records (EHRs)
- Self-reported data
- Behavioral indicators

## Features

- Real-time monitoring of patient health and behavior
- Early prediction of potential non-adherence
- Explainable AI (XAI) for clinical reasoning
- Autonomous data integration and analysis
- Privacy-compliant data handling (HIPAA, GDPR)

## Project Structure

```
clinical_trial_agent/
├── agents/                 # Core agent implementations
├── data_integration/       # Data source connectors
├── models/                 # ML models for prediction
├── explainability/         # XAI components
├── api/                    # API endpoints
├── utils/                  # Utility functions
└── tests/                  # Test suite
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

1. Start the API server:
```bash
uvicorn api.main:app --reload
```

2. Run the agent:
```bash
python -m agents.main
```

## Development

- Format code: `black .`
- Sort imports: `isort .`
- Run tests: `pytest`

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests. 
