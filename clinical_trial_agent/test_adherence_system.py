#!/usr/bin/env python3
"""
Test script for the Clinical Trial Adherence Analysis System

This script tests the basic functionality of the adherence labeling system
with sample data to ensure everything works correctly.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

from data_integration.simple_all_of_us_connector import SimpleAllOfUsConnector as AllOfUsConnector
from models.adherence_labeler import AllOfUsAdherenceLabeler, AdherenceConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_basic_functionality():
    """Test basic functionality of the adherence system."""
    logger.info("Testing basic functionality...")
    
    try:
        # Initialize components
        connector = AllOfUsConnector()
        labeler = AllOfUsAdherenceLabeler()
        
        # Test data loading
        logger.info("Testing data loading...")
        data = await connector.load_all_domains(
            person_ids=[1000, 1001, 1002],  # 3 test participants
            start_date=datetime.now() - timedelta(days=90),  # 3 months
            end_date=datetime.now(),
            domains=['person', 'fitbit_activity', 'survey']
        )
        
        # Verify data was loaded
        assert 'person' in data, "Person data not loaded"
        assert 'fitbit_activity' in data, "Fitbit activity data not loaded"
        assert 'survey' in data, "Survey data not loaded"
        
        logger.info(f"✓ Data loading successful: {len(data)} domains loaded")
        
        # Test multi-class labeling
        logger.info("Testing multi-class labeling...")
        multi_class_labels = labeler.create_multi_class_labels(
            person_data=data,
            study_period_days=90
        )
        
        # Verify labels were created
        assert not multi_class_labels.empty, "Multi-class labels are empty"
        assert 'person_id' in multi_class_labels.columns, "person_id column missing"
        assert 'adherence_level' in multi_class_labels.columns, "adherence_level column missing"
        assert 'adherence_score' in multi_class_labels.columns, "adherence_score column missing"
        
        logger.info(f"✓ Multi-class labeling successful: {len(multi_class_labels)} labels created")
        
        # Test state-based labeling
        logger.info("Testing state-based labeling...")
        state_labels = labeler.create_state_based_labels(
            person_data=data,
            start_date=datetime.now() - timedelta(days=90),
            end_date=datetime.now()
        )
        
        # Verify state labels were created
        assert not state_labels.empty, "State labels are empty"
        assert 'person_id' in state_labels.columns, "person_id column missing"
        assert 'date' in state_labels.columns, "date column missing"
        assert 'adherence_state' in state_labels.columns, "adherence_state column missing"
        
        logger.info(f"✓ State-based labeling successful: {len(state_labels)} state observations created")
        
        # Test pattern analysis
        logger.info("Testing pattern analysis...")
        analysis = labeler.analyze_adherence_patterns(multi_class_labels, data)
        
        # Verify analysis was performed
        assert 'summary_stats' in analysis, "Summary stats missing from analysis"
        assert 'demographic_analysis' in analysis, "Demographic analysis missing"
        assert 'behavioral_patterns' in analysis, "Behavioral patterns missing"
        assert 'risk_factors' in analysis, "Risk factors missing"
        
        logger.info("✓ Pattern analysis successful")
        
        # Print summary statistics
        logger.info("\n" + "="*50)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*50)
        
        # Multi-class label distribution
        level_counts = multi_class_labels['adherence_level'].value_counts()
        logger.info("Multi-class Label Distribution:")
        for level, count in level_counts.items():
            percentage = (count / len(multi_class_labels)) * 100
            logger.info(f"  {level.capitalize()}: {count} participants ({percentage:.1f}%)")
        
        # Average adherence scores
        avg_scores = multi_class_labels.groupby('adherence_level')['adherence_score'].mean()
        logger.info("\nAverage Adherence Scores:")
        for level, score in avg_scores.items():
            logger.info(f"  {level.capitalize()}: {score:.3f}")
        
        # State distribution
        state_counts = state_labels['adherence_state'].value_counts()
        logger.info("\nState Distribution:")
        for state, count in state_counts.items():
            percentage = (count / len(state_labels)) * 100
            logger.info(f"  {state.capitalize()}: {count} observations ({percentage:.1f}%)")
        
        # Data summary
        summary = connector.get_data_summary(data)
        logger.info("\nData Summary:")
        for domain, stats in summary.items():
            logger.info(f"  {domain}: {stats['record_count']} records, {stats['person_count']} persons")
        
        logger.info("\n" + "="*50)
        logger.info("ALL TESTS PASSED SUCCESSFULLY!")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

async def test_configuration():
    """Test custom configuration functionality."""
    logger.info("Testing custom configuration...")
    
    try:
        # Create custom configuration
        config = AdherenceConfig(
            fitbit_coverage_high=0.9,    # 90% instead of 80%
            fitbit_coverage_medium=0.5,  # 50% instead of 40%
            fitbit_coverage_low=0.3,     # 30% instead of 20%
            survey_completion_high=4,    # 4+ surveys instead of 3
            survey_completion_medium=2,  # 2+ surveys instead of 1
            inactive_gap=5,              # 5 days instead of 7
            exit_gap=90                  # 90 days instead of 180
        )
        
        # Initialize with custom config
        labeler = AllOfUsAdherenceLabeler(config)
        connector = AllOfUsConnector()
        
        # Load test data
        data = await connector.load_all_domains(
            person_ids=[1000, 1001, 1002],
            start_date=datetime.now() - timedelta(days=90),
            end_date=datetime.now()
        )
        
        # Test with custom configuration
        labels = labeler.create_multi_class_labels(data, study_period_days=90)
        
        # Verify configuration was applied
        config_summary = labeler.get_labeling_summary()
        assert config_summary['config']['fitbit_coverage_thresholds']['high'] == 0.9
        assert config_summary['config']['survey_completion_thresholds']['high'] == 4
        
        logger.info("✓ Custom configuration test successful")
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {str(e)}")
        return False

async def test_data_validation():
    """Test data validation functionality."""
    logger.info("Testing data validation...")
    
    try:
        connector = AllOfUsConnector()
        
        # Test with valid data
        data = await connector.load_all_domains(
            person_ids=[1000, 1001, 1002],
            start_date=datetime.now() - timedelta(days=90),
            end_date=datetime.now()
        )
        
        # Test data summary
        summary = connector.get_data_summary(data)
        assert isinstance(summary, dict), "Data summary should be a dictionary"
        
        # Test data validation for each domain
        for domain, df in data.items():
            if not df.empty:
                # Test that required columns are present
                schema_info = connector.schema_info[domain]
                for col in schema_info['required_columns']:
                    assert col in df.columns, f"Required column {col} missing from {domain}"
        
        logger.info("✓ Data validation test successful")
        return True
        
    except Exception as e:
        logger.error(f"Data validation test failed: {str(e)}")
        return False

async def test_export_functionality():
    """Test export functionality."""
    logger.info("Testing export functionality...")
    
    try:
        # Initialize components
        connector = AllOfUsConnector()
        labeler = AllOfUsAdherenceLabeler()
        
        # Load data and create labels
        data = await connector.load_all_domains(
            person_ids=[1000, 1001, 1002],
            start_date=datetime.now() - timedelta(days=90),
            end_date=datetime.now()
        )
        
        multi_class_labels = labeler.create_multi_class_labels(data, study_period_days=90)
        
        # Test CSV export
        test_output_dir = Path("test_output")
        test_output_dir.mkdir(exist_ok=True)
        
        csv_path = test_output_dir / "test_labels.csv"
        labeler.export_labels(multi_class_labels, str(csv_path), format='csv')
        
        # Verify file was created
        assert csv_path.exists(), "CSV file was not created"
        
        # Test JSON export
        json_path = test_output_dir / "test_labels.json"
        labeler.export_labels(multi_class_labels, str(json_path), format='json')
        
        # Verify file was created
        assert json_path.exists(), "JSON file was not created"
        
        # Test data summary export
        summary_path = test_output_dir / "test_summary.json"
        connector.export_data_summary(data, str(summary_path))
        
        # Verify file was created
        assert summary_path.exists(), "Summary file was not created"
        
        # Clean up test files
        for file_path in [csv_path, json_path, summary_path]:
            file_path.unlink()
        test_output_dir.rmdir()
        
        logger.info("✓ Export functionality test successful")
        return True
        
    except Exception as e:
        logger.error(f"Export functionality test failed: {str(e)}")
        return False

async def run_all_tests():
    """Run all tests."""
    logger.info("Starting comprehensive test suite...")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Configuration", test_configuration),
        ("Data Validation", test_data_validation),
        ("Export Functionality", test_export_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUITE SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! The adherence system is working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
        return False

def main():
    """Main function to run the test suite."""
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in test suite: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 