#!/usr/bin/env python3
"""
Debug script to test person data generation
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

from data_integration.simple_all_of_us_connector import SimpleAllOfUsConnector

async def debug_person_data():
    """Debug person data generation."""
    print("Testing person data generation...")
    
    connector = SimpleAllOfUsConnector()
    
    # Test person data generation directly
    print("1. Testing direct person data generation...")
    person_data = connector._generate_sample_person_data([1000, 1001, 1002])
    print(f"Person data shape: {person_data.shape}")
    print(f"Person data columns: {person_data.columns.tolist()}")
    print(f"Person data head:\n{person_data.head()}")
    
    # Test domain loading
    print("\n2. Testing domain loading...")
    data = await connector.load_domain('person', [1000, 1001, 1002])
    print(f"Loaded person data: {data is not None and not data.empty}")
    if data is not None and not data.empty:
        print(f"Loaded data shape: {data.shape}")
        print(f"Loaded data head:\n{data.head()}")
    else:
        print("No data loaded!")
    
    # Test with date filters
    print("\n3. Testing with date filters...")
    start_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now()
    data_with_dates = await connector.load_domain('person', [1000, 1001, 1002], start_date, end_date)
    print(f"Loaded with date filters: {data_with_dates is not None and not data_with_dates.empty}")
    
    # Test all domains
    print("\n4. Testing all domains...")
    all_data = await connector.load_all_domains(
        person_ids=[1000, 1001, 1002],
        start_date=start_date,
        end_date=end_date,
        domains=['person', 'fitbit_activity', 'survey']
    )
    print(f"All domains loaded: {len(all_data)} domains")
    for domain, df in all_data.items():
        print(f"  {domain}: {df.shape if df is not None else 'None'}")

if __name__ == "__main__":
    asyncio.run(debug_person_data()) 