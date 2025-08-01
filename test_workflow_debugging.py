#!/usr/bin/env python3
"""
Test script to trigger the workflow and verify the enhanced logging for product_overview and feature_overview passing to Agent 4.X
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_workflow():
    """Test the complete workflow to see enhanced logging."""
    
    # Create a session
    session = requests.Session()
    
    # Step 1: Login
    print("Step 1: Logging in...")
    login_data = {
        'username': 'admin',  # You'll need to set your actual credentials
        'password': 'admin123'  # Update with your actual password
    }
    
    login_response = session.post(f"{BASE_URL}/", data=login_data)
    if login_response.status_code != 200 or 'login' in login_response.url:
        print("❌ Login failed. Please check credentials in .env file")
        return
    
    print("✅ Login successful")
    
    # Step 2: Submit Page 1 with test data
    print("Step 2: Submitting page 1 data...")
    page1_data = {
        'industry': 'Technology',
        'sector': 'Software',
        'geography': 'North America',
        'intent': 'Create a customer management system',
        'features': 'User authentication, customer database, reporting dashboard'
    }
    
    page1_response = session.post(f"{BASE_URL}/page1", data=page1_data)
    if page1_response.status_code != 200:
        print(f"❌ Page 1 submission failed: {page1_response.status_code}")
        return
    
    print("✅ Page 1 submitted - agents should be running in background")
    
    # Step 3: Wait and check Page 2
    print("Step 3: Waiting for agents to complete and checking page 2...")
    time.sleep(10)  # Wait for background agents
    
    page2_response = session.get(f"{BASE_URL}/page2")
    if page2_response.status_code != 200:
        print(f"❌ Page 2 access failed: {page2_response.status_code}")
        return
    
    print("✅ Page 2 accessed - should show agent outputs")
    
    # Step 4: Submit page 2 to proceed to page 3
    print("Step 4: Proceeding to page 3...")
    page2_submit = session.post(f"{BASE_URL}/page2", data={
        'product_overview': 'Test product overview',
        'feature_overview': 'Test feature overview'
    })
    
    if page2_submit.status_code != 200:
        print(f"❌ Page 2 submission failed: {page2_submit.status_code}")
        return
    
    print("✅ Page 2 submitted")
    
    # Step 5: Submit page 3 to trigger Agent 4.X calls
    print("Step 5: Submitting page 3 to trigger Agent 4.X calls...")
    page3_response = session.post(f"{BASE_URL}/page3")
    
    if page3_response.status_code != 200:
        print(f"❌ Page 3 submission failed: {page3_response.status_code}")
        return
    
    print("✅ Page 3 submitted - Agent 4.X should be running with enhanced logging")
    print("📝 Check the app.log file and terminal output for detailed logging:")
    print("   - [PAGE3] Session ID and data keys")
    print("   - [PAGE3] Product/Feature overview existence and lengths")
    print("   - [PAGE3] Combined input excerpts")
    print("   - [INPUT CHECK] Agent X - Product Overview present: True/False")
    
    # Step 6: Wait a bit then check page 4
    print("Step 6: Waiting for Agent 4.X to complete...")
    time.sleep(15)  # Wait for Agent 4.X to complete
    
    page4_response = session.get(f"{BASE_URL}/page4")
    if page4_response.status_code == 200:
        print("✅ Page 4 accessed - workflow complete")
    else:
        print(f"⚠️  Page 4 access returned: {page4_response.status_code}")
    
    print("\n🔍 To debug the issue, check these logs:")
    print("1. Terminal output where Flask is running")
    print("2. app.log file for detailed logging")
    print("3. Look for these specific log patterns:")
    print("   - [PAGE3] Product overview exists: True")
    print("   - [PAGE3] Feature overview exists: True") 
    print("   - [INPUT CHECK] Agent X - Product Overview present: True")
    print("   - [INPUT CHECK] Agent X - Feature Overview present: True")

if __name__ == "__main__":
    test_workflow()
