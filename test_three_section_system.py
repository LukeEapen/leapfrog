#!/usr/bin/env python3
"""
Test script for the new Three Section UI and Backend System
This script tests the new poc2_backend_processor_three_section.py system
"""

import requests
import json
import time
import os
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:5001"  # New backend runs on port 5001
TEST_TIMEOUT = 30

def test_endpoint(endpoint, method="GET", data=None, expected_status=200):
    """Test a specific endpoint and return the result."""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n{'='*50}")
    print(f"Testing: {method} {endpoint}")
    print(f"{'='*50}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=TEST_TIMEOUT)
        elif method == "POST":
            if endpoint == "/three-section-upload-prd":
                # Handle PRD upload with form data
                response = requests.post(url, data=data, timeout=TEST_TIMEOUT)
            else:
                # Handle JSON data
                response = requests.post(url, json=data, timeout=TEST_TIMEOUT)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
            
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == expected_status:
            print("✅ Status code matches expected")
        else:
            print(f"❌ Expected {expected_status}, got {response.status_code}")
            return False
            
        # Try to parse JSON response if possible
        try:
            json_response = response.json()
            print("✅ Valid JSON response")
            if isinstance(json_response, dict) and json_response.get('success'):
                print(f"Response status: Success")
                if 'epics' in json_response:
                    epics = json_response['epics']
                    print(f"Generated {len(epics)} epics")
            elif isinstance(json_response, dict) and 'error' in json_response:
                print(f"Response error: {json_response['error']}")
        except:
            print("ℹ️  Response is not JSON (likely HTML)")
            
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - is the server running?")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Run all tests for the three-section system."""
    print("🚀 Testing Three Section UI and Backend System")
    print(f"Target URL: {BASE_URL}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        # Core UI endpoints
        ("Three Section Layout UI", "/three-section", "GET", None, 200),
        
        # PRD Upload and Epic Generation
        ("PRD Upload API", "/three-section-upload-prd", "POST", 
         {"requirements": "Create a basic user management system with authentication, profiles, and admin dashboard"}, 200),
        
        # Epic management endpoints
        ("Get Epics API", "/three-section-get-epics", "GET", None, 200),
        
        ("Approve Epics API", "/three-section-approve-epics", "POST", 
         {"approved_epics": ["Epic 1: User Registration"]}, 200),
        
        ("User Story Details API", "/three-section-user-story-details", "POST", 
         {"epic_title": "Epic 1: User Registration", "user_story_title": "User Story 1"}, 200),
         
        # Chat endpoints
        ("Epic Chat API", "/three-section-epic-chat", "POST", 
         {"message": "Tell me more about this epic", "epic_context": "User management"}, 200),
         
        ("User Story Chat API", "/three-section-user-story-chat", "POST", 
         {"message": "What are the requirements?", "user_story_context": "User registration"}, 200),
         
        ("Story Details Chat API", "/three-section-story-details-chat", "POST", 
         {"message": "Explain the acceptance criteria", "story_context": "Login functionality"}, 200),
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for test_name, endpoint, method, data, expected_status in tests:
        print(f"\n📝 Test: {test_name}")
        success = test_endpoint(endpoint, method, data, expected_status)
        results.append((test_name, success))
        
        if success:
            passed += 1
            print("✅ PASSED")
        else:
            failed += 1
            print("❌ FAILED")
            
        time.sleep(1)  # Small delay between tests
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    
    print(f"\n📋 DETAILED RESULTS:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    if failed == 0:
        print(f"\n🎉 All tests passed! The three-section system is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check the server logs for details.")
        
    return failed == 0

if __name__ == "__main__":
    print("Starting three-section system tests...")
    print("\n🔧 Prerequisites:")
    print("1. Make sure to set your OPENAI_API_KEY environment variable")
    print("2. Start the backend server: python poc2_backend_processor_three_section.py")
    print("3. Wait for the server to be ready before running this test")
    
    input("\n⏸️  Press Enter when the server is ready...")
    
    success = main()
    exit(0 if success else 1)
