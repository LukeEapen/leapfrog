#!/usr/bin/env python3
"""
Test script to validate chat functionality for the three-section layout.
This script tests all chat endpoints to ensure they work correctly.
"""

import requests
import json
import sys

# Test configuration
BASE_URL = "http://localhost:5000"
TEST_TIMEOUT = 10

def test_epic_chat():
    """Test epic chat endpoint"""
    print("Testing Epic Chat...")
    
    test_epics = [
        {
            "id": "epic-1",
            "title": "User Authentication System",
            "description": "Implement secure user authentication",
            "priority": "High"
        }
    ]
    
    test_data = {
        "message": "Can you help me improve the authentication epic?",
        "epics": test_epics
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/three-section-epic-chat",
            json=test_data,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Epic Chat: PASSED")
                print(f"   Response: {data.get('response', 'No response')[:100]}...")
                return True
            else:
                print(f"❌ Epic Chat: FAILED - {data.get('error')}")
                return False
        else:
            print(f"❌ Epic Chat: FAILED - HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Epic Chat: FAILED - {str(e)}")
        return False

def test_user_story_chat():
    """Test user story chat endpoint"""
    print("Testing User Story Chat...")
    
    test_user_stories = [
        {
            "id": "us-1",
            "title": "User Login",
            "description": "As a user, I want to log in securely",
            "priority": "High"
        }
    ]
    
    test_epic = {
        "id": "epic-1",
        "title": "User Authentication System"
    }
    
    test_data = {
        "message": "Can you help me refine the user login story?",
        "user_stories": test_user_stories,
        "epic": test_epic
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/three-section-user-story-chat",
            json=test_data,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ User Story Chat: PASSED")
                print(f"   Response: {data.get('response', 'No response')[:100]}...")
                return True
            else:
                print(f"❌ User Story Chat: FAILED - {data.get('error')}")
                return False
        else:
            print(f"❌ User Story Chat: FAILED - HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ User Story Chat: FAILED - {str(e)}")
        return False

def test_story_details_chat():
    """Test story details chat endpoint"""
    print("Testing Story Details Chat...")
    
    test_story_details = {
        "acceptance_criteria": ["User can enter username and password", "System validates credentials"],
        "tagged_requirements": ["REQ-001: Authentication", "REQ-002: Security"],
        "traceability_matrix": "# Traceability\n- Login requirement maps to authentication epic"
    }
    
    test_user_story = {
        "id": "us-1",
        "title": "User Login",
        "description": "As a user, I want to log in securely"
    }
    
    test_epic = {
        "id": "epic-1",
        "title": "User Authentication System"
    }
    
    test_data = {
        "message": "Can you add more acceptance criteria for the login story?",
        "section": "criteria",
        "story_details": test_story_details,
        "user_story": test_user_story,
        "epic": test_epic
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/three-section-story-details-chat",
            json=test_data,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Story Details Chat: PASSED")
                print(f"   Response: {data.get('response', 'No response')[:100]}...")
                return True
            else:
                print(f"❌ Story Details Chat: FAILED - {data.get('error')}")
                return False
        else:
            print(f"❌ Story Details Chat: FAILED - HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Story Details Chat: FAILED - {str(e)}")
        return False

def test_server_availability():
    """Test if server is running"""
    print("Testing Server Availability...")
    
    try:
        response = requests.get(f"{BASE_URL}/three-section", timeout=5)
        if response.status_code == 200:
            print("✅ Server: AVAILABLE")
            return True
        else:
            print(f"❌ Server: UNAVAILABLE - HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server: UNAVAILABLE - {str(e)}")
        return False

def main():
    """Run all chat functionality tests"""
    print("=" * 50)
    print("CHAT FUNCTIONALITY VALIDATION")
    print("=" * 50)
    
    # Check server availability first
    if not test_server_availability():
        print("\n❌ Server is not available. Please start the Flask server first.")
        print("Run: flask run --port=5000")
        sys.exit(1)
    
    print()
    
    # Run chat tests
    results = []
    results.append(test_epic_chat())
    results.append(test_user_story_chat())
    results.append(test_story_details_chat())
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All chat functionality tests PASSED!")
        return 0
    else:
        print("⚠️  Some chat functionality tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
