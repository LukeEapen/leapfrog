#!/usr/bin/env python3
"""
Test script to verify the Flask backend Jira submission endpoint works correctly.
"""

import requests
import json

def test_flask_jira_submission():
    """Test the Flask backend Jira submission endpoint with POST data."""
    try:
        # Test data matching the form fields expected by the backend
        form_data = {
            'epic_title': 'Test Epic from Flask Test',
            'user_story_name': 'Test User Story from Flask Backend',
            'user_story_description': 'This is a test user story submitted through the Flask backend to verify the Jira integration works correctly.',
            'priority': 'High',
            'responsible_systems': 'Backend API, Jira Integration',
            'acceptance_criteria': 'The ticket should be created successfully|||No errors should occur|||The ticket should appear in Jira',
            'tagged_requirements': 'API Integration|||Error Handling|||User Story Management'
        }
        
        print("🧪 Testing Flask Jira submission endpoint...")
        print(f"Submitting data: {json.dumps(form_data, indent=2)}")
        
        # Submit to the Flask backend
        response = requests.post(
            'http://localhost:5000/submit-jira-ticket',
            data=form_data,
            timeout=30
        )
        
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ Successfully submitted to Flask backend!")
            print("Response content preview:")
            print(response.text[:500])
            if 'ticket' in response.text.lower() or 'success' in response.text.lower():
                print("🎉 Looks like Jira ticket was created successfully!")
            return True
        else:
            print(f"❌ Flask backend returned error: {response.status_code}")
            print(f"Response content: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("Starting Flask Jira submission test...")
    success = test_flask_jira_submission()
    
    if success:
        print("🎉 Test completed successfully!")
    else:
        print("💥 Test failed!")
