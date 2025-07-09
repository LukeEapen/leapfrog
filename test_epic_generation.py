#!/usr/bin/env python3
"""
Test script for epic generation functionality
"""

import requests
import json

def test_epic_generation():
    """Test the epic generation endpoint."""
    print("Testing epic generation...")
    
    # Test data
    test_data = {
        "text": """
        Project: Customer Management System
        
        Requirements:
        - User registration and authentication
        - Customer profile management
        - Order tracking and history
        - Payment processing
        - Customer support ticketing
        - Reporting and analytics
        - Mobile application support
        - Integration with external systems
        """,
        "input_type": "requirements"
    }
    
    try:
        # Test the epic generation endpoint
        url = "http://localhost:5001/generate-epics-from-text"
        
        print(f"Sending request to: {url}")
        print(f"Test data: {json.dumps(test_data, indent=2)}")
        
        response = requests.post(url, json=test_data, timeout=30)
        
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                epics = result.get('epics', [])
                print(f"✅ Successfully generated {len(epics)} epics!")
                for i, epic in enumerate(epics, 1):
                    print(f"\nEpic {i}:")
                    print(f"  Title: {epic.get('title')}")
                    print(f"  Priority: {epic.get('priority')}")
                    print(f"  Description: {epic.get('description', '')[:100]}...")
            else:
                print(f"❌ Epic generation failed: {result.get('error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the Flask app is running on localhost:5001")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_epic_generation()
