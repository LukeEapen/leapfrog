#!/usr/bin/env python3
"""
Test script for System Mapping Chat functionality
"""

import requests
import json

def test_system_mapping_chat():
    """Test the system mapping chat endpoint"""
    
    # Test data
    test_message = "I need systems for credit card processing"
    test_data = {
        "message": test_message,
        "current_systems": [],
        "available_systems": [
            "Customer acquisition platform",
            "Credit decision engine", 
            "Card issuance manager",
            "Payment setup module"
        ]
    }
    
    try:
        # Make request to the chat endpoint
        response = requests.post(
            'http://127.0.0.1:5000/system-mapping-chat',
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== CHAT RESPONSE ===")
            print(f"Success: {result.get('success')}")
            print(f"Message: {result.get('message')}")
            
            if 'suggestions' in result:
                print(f"\nSuggestions ({len(result['suggestions'])}):")
                for i, suggestion in enumerate(result['suggestions'], 1):
                    print(f"  {i}. {suggestion.get('system')} - {suggestion.get('reason')}")
            
            if 'warnings' in result:
                print(f"\nWarnings ({len(result['warnings'])}):")
                for i, warning in enumerate(result['warnings'], 1):
                    print(f"  {i}. {warning}")
                    
            print("\n=== END RESPONSE ===")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    print("Testing System Mapping Chat Endpoint...")
    test_system_mapping_chat()
