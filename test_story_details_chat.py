import requests
import json

# Test the story details chat endpoint
BASE_URL = "http://localhost:5000"

def test_story_details_chat():
    test_data = {
        "message": "Can you improve the acceptance criteria to be more specific and detailed?",
        "section": "acceptance_criteria",
        "story_details": {
            "acceptance_criteria": [
                "User can create account",
                "System validates data"
            ],
            "tagged_requirements": [
                "REQ-001: User registration",
                "REQ-002: Data validation"
            ],
            "traceability_matrix": "Basic traceability available"
        },
        "user_story": {
            "title": "User Account Creation",
            "description": "As a user, I want to create an account so that I can access the system"
        },
        "epic": {
            "title": "Customer Onboarding",
            "description": "Enable new customers to onboard and access services"
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/three-section-story-details-chat",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        
        if response.status_code == 200:
            data = response.json()
            print("\nResponse Data:")
            print(json.dumps(data, indent=2))
            
            if data.get("success"):
                print(f"\nAI Response: {data.get('response', 'No response')}")
                print(f"\nUpdated Story Details: {data.get('updated_story_details', 'No updates')}")
            else:
                print(f"Error: {data.get('error', 'Unknown error')}")
        else:
            print(f"HTTP Error: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {str(e)}")

if __name__ == "__main__":
    test_story_details_chat()
