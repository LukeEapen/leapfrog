#!/usr/bin/env python3
"""
Test script to verify epic generation functionality.
"""

import requests
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_epic_generation():
    """Test epic generation from text input."""
    
    test_requirements = """
User Authentication System
- User registration with email validation
- Secure login/logout functionality
- Password reset capability
- User profile management

Dashboard Features
- Real-time analytics display
- Customizable widget system
- Data visualization charts
- Export functionality

API Integration
- RESTful API endpoints
- Authentication and authorization
- Rate limiting
- Documentation
"""
    
    try:
        url = "http://localhost:5001/generate-epics-from-text"
        
        # Test JSON payload
        payload = {
            "text": test_requirements,
            "input_type": "requirements"
        }
        
        logger.info(f"Testing epic generation with {len(test_requirements)} characters")
        logger.info(f"URL: {url}")
        logger.info(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, json=payload, timeout=60)
        
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ Epic generation successful")
            logger.info(f"Response: {json.dumps(result, indent=2)}")
            
            if result.get('success'):
                epic_count = len(result.get('epics', []))
                story_count = len(result.get('user_stories', []))
                logger.info(f"✅ Generated {epic_count} epics and {story_count} user stories")
            else:
                logger.error(f"❌ Epic generation failed: {result.get('error', 'Unknown error')}")
                
        else:
            logger.error(f"❌ Request failed with status {response.status_code}")
            logger.error(f"Response text: {response.text}")
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Could not connect to server. Make sure Flask app is running on port 5001")
    except requests.exceptions.Timeout:
        logger.error("❌ Request timed out. The epic generation might be taking too long.")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")

def test_get_epics():
    """Test getting generated epics."""
    try:
        url = "http://localhost:5001/get-epics"
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ Get epics successful")
            logger.info(f"Epics data: {json.dumps(result, indent=2)}")
        else:
            logger.error(f"❌ Get epics failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Get epics test failed: {e}")

if __name__ == "__main__":
    logger.info("Starting epic generation tests...")
    test_epic_generation()
    logger.info("\nTesting get epics...")
    test_get_epics()
    logger.info("Tests completed.")
