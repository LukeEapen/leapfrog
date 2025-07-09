#!/usr/bin/env python3
"""
Test session persistence between epic generation and retrieval.
"""

import requests
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_session_persistence():
    """Test that epics are stored and retrieved properly using session."""
    
    # Create a session to maintain cookies
    session = requests.Session()
    
    test_requirements = """
    User Authentication System
    - User registration with email validation
    - Secure login/logout functionality
    - Password reset capability
    
    Dashboard Features  
    - Real-time analytics display
    - Customizable widget system
    """
    
    try:
        # Step 1: Generate epics
        logger.info("Step 1: Generating epics...")
        url = "http://localhost:5001/generate-epics-from-text"
        
        payload = {
            "text": test_requirements,
            "input_type": "requirements"
        }
        
        response = session.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                epic_count = len(result.get('epics', []))
                story_count = len(result.get('user_stories', []))
                logger.info(f"✅ Generated {epic_count} epics and {story_count} user stories")
            else:
                logger.error(f"❌ Epic generation failed: {result.get('error')}")
                return
        else:
            logger.error(f"❌ Epic generation request failed: {response.status_code}")
            return
        
        # Step 2: Retrieve epics using same session
        logger.info("\nStep 2: Retrieving epics...")
        url = "http://localhost:5001/get-epics"
        
        response = session.get(url)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('epics'):
                epic_count = len(result.get('epics', []))
                logger.info(f"✅ Retrieved {epic_count} epics from session")
                logger.info("Epic titles:")
                for epic in result.get('epics', []):
                    logger.info(f"  - {epic.get('title', 'No title')}")
            else:
                logger.error("❌ No epics found in session")
                logger.error(f"Response: {json.dumps(result, indent=2)}")
        else:
            logger.error(f"❌ Get epics request failed: {response.status_code}")
        
        # Step 3: Test with a fresh session (should fail)
        logger.info("\nStep 3: Testing with fresh session (should fail)...")
        fresh_session = requests.Session()
        response = fresh_session.get("http://localhost:5001/get-epics")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('epics'):
                logger.warning("⚠️ Fresh session unexpectedly has epics (possible persistence issue)")
            else:
                logger.info("✅ Fresh session correctly has no epics")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    logger.info("Testing session persistence for epic generation and retrieval...")
    test_session_persistence()
    logger.info("Test completed.")
