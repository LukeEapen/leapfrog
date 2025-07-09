#!/usr/bin/env python3
"""
Test to simulate browser behavior for epic generation and retrieval.
"""

import requests
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_browser_like_flow():
    """Test that mimics browser behavior for the tabbed layout."""
    
    # Create a session to maintain cookies (like a browser)
    session = requests.Session()
    
    # Add browser-like headers
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin'
    })
    
    try:
        # Step 1: Load the tabbed layout page (to establish session)
        logger.info("Step 1: Loading tabbed layout page...")
        response = session.get("http://localhost:5001/tabbed-layout")
        
        if response.status_code == 200:
            logger.info("✅ Tabbed layout page loaded successfully")
        else:
            logger.error(f"❌ Failed to load tabbed layout: {response.status_code}")
            return
        
        # Step 2: Check initial epics (should be empty)
        logger.info("Step 2: Checking initial epics...")
        response = session.get("http://localhost:5001/get-epics")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('epics'):
                logger.warning(f"⚠️ Found {len(result.get('epics', []))} existing epics")
            else:
                logger.info("✅ No existing epics (as expected)")
        
        # Step 3: Generate epics (simulating the tabbed layout flow)
        logger.info("Step 3: Generating epics...")
        
        test_data = {
            "text": """User Authentication System
- Secure login and registration
- Password reset functionality
- User profile management

Dashboard Features
- Real-time analytics
- Customizable widgets
- Data export capabilities""",
            "input_type": "prd"
        }
        
        response = session.post(
            "http://localhost:5001/generate-epics-from-text",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
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
            logger.error(f"Response: {response.text}")
            return
        
        # Step 4: Immediately retrieve epics (like the browser would)
        logger.info("Step 4: Retrieving epics immediately after generation...")
        response = session.get("http://localhost:5001/get-epics")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('epics'):
                epic_count = len(result.get('epics', []))
                logger.info(f"✅ Successfully retrieved {epic_count} epics")
                logger.info("Epic titles:")
                for epic in result.get('epics', []):
                    logger.info(f"  - {epic.get('title', 'No title')}")
            else:
                logger.error("❌ No epics found after generation!")
                logger.error(f"Response: {json.dumps(result, indent=2)}")
        else:
            logger.error(f"❌ Get epics request failed: {response.status_code}")
        
        # Step 5: Wait a moment and try again (simulating user interaction delay)
        import time
        time.sleep(2)
        
        logger.info("Step 5: Retrieving epics after delay...")
        response = session.get("http://localhost:5001/get-epics")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('epics'):
                epic_count = len(result.get('epics', []))
                logger.info(f"✅ Still have {epic_count} epics after delay")
            else:
                logger.error("❌ Epics disappeared after delay!")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    logger.info("Testing browser-like flow for epic generation and retrieval...")
    test_browser_like_flow()
    logger.info("Test completed.")
