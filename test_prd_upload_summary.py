#!/usr/bin/env python3
"""
Test script to verify PRD upload functionality and content summary generation.
"""

import requests
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_prd_upload():
    """Test PRD upload with content that needs summarization."""
    
    # Create test PRD content that's long enough to trigger summarization
    test_prd_content = """
PROJECT REQUIREMENTS DOCUMENT

OVERVIEW
This is a comprehensive PRD for testing the upload and summarization functionality.
The content is designed to be long enough to trigger the content summary feature.

OBJECTIVES
- Test file upload functionality
- Verify content summarization works correctly
- Ensure session storage is working
- Validate epic generation preparation

REQUIREMENTS
1. User Authentication System
   - Login functionality with email/password
   - OAuth integration with Google and Microsoft
   - Password reset functionality
   - User profile management

2. Dashboard Features
   - Real-time analytics dashboard
   - Customizable widget system
   - Data export functionality
   - Notification center

3. API Integration
   - RESTful API endpoints
   - Rate limiting and authentication
   - Data validation and error handling
   - API documentation

FEATURES
- Multi-tenant architecture support
- Advanced search and filtering
- Mobile-responsive design
- Offline mode capabilities
- Real-time collaboration tools
- Audit logging and compliance

USER STORIES
As a user, I want to be able to log in securely so that I can access my personalized dashboard.
As an admin, I want to manage user permissions so that I can control access to sensitive features.
As a developer, I want clear API documentation so that I can integrate with the system effectively.

TECHNICAL SPECIFICATIONS
- Frontend: React with TypeScript
- Backend: Python Flask/FastAPI
- Database: PostgreSQL with Redis cache
- Authentication: JWT tokens
- Deployment: Docker containers on AWS

This document contains enough content to test the summarization functionality while maintaining
realistic PRD structure and content.
""" * 3  # Multiply to make it long enough to trigger summarization

    try:
        # Test the upload endpoint
        url = "http://localhost:5001/three-section-document-upload"
        
        # Create file-like object
        prd_file = io.BytesIO(test_prd_content.encode('utf-8'))
        
        files = {
            'prd_file': ('test_prd.txt', prd_file, 'text/plain')
        }
        
        logger.info(f"Testing PRD upload with {len(test_prd_content)} characters")
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ PRD upload successful")
            logger.info(f"Response: {result}")
            
            # Check if summary was created
            if 'prd_summary' in str(result):
                logger.info("✅ Content summary functionality working")
            else:
                logger.warning("⚠️ Content summary not found in response")
                
        else:
            logger.error(f"❌ Upload failed with status {response.status_code}")
            logger.error(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Could not connect to server. Make sure Flask app is running on port 5001")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")

def test_session_content():
    """Test getting session content after upload."""
    try:
        url = "http://localhost:5001/get-session-content"
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ Session content retrieval successful")
            logger.info(f"Session data keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        else:
            logger.error(f"❌ Session content retrieval failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Session content test failed: {e}")

if __name__ == "__main__":
    logger.info("Starting PRD upload and content summary tests...")
    test_prd_upload()
    test_session_content()
    logger.info("Tests completed.")
