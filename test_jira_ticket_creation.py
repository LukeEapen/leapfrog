#!/usr/bin/env python3
"""
Test script to verify Jira ticket creation after removing priority field.
"""

import os
import sys
from dotenv import load_dotenv
from jira import JIRA
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_jira_ticket_creation():
    """Test creating a Jira ticket without the priority field."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Jira configuration
        JIRA_SERVER = 'https://lalluluke.atlassian.net/'
        EMAIL = 'lalluluke@gmail.com'
        API_TOKEN = os.getenv("JIRA_API_TOKEN")
        
        if not API_TOKEN:
            logger.error("JIRA_API_TOKEN not found in environment variables")
            return False
        
        logger.info("Connecting to Jira...")
        
        # Connect to Jira
        jira = JIRA(server=JIRA_SERVER, basic_auth=(EMAIL, API_TOKEN))
        
        # Test ticket data
        test_summary = "Test User Story - Priority Field Removed"
        test_description = """This is a test user story to verify Jira ticket creation works after removing the priority field.

*Acceptance Criteria:*
• The ticket should be created successfully
• No 400 errors should occur
• The ticket should appear in the SCRUM project

*Tagged Requirements:*
• Jira API integration
• Error handling

*Responsible Systems:* Backend API, Jira Integration"""
        
        # Create issue dictionary (without priority field)
        issue_dict = {
            'project': {'key': 'SCRUM'},
            'summary': test_summary,
            'description': test_description,
            'issuetype': {'name': 'Story'}
        }
        
        logger.info("Creating test Jira ticket...")
        
        # Create the Jira issue
        new_issue = jira.create_issue(fields=issue_dict)
        
        logger.info(f"✅ Successfully created Jira ticket: {new_issue.key}")
        logger.info(f"Ticket URL: {JIRA_SERVER}browse/{new_issue.key}")
        
        # Get and display ticket details
        issue = jira.issue(new_issue.key)
        logger.info(f"Ticket Summary: {issue.fields.summary}")
        logger.info(f"Ticket Status: {issue.fields.status}")
        logger.info(f"Issue Type: {issue.fields.issuetype}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating Jira ticket: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Print additional error details if available
        if hasattr(e, 'response'):
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        
        return False

if __name__ == "__main__":
    logger.info("Starting Jira ticket creation test...")
    success = test_jira_ticket_creation()
    
    if success:
        logger.info("🎉 Test completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Test failed!")
        sys.exit(1)
