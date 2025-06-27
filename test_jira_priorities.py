#!/usr/bin/env python3
"""
Test script to check JIRA priorities and project configuration.
This script helps debug JIRA integration issues.
"""
import os
from jira import JIRA
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_jira_connection():
    """Test JIRA connection and list available priorities."""
    try:
        # Load environment variables
        JIRA_SERVER = os.getenv('JIRA_SERVER')
        JIRA_EMAIL = os.getenv('JIRA_EMAIL')
        JIRA_API_TOKEN = os.getenv('JIRA_API_TOKEN')
        JIRA_PROJECT_KEY = os.getenv('JIRA_PROJECT_KEY', 'SCRUM')
        
        print("JIRA Configuration:")
        print(f"  Server: {JIRA_SERVER}")
        print(f"  Email: {JIRA_EMAIL}")
        print(f"  API Token: {'***' if JIRA_API_TOKEN else 'Not set'}")
        print(f"  Project Key: {JIRA_PROJECT_KEY}")
        print()
        
        if not all([JIRA_SERVER, JIRA_EMAIL, JIRA_API_TOKEN]):
            print("❌ Missing JIRA configuration. Please check your environment variables.")
            return False
        
        # Connect to JIRA
        print("Connecting to JIRA...")
        jira_client = JIRA(
            server=JIRA_SERVER,
            basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN)
        )
        print("✅ Connected to JIRA successfully!")
        print()
        
        # Test project access
        try:
            project = jira_client.project(JIRA_PROJECT_KEY)
            print(f"✅ Project '{JIRA_PROJECT_KEY}' found: {project.name}")
        except Exception as e:
            print(f"❌ Project '{JIRA_PROJECT_KEY}' not found: {e}")
            return False
        print()
        
        # Get available priorities
        try:
            priorities = jira_client.priorities()
            print("Available priorities:")
            for i, priority in enumerate(priorities, 1):
                print(f"  {i}. {priority.name} (ID: {priority.id})")
            print()
        except Exception as e:
            print(f"❌ Could not retrieve priorities: {e}")
            return False
        
        # Get available issue types
        try:
            issue_types = jira_client.issue_types()
            print("Available issue types:")
            for i, issue_type in enumerate(issue_types, 1):
                print(f"  {i}. {issue_type.name} (ID: {issue_type.id})")
            print()
        except Exception as e:
            print(f"❌ Could not retrieve issue types: {e}")
        
        # Test creating a basic issue (just validate, don't actually create)
        try:
            print("Testing issue creation fields...")
            available_priorities = [p.name for p in priorities]
            
            # Test different priority values
            test_priorities = ['Medium', 'High', 'Low', 'Normal', 'Major', 'Minor']
            for test_priority in test_priorities:
                if test_priority in available_priorities:
                    print(f"  ✅ Priority '{test_priority}' is available")
                    break
            else:
                print(f"  ⚠️  None of the common priorities found. Using first available: '{available_priorities[0]}'")
                
        except Exception as e:
            print(f"❌ Error testing issue creation: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to JIRA: {e}")
        return False

if __name__ == "__main__":
    print("JIRA Configuration Test")
    print("=" * 50)
    
    # Load environment variables from .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded environment variables from .env file")
    except ImportError:
        print("⚠️  python-dotenv not installed. Loading environment variables manually.")
    except FileNotFoundError:
        print("⚠️  .env file not found. Using system environment variables.")
    
    print()
    
    success = test_jira_connection()
    
    if success:
        print("✅ JIRA configuration test completed successfully!")
    else:
        print("❌ JIRA configuration test failed. Please check your settings.")
