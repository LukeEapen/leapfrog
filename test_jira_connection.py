#!/usr/bin/env python3
"""
JIRA Connection Test Script
This script tests your JIRA integration configuration.
Run this before using the main application to ensure JIRA is properly configured.
"""

import os
import sys
from dotenv import load_dotenv

def test_jira_connection():
    """Test JIRA connection with environment variables."""
    print("🔧 Testing JIRA Integration Setup...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    required_vars = {
        'JIRA_SERVER': os.getenv('JIRA_SERVER'),
        'JIRA_EMAIL': os.getenv('JIRA_EMAIL'),
        'JIRA_API_TOKEN': os.getenv('JIRA_API_TOKEN'),
        'JIRA_PROJECT_KEY': os.getenv('JIRA_PROJECT_KEY', 'SCRUM')
    }
    
    print("📋 Checking Environment Variables:")
    missing_vars = []
    for var, value in required_vars.items():
        if value:
            if var == 'JIRA_API_TOKEN':
                print(f"✅ {var}: {'*' * len(value[:10])}... (hidden)")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: NOT SET")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file and ensure all JIRA variables are set.")
        return False
    
    # Try to import and connect to JIRA
    try:
        from jira import JIRA
        print(f"\n🔌 Connecting to JIRA server: {required_vars['JIRA_SERVER']}")
        
        jira_client = JIRA(
            server=required_vars['JIRA_SERVER'],
            basic_auth=(required_vars['JIRA_EMAIL'], required_vars['JIRA_API_TOKEN'])
        )
        
        print("✅ JIRA connection successful!")
        
        # Test project access
        try:
            project = jira_client.project(required_vars['JIRA_PROJECT_KEY'])
            print(f"✅ Project access verified: {project.name} ({project.key})")
        except Exception as e:
            print(f"⚠️  Project access warning: {str(e)}")
            print(f"Please verify that project '{required_vars['JIRA_PROJECT_KEY']}' exists and you have access to it.")
        
        # Test issue types
        try:
            issue_types = jira_client.issue_types()
            story_type = next((it for it in issue_types if it.name.lower() == 'story'), None)
            if story_type:
                print(f"✅ Story issue type available: {story_type.name}")
            else:
                print("⚠️  'Story' issue type not found. Available types:")
                for issue_type in issue_types[:5]:  # Show first 5
                    print(f"   - {issue_type.name}")
        except Exception as e:
            print(f"⚠️  Could not retrieve issue types: {str(e)}")
        
        print("\n🎉 JIRA integration is ready to use!")
        return True
        
    except ImportError:
        print("❌ JIRA library not installed. Please run: pip install jira")
        return False
    except Exception as e:
        print(f"❌ JIRA connection failed: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Verify your JIRA_SERVER URL is correct")
        print("2. Check that your JIRA_EMAIL is correct")
        print("3. Ensure your JIRA_API_TOKEN is valid and not expired")
        print("4. Verify you have permissions to access the JIRA instance")
        return False

if __name__ == "__main__":
    print("JIRA Integration Test")
    print("=" * 50)
    
    success = test_jira_connection()
    
    if success:
        print("\n✅ All tests passed! Your JIRA integration is ready.")
        sys.exit(0)
    else:
        print("\n❌ JIRA integration test failed. Please check the errors above.")
        sys.exit(1)
