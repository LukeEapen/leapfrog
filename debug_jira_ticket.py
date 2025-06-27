"""
Debug script to test Jira ticket creation with exact parameters used in backend
"""
import os
from dotenv import load_dotenv
from jira import JIRA

def debug_jira_ticket_creation():
    """Test ticket creation with the exact same parameters as the backend"""
    print("🐛 Debugging Jira Ticket Creation...")
    
    # Load environment variables
    load_dotenv()
    
    # Jira configuration (same as backend)
    JIRA_SERVER = 'https://lalluluke.atlassian.net/'
    EMAIL = 'lalluluke@gmail.com'
    API_TOKEN = os.getenv("JIRA_API_TOKEN")
    
    if not API_TOKEN:
        print("❌ JIRA_API_TOKEN not found")
        return
    
    try:
        # Connect to Jira
        jira = JIRA(server=JIRA_SERVER, basic_auth=(EMAIL, API_TOKEN))
        print("✅ Connected to Jira successfully")
        
        # Test 1: List all projects to verify project key
        print("\n📁 Available Projects:")
        projects = jira.projects()
        for project in projects:
            print(f"   - {project.key}: {project.name}")
        
        # Test 2: List all issue types
        print("\n🎫 Available Issue Types:")
        # Get issue types for SCRUM project specifically
        project = jira.project('SCRUM')
        for issue_type in project.issueTypes:
            print(f"   - {issue_type.name} (ID: {issue_type.id})")
        
        # Test 3: List available priorities
        print("\n⚡ Available Priorities:")
        priorities = jira.priorities()
        for priority in priorities:
            print(f"   - {priority.name} (ID: {priority.id})")
        
        # Test 4: Try creating a ticket with the exact same parameters as backend
        print("\n🎯 Testing Ticket Creation (same as backend)...")
        
        # Same issue dict as in the backend
        issue_dict = {
            'project': {'key': 'SCRUM'},
            'summary': 'Test User Story from Debug Script',
            'description': 'This is a test user story created from the debug script to identify the 404 error.',
            'issuetype': {'name': 'Story'},
            'priority': {'name': 'High'}
        }
        
        print(f"Issue dict: {issue_dict}")
        
        # Try to create the issue
        new_issue = jira.create_issue(fields=issue_dict)
        print(f"✅ Successfully created ticket: {new_issue.key}")
        print(f"   URL: {JIRA_SERVER}browse/{new_issue.key}")
        
        # Delete the test ticket to keep things clean
        try:
            new_issue.delete()
            print("🗑️  Test ticket deleted successfully")
        except Exception as delete_error:
            print(f"⚠️  Could not delete test ticket (may need manual cleanup): {delete_error}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # If it's a JIRAError, get more details
        if hasattr(e, 'status_code'):
            print(f"   Status Code: {e.status_code}")
        if hasattr(e, 'text'):
            print(f"   Error Text: {e.text}")
        if hasattr(e, 'response'):
            print(f"   Response: {e.response}")

if __name__ == "__main__":
    debug_jira_ticket_creation()
