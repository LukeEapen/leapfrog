"""
Test script to validate Jira integration functionality
"""
import os
import sys
from dotenv import load_dotenv

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_jira_connection():
    """Test Jira connection and basic operations"""
    print("🔧 Testing Jira Integration...")
    
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    api_token = os.getenv("JIRA_API_TOKEN")
    if not api_token:
        print("❌ JIRA_API_TOKEN not found in environment variables")
        print("   Please add JIRA_API_TOKEN to your .env file")
        return False
    
    try:
        from jira import JIRA
        print("✅ Jira library imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Jira library: {e}")
        print("   Run: pip install jira")
        return False
    
    # Test connection
    try:
        JIRA_SERVER = 'https://lalluluke.atlassian.net/'
        EMAIL = 'lalluluke@gmail.com'
        
        print(f"🔗 Connecting to Jira server: {JIRA_SERVER}")
        print(f"📧 Using email: {EMAIL}")
        
        jira = JIRA(server=JIRA_SERVER, basic_auth=(EMAIL, api_token))
        
        # Test by getting server info
        server_info = jira.server_info()
        print(f"✅ Connected to Jira successfully!")
        print(f"   Server: {server_info.get('serverTitle', 'Unknown')}")
        print(f"   Version: {server_info.get('version', 'Unknown')}")
        
        # Test by getting projects
        projects = jira.projects()
        print(f"📁 Found {len(projects)} projects:")
        for project in projects[:3]:  # Show first 3 projects
            print(f"   - {project.key}: {project.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Jira: {e}")
        print("   Please check your credentials and network connection")
        return False

def test_create_sample_ticket():
    """Test creating a sample ticket (optional)"""
    print("\n🎫 Testing Ticket Creation...")
    
    response = input("Do you want to create a test ticket? (y/N): ")
    if response.lower() != 'y':
        print("⏭️  Skipping ticket creation test")
        return True
    
    try:
        from jira import JIRA
        load_dotenv()
        
        JIRA_SERVER = 'https://lalluluke.atlassian.net/'
        EMAIL = 'lalluluke@gmail.com'
        API_TOKEN = os.getenv("JIRA_API_TOKEN")
        
        jira = JIRA(server=JIRA_SERVER, basic_auth=(EMAIL, API_TOKEN))
        
        # Create test issue
        issue_dict = {
            'project': {'key': 'SCRUM'},
            'summary': 'Test User Story from Python Integration',
            'description': 'This is a test user story created to validate the Jira integration.\n\n*Acceptance Criteria:*\n• Test criterion 1\n• Test criterion 2\n\n*Systems:* Test System',
            'issuetype': {'name': 'Story'},
            'priority': {'name': 'Medium'}
        }
        
        print("📝 Creating test ticket...")
        new_issue = jira.create_issue(fields=issue_dict)
        print(f"✅ Test ticket created successfully!")
        print(f"   Ticket Key: {new_issue.key}")
        print(f"   URL: {JIRA_SERVER}browse/{new_issue.key}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create test ticket: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("🧪 JIRA INTEGRATION TEST SUITE")
    print("=" * 50)
    
    # Test connection
    connection_ok = test_jira_connection()
    
    if connection_ok:
        # Test ticket creation (optional)
        test_create_sample_ticket()
        
        print("\n" + "=" * 50)
        print("✅ Jira integration is working correctly!")
        print("   You can now submit user stories to Jira.")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ Jira integration test failed!")
        print("   Please fix the issues above before using Jira features.")
        print("=" * 50)

if __name__ == "__main__":
    main()
