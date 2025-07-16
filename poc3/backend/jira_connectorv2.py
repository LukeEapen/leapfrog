# filepath: c:\Users\73042C\luke\openai-assistant-clean\jira_connector.py
from jira import JIRA
from flask import Flask, request, render_template
import webbrowser
import threading
import logging
from dotenv import load_dotenv
import os

app = Flask(__name__)
app.logger.setLevel(logging.INFO)  # Add this line to set the logging level

# Replace these with your own
JIRA_SERVER = 'https://lalluluke.atlassian.net/'
EMAIL = 'lalluluke@gmail.com'

load_dotenv()
API_TOKEN = os.getenv("JIRA_API_TOKEN")

# Connect to Jira
jira = JIRA(server=JIRA_SERVER, basic_auth=(EMAIL, API_TOKEN))


@app.route('/create_epic', methods=['POST'])
def create_epic():

    fields = jira.fields()
    for f in fields:
        if 'epic' in f['name'].lower():
            print(f"{f['id']}: {f['name']}")

    project_key = request.form.get('project_key', 'SCRUM')
    summary = request.form.get('summary', 'Auto Epic Summary')
    description = request.form.get('description', 'Auto Description')
    epic_name = request.form.get('epic_name', 'Auto Epic Name')
    epic_desc = request.form.get('epic_desc', '')
    if epic_desc:
        description += f"\n\nEpic Details:\n{epic_desc}"
    # Get additional fields from the form
    assignee = request.form.get('assignee')  # Use .get() to avoid KeyError if not present
    reporter = request.form.get('reporter')
    priority = request.form.get('priority')
    labels = request.form.getlist('labels')  # Use getlist for multiple labels
    components = request.form.getlist('components') # Use getlist for multiple components

    issue_dict = {
        'project': {'key': project_key},
        'summary': summary,
        'description': description,
        'issuetype': {'name': 'Epic'}
        #,
       # 'customfield_10011': epic_name,  # Epic Name custom field (Jira Cloud default)
    }

    # Add optional fields to the issue dictionary if they exist
    if assignee:
        issue_dict['assignee'] = {'name': assignee}  # Assuming you pass the username/account ID
    if reporter:
        issue_dict['reporter'] = {'name': reporter}
    if priority:
        issue_dict['priority'] = {'name': priority}  # Or {'id': priority} depending on your setup
    if labels:
        issue_dict['labels'] = labels
    if components:
        issue_dict['components'] = [{'name': component} for component in components]

    try:
        new_issue = jira.create_issue(fields=issue_dict)
        return f"✅ Epic created: {new_issue.key}"
    except Exception as e:
        app.logger.error(f"Error creating epic: {e}")  # Log the full error
        return f"❌ Error creating epic: {e}", 500

@app.route('/create_story', methods=['POST'])
def create_story():
    """Create a user story in JIRA."""
    
    project_key = request.form.get('project_key', 'SCRUM')
    summary = request.form.get('summary', 'Auto Story Summary')
    description = request.form.get('description', 'Auto Description')
    assignee = request.form.get('assignee')
    reporter = request.form.get('reporter')
    priority = request.form.get('priority', 'High')
    labels = request.form.getlist('labels')
    components = request.form.getlist('components')
    epic_link = request.form.get('epic_link')  # Optional epic to link to
    
    issue_dict = {
        'project': {'key': project_key},
        'summary': summary,
        'description': description,
        'issuetype': {'name': 'Story'},
        'priority': {'name': priority}
    }
    
    # Add optional fields if they exist
    if assignee:
        issue_dict['assignee'] = {'name': assignee}
    if reporter:
        issue_dict['reporter'] = {'name': reporter}
    if labels:
        issue_dict['labels'] = labels
    if components:
        issue_dict['components'] = [{'name': component} for component in components]
    if epic_link:
        # Epic Link field - this might be a custom field depending on your JIRA setup
        issue_dict['customfield_10014'] = epic_link  # Common Epic Link field ID
    
    try:
        new_issue = jira.create_issue(fields=issue_dict)
        return f"✅ Story created: {new_issue.key}"
    except Exception as e:
        app.logger.error(f"Error creating story: {e}")
        return f"❌ Error creating story: {e}", 500


def create_story_programmatic(project_key, summary, description, priority='High', assignee=None, epic_link=None):
    """Create a user story programmatically (for use by other modules)."""
    
    issue_dict = {
        'project': {'key': project_key},
        'summary': summary,
        'description': description,
        'issuetype': {'name': 'Story'},
        'priority': {'name': priority}
    }
    
    if assignee:
        issue_dict['assignee'] = {'name': assignee}
    if epic_link:
        issue_dict['customfield_10014'] = epic_link  # Epic Link field
    
    try:
        new_issue = jira.create_issue(fields=issue_dict)
        return {'success': True, 'key': new_issue.key, 'id': new_issue.id}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/')
def index():
    return render_template('jira-interface.html')

def open_browser():
    webbrowser.open_new('http://127.0.0.1:8001/')

if __name__ == '__main__':
    threading.Timer(1.0, open_browser).start()
    app.run(port=8001)