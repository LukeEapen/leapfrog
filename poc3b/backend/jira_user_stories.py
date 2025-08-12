import os
from jira import JIRA
from dotenv import load_dotenv

def get_jira_connection():
    load_dotenv()
    JIRA_SERVER = os.getenv('JIRA_SERVER', 'https://lalluluke.atlassian.net/')
    EMAIL = os.getenv('JIRA_EMAIL', 'lalluluke@gmail.com')
    API_TOKEN = os.getenv('JIRA_API_TOKEN')
    jira = JIRA(server=JIRA_SERVER, basic_auth=(EMAIL, API_TOKEN))
    return jira

def fetch_user_stories(project_key='SCRUM'):
    jira = get_jira_connection()
    # JQL for user stories in the project
    jql = f"project={project_key} AND issuetype=Story ORDER BY created DESC"
    issues = jira.search_issues(jql, maxResults=20)
    user_stories = []
    for issue in issues:
        user_stories.append({
            'key': issue.key,
            'summary': issue.fields.summary,
            'description': getattr(issue.fields, 'description', ''),
        })
    return user_stories

if __name__ == '__main__':
    stories = fetch_user_stories()
    for s in stories:
        print(f"{s['key']}: {s['summary']}")
