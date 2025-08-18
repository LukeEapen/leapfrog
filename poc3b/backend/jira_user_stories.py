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

def fetch_user_stories(project_key: str = 'SCRUM', startAt: int = 0, maxResults: int = 20):
    """Fetch user stories with simple pagination.

    Returns a dict containing stories plus paging metadata (total/startAt/maxResults).
    """
    jira = get_jira_connection()
    # JQL for user stories in the project
    jql = f"project={project_key} AND issuetype=Story ORDER BY created DESC"
    # Support pagination
    issues = jira.search_issues(jql, startAt=startAt, maxResults=maxResults)
    user_stories = []
    for issue in issues:
        user_stories.append({
            'key': issue.key,
            'summary': getattr(issue.fields, 'summary', ''),
            'description': getattr(issue.fields, 'description', ''),
        })
    # JIRA's ResultList exposes total
    total = getattr(issues, 'total', len(user_stories))
    return {
        'stories': user_stories,
        'total': int(total) if isinstance(total, (int, float)) else len(user_stories),
        'startAt': int(startAt) if isinstance(startAt, (int, float)) else 0,
        'maxResults': int(maxResults) if isinstance(maxResults, (int, float)) else 20,
    }

if __name__ == '__main__':
    stories = fetch_user_stories()
    for s in stories:
        print(f"{s['key']}: {s['summary']}")
