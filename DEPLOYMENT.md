# Three Section JIRA UI - Deployment Guide

## Quick Start (Local Development)
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
set OPENAI_API_KEY=your_key_here
set FLASK_ENV=development

# Start the application
python launch_three_section.py
```

Access the app at: `http://localhost:5001/three-section`

## Production Deployment Options

### 1. Render.com
- Push code to GitHub
- Connect repository to Render
- The `render.yaml` is already configured
- Set environment variables in Render dashboard:
  - `OPENAI_API_KEY`
  - `JIRA_SERVER` (if using JIRA integration)
  - `JIRA_USERNAME` (if using JIRA integration)
  - `JIRA_API_TOKEN` (if using JIRA integration)

### 2. Heroku
```bash
# Deploy to Heroku
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your_key_here
heroku config:set FLASK_ENV=production
git push heroku main
```

### 3. Docker
```bash
# Build and run with Docker
docker build -t three-section-ui .
docker run -p 5001:5001 -e OPENAI_API_KEY=your_key_here three-section-ui
```

### 4. Railway
- Connect GitHub repository
- Set environment variables in Railway dashboard
- Railway will automatically detect the Python app

### 5. Google Cloud Run / AWS / Azure
- Use the Dockerfile for container deployment
- Set environment variables in the cloud platform
- Deploy container to cloud service

## Environment Variables

### Required
- `OPENAI_API_KEY` - Your OpenAI API key for AI features

### Optional
- `PORT` - Port to run on (default: 5001)
- `FLASK_ENV` - Environment (development/production)
- `JIRA_SERVER` - JIRA server URL for ticket creation
- `JIRA_USERNAME` - JIRA username
- `JIRA_API_TOKEN` - JIRA API token

## File Structure
- `launch_three_section.py` - Production-ready launcher (recommended)
- `poc2_backend_processor_three_section.py` - Main Flask backend
- `templates/poc2_three_section_layout.html` - Frontend UI
- `render.yaml` - Render.com deployment config
- `Procfile` - Heroku deployment config
- `Dockerfile` - Docker deployment config

## Health Checks
- Health endpoint: `/health`
- Debug info: `/debug-info` (development only)

## Troubleshooting
- Check `three_section_debug.log` for detailed logs
- Ensure all environment variables are set
- Verify all dependencies are installed from `requirements.txt`
