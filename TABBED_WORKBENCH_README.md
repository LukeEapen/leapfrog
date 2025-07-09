# Tabbed Workbench System

This is a completely new tabbed interface for the Product Management Workbench, created as an alternative to the three-section layout.

## Features

### ✨ Tabbed Experience
- **Epic Tab**: Upload PRD and generate epics with integrated chat
- **User Stories Tab**: Select epic and generate user stories with integrated chat  
- **Story Details Tab**: View and refine story details with integrated chat

### 🗨️ Integrated Chat
- Chat functionality is built into each tab (no popups)
- Context-aware AI assistance at every step
- Real-time refinement and suggestions

### 🎨 Modern UI/UX
- Clean tabbed interface with smooth transitions
- Same design language as the three-section layout
- Responsive design for mobile and desktop
- Drag-and-drop file upload

### 🔄 Workflow
1. **Upload PRD** → Generate epics
2. **Select Epic** → Generate user stories  
3. **Select Story** → View details and submit to Jira

## Files

- `poc2_backend_processor_tabbed.py` - Backend Flask application
- `templates/poc2_tabbed_workbench.html` - Frontend HTML template
- `launch_tabbed_workbench.py` - Launcher script

## Quick Start

```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Set your OpenAI API key
set OPENAI_API_KEY=your_key_here

# Launch the tabbed workbench
python launch_tabbed_workbench.py
```

Then visit: http://localhost:5002/tabbed-layout

## Key Differences from Three-Section Layout

| Feature | Three-Section | Tabbed Workbench |
|---------|---------------|------------------|
| Layout | Three panels side-by-side | Tabbed interface |
| Chat | Modal popups | Integrated in each tab |
| Navigation | Click between sections | Tab navigation |
| Port | 5001 | 5002 |
| File | poc2_backend_processor_three_section.py | poc2_backend_processor_tabbed.py |

## Backend API Endpoints

- `GET /tabbed-layout` - Main tabbed interface
- `POST /tabbed-upload-prd` - Upload PRD and generate epics
- `POST /tabbed-select-epic` - Select epic and generate user stories
- `POST /tabbed-select-story` - Select story and load details
- `POST /tabbed-epic-chat` - Epic refinement chat
- `POST /tabbed-story-chat` - User story refinement chat  
- `POST /tabbed-details-chat` - Story details refinement chat
- `POST /tabbed-submit-jira` - Submit to Jira
- `GET /health` - Health check
- `GET /debug-info` - Debug information

## Architecture

The system is completely standalone and doesn't interfere with the existing three-section layout. It uses:

- **Flask** for the backend API
- **Bootstrap 5** for UI components
- **OpenAI GPT-4** for AI assistance
- **Font Awesome** for icons
- **Inter font** for typography

## Chat Integration

Each tab has its own chat interface that:
- Maintains context for that specific workflow step
- Provides AI assistance based on current state
- Offers real-time suggestions and improvements
- No modal popups - everything is inline

## Responsive Design

The interface adapts to different screen sizes:
- Desktop: Full tabbed layout with side chat
- Mobile: Stacked layout with collapsible chat

This provides the same functionality as the three-section layout but with a completely different user experience focused on tabbed navigation and integrated chat.
