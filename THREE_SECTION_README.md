# Three Section UI & Backend System

## Overview

This is a **new, independent** three-section user interface and backend system that replicates all features of the existing application while providing an improved user experience. The system operates completely separately from existing files and can run alongside the original application.

## Key Features

### 🎯 Three-Section Layout
- **Left Section**: List of Epics (clickable to select)
- **Middle Section**: User Stories for the selected epic (clickable to view details)
- **Right Section**: Detailed view of the selected user story with:
  - Acceptance criteria
  - Requirements traceability
  - Interactive chat for refinement
  - Jira submission capabilities

### 🔄 Complete Feature Parity
All features from the original system are available:
- ✅ Epic generation from requirements
- ✅ User story creation and refinement
- ✅ Interactive chat agents for all levels (Epic, User Story, Story Details)
- ✅ Jira integration and ticket submission
- ✅ System mapping capabilities
- ✅ RAG (Retrieval Augmented Generation) support
- ✅ Vector database integration
- ✅ File upload and processing

### 🚀 Modern Technology Stack
- **Backend**: Flask-based Python application (`poc2_backend_processor_three_section.py`)
- **Frontend**: Modern HTML5 with Bootstrap 4 and custom CSS (`templates/poc2_three_section_layout.html`)
- **Database**: ChromaDB for vector storage (optional)
- **AI**: OpenAI GPT models for all agent interactions

## Quick Start

### Option 1: Use the Startup Script (Recommended)
```bash
python start_three_section.py
```
This script will:
- Check system requirements
- Help set up environment variables
- Install dependencies (if needed)
- Start the server automatically

### Option 2: Manual Start
1. **Set Environment Variables**:
   ```bash
   set OPENAI_API_KEY=your_openai_api_key_here
   set FLASK_DEBUG=true
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Server**:
   ```bash
   python poc2_backend_processor_three_section.py
   ```

4. **Access the Application**:
   - Navigate to: http://localhost:5001/three-section
   - (Note: Uses port 5001 to avoid conflicts with existing app)

## File Structure

### New Files Created
```
📁 Project Root
├── 🆕 poc2_backend_processor_three_section.py  # New backend (port 5001)
├── 🆕 start_three_section.py                   # Startup helper script
├── 🆕 test_three_section_system.py             # Test suite
└── 📁 templates/
    └── 🆕 poc2_three_section_layout.html       # New three-section UI
```

### Existing Files (Unchanged)
```
📁 Project Root
├── 📄 poc2_backend_processor_optimized.py     # Original backend (untouched)
├── 📄 requirements.txt                        # Shared dependencies
├── 📁 agents/                                 # Shared agent definitions
├── 📁 templates/                              # Original templates (untouched)
└── 📁 vector_db/                             # Shared vector database
```

## API Endpoints

### Three-Section Specific Routes
- `GET /three-section` - Main three-section UI
- `POST /three-section-upload-prd` - Upload PRD and generate epics
- `POST /three-section-get-epics` - Get existing epics from session
- `POST /three-section-approve-epics` - Approve and process epics
- `POST /three-section-user-story-details` - Get user story details
- `POST /three-section-epic-chat` - Chat about epics
- `POST /three-section-user-story-chat` - Chat about user stories
- `POST /three-section-story-details-chat` - Chat about story details
- `POST /three-section-submit-jira` - Submit to Jira

### Shared Routes (from original system)
- `GET /system-mapping` - System mapping interface
- All agent and utility routes are preserved

## Testing

### Automated Testing
```bash
# Start the server first, then run:
python test_three_section_system.py
```

### Manual Testing
1. Start the application
2. Navigate to http://localhost:5001/three-section
3. Test the workflow:
   - Enter requirements → Generate epics
   - Select an epic → View user stories
   - Select a user story → View details
   - Use chat features for refinement
   - Submit to Jira (if configured)

## Configuration

### Environment Variables
- `OPENAI_API_KEY` - Required for AI features
- `FLASK_DEBUG` - Set to "true" for development
- `PORT` - Server port (default: 5001)

### Optional Features
- **Vector Database**: ChromaDB for enhanced RAG capabilities
- **Jira Integration**: Configure Jira credentials for ticket submission
- **System Mapping**: File upload and processing capabilities

## Deployment

### Development
```bash
python start_three_section.py
```

### Production
The system is designed to be deployment-ready with:
- Graceful handling of missing optional dependencies
- Environment-based configuration
- Error logging and monitoring
- Resource optimization for low-memory environments

## Architecture

### Backend Architecture
```
📊 poc2_backend_processor_three_section.py
├── 🔧 Core Flask Application
├── 🤖 Agent Integration Layer
├── 💾 Vector Database Layer (optional)
├── 🔗 API Route Handlers
└── 🛡️ Error Handling & Logging
```

### Frontend Architecture
```
🎨 poc2_three_section_layout.html
├── 📱 Responsive Three-Section Layout
├── ⚡ Interactive JavaScript
├── 🎭 Modal System (Chat, Jira)
├── 🔄 Dynamic Content Loading
└── 💎 Modern CSS with Design System
```

## Comparison with Original System

| Feature | Original System | Three-Section System |
|---------|----------------|---------------------|
| **UI Layout** | Multi-page workflow | Single-page three sections |
| **Navigation** | Page-to-page transitions | In-page section updates |
| **Port** | 5000 (default) | 5001 (no conflicts) |
| **Files** | Existing files modified | New files, no modifications |
| **Features** | Full feature set | ✅ Complete parity |
| **Dependencies** | Shared requirements.txt | ✅ Shared requirements.txt |
| **Agents** | Shared agent definitions | ✅ Shared agent definitions |
| **Database** | Shared vector DB | ✅ Shared vector DB |

## Support

### Troubleshooting
1. **Connection Refused**: Ensure the server is running on port 5001
2. **AI Features Not Working**: Check OPENAI_API_KEY environment variable
3. **Import Errors**: Run `pip install -r requirements.txt`
4. **Port Conflicts**: Both systems can run simultaneously (ports 5000 & 5001)

### Logs
- Application logs: `app.log`
- Error details: Console output when running with `FLASK_DEBUG=true`

## Development

### Adding New Features
1. **Backend**: Modify `poc2_backend_processor_three_section.py`
2. **Frontend**: Update `templates/poc2_three_section_layout.html`
3. **Testing**: Add tests to `test_three_section_system.py`

### Maintaining Compatibility
- The system shares `requirements.txt`, `agents/`, and `vector_db/` with the original
- No modifications to existing files ensures zero conflicts
- Independent operation allows safe experimentation

---

## 🎉 Success!

Your new three-section UI and backend system is ready to use! This modern interface provides an improved user experience while maintaining all the powerful features of the original system.

**Next Steps:**
1. Run `python start_three_section.py`
2. Navigate to http://localhost:5001/three-section
3. Start creating epics and user stories with the new interface!

Enjoy the enhanced workflow! 🚀
