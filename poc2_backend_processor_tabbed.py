#!/usr/bin/env python3
"""
Tabbed Layout Backend Processor - Complete Standalone System
Based on the working three-section system but with tabbed UI experience
"""

import os
import sys
import logging
import traceback
import json
import tempfile
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
from openai import OpenAI
from werkzeug.utils import secure_filename
import docx
import PyPDF2
import io

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tabbed_backend.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production-very-long-secret")

# Configure Flask-Session for persistent sessions
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'tabbed_workbench:'
app.config['SESSION_FILE_DIR'] = os.path.join(tempfile.gettempdir(), 'tabbed_workbench_sessions')

# Ensure session directory exists
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Initialize Flask-Session
Session(app)

# Constants
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'md', 'csv'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_read(file_obj):
    """Safely read file content with proper encoding and format handling."""
    try:
        if not file_obj:
            return ""
        
        filename = secure_filename(file_obj.filename)
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        # Read file content based on extension
        if file_ext == 'txt' or file_ext == 'md':
            content = file_obj.read().decode('utf-8', errors='ignore')
        elif file_ext == 'docx':
            doc = docx.Document(file_obj)
            content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        elif file_ext == 'pdf':
            pdf_reader = PyPDF2.PdfReader(file_obj)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        elif file_ext == 'csv':
            content = file_obj.read().decode('utf-8', errors='ignore')
        else:
            content = file_obj.read().decode('utf-8', errors='ignore')
        
        return content.strip()
        
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        return ""

@app.route("/")
def landing_page():
    """Landing page with navigation options."""
    return render_template("landing_page.html")

@app.route("/tabbed-layout", methods=["GET"])
def tabbed_layout():
    """Main tabbed layout page."""
    try:
        logger.info("GET request to /tabbed-layout")
        logger.info(f"Current session keys: {list(session.keys())}")
        logger.info(f"Has epics in session: {'generated_epics' in session}")
        logger.info(f"Epics count: {len(session.get('generated_epics', []))}")
        
        # Don't clear session data - preserve user's work
        # Only clear session if explicitly requested via query parameter
        if request.args.get('clear_session') == 'true':
            logger.info("Clearing session data due to clear_session parameter")
            session.pop('generated_epics', None)
            session.pop('current_epic', None)
            session.pop('generated_user_stories', None)
            session.pop('current_user_story', None)
            session.pop('story_details', None)
            session.modified = True
        
        return render_template("poc2_tabbed_workbench.html")
    except Exception as e:
        logger.error(f"Error loading tabbed layout: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_configured": bool(os.getenv('OPENAI_API_KEY'))
    })

@app.route("/debug-info", methods=["GET"])
def debug_info():
    """Debug information endpoint."""
    session_dir = app.config.get('SESSION_FILE_DIR', 'Not configured')
    session_files = []
    if os.path.exists(session_dir):
        session_files = os.listdir(session_dir)
    
    return jsonify({
        "environment": os.environ.get('FLASK_ENV', 'development'),
        "openai_key_set": bool(os.getenv('OPENAI_API_KEY')),
        "session_type": app.config.get('SESSION_TYPE', 'default'),
        "session_directory": session_dir,
        "session_files_count": len(session_files),
        "session_data": {
            "has_epics": 'generated_epics' in session,
            "has_user_stories": 'generated_user_stories' in session,
            "current_epic": session.get('current_epic', {}).get('title', 'None') if session.get('current_epic') else 'None',
            "session_keys": list(session.keys()),
            "epics_count": len(session.get('generated_epics', [])),
            "session_id": request.cookies.get('session')
        }
    })

@app.route("/debug-session", methods=["GET"])
def debug_session():
    """Debug session data."""
    try:
        epics = session.get('generated_epics', [])
        return jsonify({
            "session_id": request.cookies.get('session', 'None'),
            "has_epics": 'generated_epics' in session,
            "epics_count": len(epics),
            "current_epic": session.get('current_epic', {}).get('title', 'None') if session.get('current_epic') else 'None',
            "session_keys": list(session.keys()),
            "session_data": {
                key: (f"<{len(value)} items>" if isinstance(value, list) else str(value)[:100])
                for key, value in session.items()
            },
            "cookies": dict(request.cookies),
            "epic_ids": [epic.get('id') for epic in epics] if epics else []
        })
    except Exception as e:
        logger.error(f"Debug session error: {str(e)}")
        return jsonify({
            "error": str(e),
            "session_keys": list(session.keys()) if session else [],
            "cookies": dict(request.cookies)
        })

# Epic Generation Functions
def create_epic_generation_context(prd_content, additional_content="", context_notes=""):
    """Create enhanced context for epic generation."""
    context = f"""
    You are an expert product manager and business analyst. Based on the provided Product Requirements Document (PRD), generate 3-5 comprehensive epics that break down the requirements into manageable chunks.

    PRD Content:
    {prd_content}
    """
    
    if additional_content:
        context += f"\n\nAdditional Documentation:\n{additional_content}"
    
    if context_notes:
        context += f"\n\nUser Context & Instructions:\n{context_notes}"
    
    context += """
    
    Please generate epics in the following format:

    Epic 1: [Epic Title]
    Description: [Detailed description of what this epic covers]
    Priority: High/Medium/Low
    
    Epic 2: [Epic Title]
    Description: [Detailed description of what this epic covers]
    Priority: High/Medium/Low
    
    Continue this format for all epics (aim for 3-5 epics total).
    
    Each epic should:
    - Represent a significant business capability or user journey
    - Be independently deliverable
    - Have clear business value
    - Be appropriately scoped (not too big, not too small)
    """
    
    return context

def create_epic_generation_context_with_system_mapping(prd_content, system_mapping_content, context_notes=""):
    """Create enhanced context for epic generation with both PRD and system mapping."""
    context = f"""
    You are an expert product manager and business analyst. Based on the provided Product Requirements Document (PRD) and System Mapping documentation, generate 3-5 comprehensive epics that break down the requirements into manageable chunks.

    Consider both the business requirements from the PRD and the technical system architecture from the System Mapping to create well-informed epics.

    PRD Content:
    {prd_content}

    System Mapping Content:
    {system_mapping_content}
    """
    
    if context_notes:
        context += f"\n\nUser Context & Instructions:\n{context_notes}"
    
    context += """
    
    Please generate epics in the following format:

    Epic 1: [Epic Title]
    Description: [Detailed description of what this epic covers, considering both business needs and system architecture]
    Priority: High/Medium/Low
    
    Epic 2: [Epic Title]
    Description: [Detailed description of what this epic covers, considering both business needs and system architecture]
    Priority: High/Medium/Low
    
    Continue this format for all epics (aim for 3-5 epics total).
    
    Each epic should:
    - Represent a significant business capability or user journey
    - Be independently deliverable
    - Have clear business value
    - Consider the existing system architecture and constraints
    - Be appropriately scoped (not too big, not too small)
    - Take into account system integration points and dependencies
    """
    
    return context

@app.route("/tabbed-upload-files", methods=["POST"])
def tabbed_upload_files():
    """Handle PRD and System Mapping upload and generate epics for tabbed layout."""
    try:
        logger.info("POST request to /tabbed-upload-files")
        
        # Handle file uploads
        prd_file = request.files.get('prd_file')
        system_mapping_file = request.files.get('system_mapping_file')
        context_notes = request.form.get('context_notes', '')
        
        if not prd_file:
            return jsonify({"success": False, "error": "PRD file is required"})
        
        # Process file contents
        prd_content = safe_read(prd_file)
        system_mapping_content = ""
        
        if system_mapping_file and system_mapping_file.filename:
            system_mapping_content = safe_read(system_mapping_file)
            logger.info(f"System mapping file provided: {system_mapping_file.filename}")
        else:
            logger.info("No system mapping file provided")
        
        if not prd_content:
            return jsonify({"success": False, "error": "Could not extract content from PRD file"})
        
        # Store in session with explicit modification
        session['prd_content'] = prd_content
        session['system_mapping_content'] = system_mapping_content
        session['context_notes'] = context_notes
        session.modified = True  # Ensure session is saved
        
        logger.info(f"Session updated - Keys: {list(session.keys())}")
        
        # Create combined content
        combined_content = prd_content
        if system_mapping_content:
            combined_content += f"\n\n--- System Mapping ---\n{system_mapping_content}"
        if context_notes:
            combined_content += f"\n\n--- User Context ---\n{context_notes}"
        
        session['combined_content'] = combined_content
        session.modified = True  # Ensure session is saved
        
        # Generate epics using OpenAI - choose context function based on available files
        if system_mapping_content:
            enhanced_context = create_epic_generation_context_with_system_mapping(prd_content, system_mapping_content, context_notes)
            context_message = "Generate comprehensive epics that break down requirements into manageable chunks, considering both the PRD and system mapping context."
        else:
            enhanced_context = create_epic_generation_context(prd_content, "", context_notes)
            context_message = "Generate comprehensive epics that break down requirements into manageable chunks."
        
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are an expert product manager and business analyst. {context_message}"},
                    {"role": "user", "content": enhanced_context}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            epics_text = response.choices[0].message.content
            epics = parse_epics_from_response(epics_text)
            
            if epics:
                session['generated_epics'] = epics
                session['combined_content'] = combined_content  # Store content for user story generation
                session.modified = True  # Ensure session is saved
                files_used = "PRD and System Mapping" if system_mapping_content else "PRD"
                logger.info(f"Generated {len(epics)} epics successfully from {files_used}")
                logger.info(f"Session after storing epics - Keys: {list(session.keys())}")
                logger.info(f"Stored epics count: {len(session.get('generated_epics', []))}")
                logger.info(f"Session ID: {request.cookies.get('session', 'None')}")
                logger.info(f"Epics data being returned: {epics}")
                
                # Double-check the epics are valid
                for i, epic in enumerate(epics):
                    logger.info(f"Epic {i}: ID={epic.get('id')}, Title={epic.get('title')}, Desc length={len(epic.get('description', ''))}")
                
                return jsonify({
                    "success": True,
                    "epics": epics,
                    "message": f"Successfully generated {len(epics)} epics from {files_used}"
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to parse epics from OpenAI response"
                })
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Failed to generate epics. Please check your OpenAI API key."
            })
            
    except Exception as e:
        logger.error(f"Error in file upload: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

def parse_epics_from_response(epics_text):
    """Parse epics from OpenAI response text."""
    try:
        epics = []
        epic_sections = epics_text.split('Epic ')
        
        logger.info(f"Raw OpenAI response for epics: {epics_text[:500]}...")  # Log first 500 chars
        logger.info(f"Split into {len(epic_sections)} sections")
        
        epic_counter = 1  # Start from 1 for cleaner IDs
        
        for i, section in enumerate(epic_sections):
            if i == 0:  # Skip the first split part (usually empty or preamble)
                continue
                
            lines = section.strip().split('\n')
            if not lines:
                continue
                
            # Extract epic number and title from first line
            first_line = lines[0]
            if ':' in first_line:
                title_part = first_line.split(':', 1)[1].strip()
            else:
                title_part = first_line.strip()
            
            # Find description and priority
            description = ""
            priority = "Medium"
            
            for line in lines[1:]:
                line = line.strip()
                if line.startswith('Description:'):
                    description = line.replace('Description:', '').strip()
                elif line.startswith('Priority:'):
                    priority = line.replace('Priority:', '').strip()
                elif not line.startswith('Epic') and not line.startswith('Priority:') and description == "":
                    if line and not line.lower().startswith('priority'):
                        description = line
            
            if title_part and description:
                epic_id = f'epic_{epic_counter}'
                epic = {
                    'id': epic_id,
                    'title': title_part,
                    'description': description,
                    'priority': priority,
                    'estimated_stories': 'TBD',
                    'estimated_effort': 'TBD'
                }
                epics.append(epic)
                logger.info(f"Created epic {epic_id}: {title_part}")
                epic_counter += 1
        
        logger.info(f"Parsed {len(epics)} epics from response")
        return epics
        
    except Exception as e:
        logger.error(f"Error parsing epics: {str(e)}")
        return []

@app.route("/tabbed-select-epic", methods=["POST"])
def tabbed_select_epic():
    """Handle epic selection and generate user stories."""
    try:
        logger.info("POST request to /tabbed-select-epic")
        
        data = request.get_json()
        epic_id = data.get('epic_id')
        
        logger.info(f"Received epic_id: {epic_id}")
        logger.info(f"Session keys: {list(session.keys())}")
        
        if not epic_id:
            return jsonify({"success": False, "error": "Epic ID is required"})
        
        # Get the selected epic from session
        epics = session.get('generated_epics', [])
        logger.info(f"Session epics count: {len(epics)}")
        logger.info(f"Available epic IDs: {[epic.get('id') for epic in epics]}")
        logger.info(f"All epics data: {epics}")
        
        if not epics:
            logger.error(f"No epics in session. Session keys: {list(session.keys())}")
            return jsonify({
                "success": False, 
                "error": "No epics found in session. Please generate epics first.",
                "debug_info": {
                    "session_keys": list(session.keys()),
                    "session_id": request.cookies.get('session', 'None'),
                    "has_prd_content": 'prd_content' in session,
                    "has_combined_content": 'combined_content' in session
                }
            })
        
        selected_epic = None
        
        for epic in epics:
            if epic.get('id') == epic_id:
                selected_epic = epic
                break
        
        if not selected_epic:
            logger.error(f"Epic with ID {epic_id} not found in {[epic.get('id') for epic in epics]}")
            return jsonify({"success": False, "error": f"Selected epic not found. Available epics: {[epic.get('id') for epic in epics]}"})
        
        # Store selected epic in session
        session['current_epic'] = selected_epic
        
        # Generate user stories for the selected epic
        prd_content = session.get('combined_content', '')
        user_stories = generate_user_stories_for_epic(selected_epic, prd_content)
        
        if user_stories:
            session['generated_user_stories'] = user_stories
            logger.info(f"Generated {len(user_stories)} user stories for epic: {selected_epic.get('title')}")
            
            return jsonify({
                "success": True,
                "epic": selected_epic,
                "user_stories": user_stories,
                "message": f"Generated {len(user_stories)} user stories"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to generate user stories for the selected epic"
            })
            
    except Exception as e:
        logger.error(f"Error selecting epic: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

def generate_user_stories_for_epic(epic, prd_content):
    """Generate user stories for a specific epic."""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = f"""
        Based on the following PRD content and the specific epic, generate detailed user stories.

        PRD Content:
        {prd_content[:3000] if prd_content else "No additional PRD content available"}

        Epic Details:
        Title: {epic.get('title', '')}
        Description: {epic.get('description', '')}

        Please generate 5-8 user stories that specifically support this epic. For each user story, provide:

        User Story 1: [Story Title]
        Description: As a [user type], I want [goal] so that [benefit]
        Acceptance Criteria:
        - [Specific, testable criterion 1]
        - [Specific, testable criterion 2]
        - [Specific, testable criterion 3]
        Priority: High/Medium/Low
        Effort: Small/Medium/Large

        User Story 2: [Story Title]
        Description: As a [user type], I want [goal] so that [benefit]
        Acceptance Criteria:
        - [Specific, testable criterion 1]
        - [Specific, testable criterion 2]
        - [Specific, testable criterion 3]
        Priority: High/Medium/Low
        Effort: Small/Medium/Large

        Continue this format for all user stories (aim for 5-8 stories).
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert product manager and business analyst. Generate detailed, actionable user stories that align with the given epic and PRD requirements."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        user_stories = parse_user_stories_from_response(content, epic)
        
        return user_stories
        
    except Exception as e:
        logger.error(f"Error generating user stories for epic: {str(e)}")
        return []

def parse_user_stories_from_response(content, epic):
    """Parse user stories from OpenAI response."""
    try:
        user_stories = []
        story_sections = content.split('User Story ')
        
        for i, section in enumerate(story_sections):
            if i == 0:  # Skip the first split part
                continue
                
            lines = section.strip().split('\n')
            if not lines:
                continue
            
            # Extract story number and title from first line
            first_line = lines[0]
            if ':' in first_line:
                title_part = first_line.split(':', 1)[1].strip()
            else:
                title_part = first_line.strip()
            
            # Parse story details
            description = ""
            acceptance_criteria = []
            priority = "Medium"
            effort = "Medium"
            
            current_section = None
            for line in lines[1:]:
                line = line.strip()
                if line.startswith('Description:'):
                    description = line.replace('Description:', '').strip()
                    current_section = 'description'
                elif line.startswith('Acceptance Criteria:'):
                    current_section = 'criteria'
                elif line.startswith('Priority:'):
                    priority = line.replace('Priority:', '').strip()
                    current_section = None
                elif line.startswith('Effort:'):
                    effort = line.replace('Effort:', '').strip()
                    current_section = None
                elif line.startswith('- ') and current_section == 'criteria':
                    acceptance_criteria.append(line[2:])
                elif current_section == 'description' and line and not line.startswith('Acceptance') and not line.startswith('Priority') and not line.startswith('Effort'):
                    if description:
                        description += " " + line
                    else:
                        description = line
            
            if title_part and description:
                story = {
                    'id': f'story_{i}',
                    'title': title_part,
                    'description': description,
                    'acceptance_criteria': '\n'.join([f"• {criteria}" for criteria in acceptance_criteria]) if acceptance_criteria else "Acceptance criteria to be defined",
                    'priority': priority,
                    'estimated_effort': effort,
                    'epic_id': epic.get('id', 'epic_1'),
                    'epic_title': epic.get('title', '')
                }
                user_stories.append(story)
        
        logger.info(f"Parsed {len(user_stories)} user stories from response")
        return user_stories
        
    except Exception as e:
        logger.error(f"Error parsing user stories: {str(e)}")
        return []

@app.route("/tabbed-select-story", methods=["POST"])
def tabbed_select_story():
    """Handle user story selection and generate detailed story information."""
    try:
        logger.info("POST request to /tabbed-select-story")
        
        data = request.get_json()
        story_id = data.get('story_id')
        
        if not story_id:
            return jsonify({"success": False, "error": "Story ID is required"})
        
        # Get the selected story from session
        user_stories = session.get('generated_user_stories', [])
        selected_story = None
        
        for story in user_stories:
            if story.get('id') == story_id:
                selected_story = story
                break
        
        if not selected_story:
            return jsonify({"success": False, "error": "Selected story not found"})
        
        # Store selected story in session
        session['current_user_story'] = selected_story
        
        # Generate enhanced story details
        story_details = generate_story_details(selected_story)
        session['story_details'] = story_details
        
        logger.info(f"Generated details for story: {selected_story.get('title')}")
        
        return jsonify({
            "success": True,
            "story": selected_story,
            "story_details": story_details,
            "message": "Story details generated successfully"
        })
        
    except Exception as e:
        logger.error(f"Error selecting story: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

def generate_story_details(user_story):
    """Generate enhanced details for a user story."""
    try:
        # For now, return the basic story information
        # This can be enhanced with additional OpenAI calls for more detailed analysis
        
        current_epic = session.get('current_epic', {})
        
        story_details = {
            'title': user_story.get('title', ''),
            'description': user_story.get('description', ''),
            'acceptance_criteria': user_story.get('acceptance_criteria', ''),
            'priority': user_story.get('priority', 'Medium'),
            'estimated_effort': user_story.get('estimated_effort', 'Medium'),
            'epic_title': current_epic.get('title', 'Unknown Epic'),
            'epic_description': current_epic.get('description', ''),
            'responsible_systems': 'To be determined',
            'tagged_requirements': 'To be determined',
            'traceability_matrix': 'To be generated'
        }
        
        return story_details
        
    except Exception as e:
        logger.error(f"Error generating story details: {str(e)}")
        return {}

# Chat functionality endpoints
@app.route("/tabbed-epic-chat", methods=["POST"])
def tabbed_epic_chat():
    """Handle epic chat interactions."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({"success": False, "error": "Message is required"})
        
        # Get current epics context
        epics = session.get('generated_epics', [])
        
        if not epics:
            return jsonify({"success": False, "error": "No epics available for refinement"})
        
        # Create context for the chat
        epics_context = "Current Epics:\n"
        for i, epic in enumerate(epics, 1):
            epics_context += f"{i}. {epic.get('title', '')}\n   {epic.get('description', '')}\n"
        
        # Call OpenAI for chat response
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are helping refine epics. Here are the current epics:\n{epics_context}\n\nProvide helpful suggestions for improvement or answer questions about the epics."},
                {"role": "user", "content": message}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        return jsonify({
            "success": True,
            "response": ai_response
        })
        
    except Exception as e:
        logger.error(f"Error in epic chat: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/tabbed-story-chat", methods=["POST"])
def tabbed_story_chat():
    """Handle user story chat interactions."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({"success": False, "error": "Message is required"})
        
        # Get current user stories context
        user_stories = session.get('generated_user_stories', [])
        current_epic = session.get('current_epic', {})
        
        if not user_stories:
            return jsonify({"success": False, "error": "No user stories available for refinement"})
        
        # Create context for the chat
        stories_context = f"Current Epic: {current_epic.get('title', '')}\n"
        stories_context += f"Epic Description: {current_epic.get('description', '')}\n\n"
        stories_context += "Current User Stories:\n"
        
        for i, story in enumerate(user_stories, 1):
            stories_context += f"{i}. {story.get('title', '')}\n   {story.get('description', '')}\n"
        
        # Call OpenAI for chat response
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are helping refine user stories. Here is the context:\n{stories_context}\n\nProvide helpful suggestions for improvement or answer questions about the user stories."},
                {"role": "user", "content": message}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        return jsonify({
            "success": True,
            "response": ai_response
        })
        
    except Exception as e:
        logger.error(f"Error in story chat: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/tabbed-details-chat", methods=["POST"])
def tabbed_details_chat():
    """Handle story details chat interactions."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({"success": False, "error": "Message is required"})
        
        # Get current story details context
        current_story = session.get('current_user_story', {})
        story_details = session.get('story_details', {})
        
        if not current_story:
            return jsonify({"success": False, "error": "No story selected for refinement"})
        
        # Create context for the chat
        details_context = f"Current User Story: {current_story.get('title', '')}\n"
        details_context += f"Description: {current_story.get('description', '')}\n"
        details_context += f"Acceptance Criteria: {current_story.get('acceptance_criteria', '')}\n"
        details_context += f"Priority: {current_story.get('priority', '')}\n"
        details_context += f"Estimated Effort: {current_story.get('estimated_effort', '')}\n"
        
        # Call OpenAI for chat response
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are helping refine story details. Here is the current story:\n{details_context}\n\nProvide helpful suggestions for improvement or answer questions about the story details."},
                {"role": "user", "content": message}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        return jsonify({
            "success": True,
            "response": ai_response
        })
        
    except Exception as e:
        logger.error(f"Error in details chat: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/tabbed-submit-jira", methods=["POST"])
def tabbed_submit_jira():
    """Submit the selected story to Jira."""
    try:
        logger.info("POST request to /tabbed-submit-jira")
        
        current_epic = session.get('current_epic', {})
        current_story = session.get('current_user_story', {})
        story_details = session.get('story_details', {})
        
        if not current_story:
            return jsonify({"success": False, "error": "No story selected for submission"})
        
        # Prepare submission data
        submission_data = {
            'epic_title': current_epic.get('title', ''),
            'epic_description': current_epic.get('description', ''),
            'story_title': current_story.get('title', ''),
            'story_description': current_story.get('description', ''),
            'acceptance_criteria': current_story.get('acceptance_criteria', ''),
            'priority': current_story.get('priority', 'Medium'),
            'estimated_effort': current_story.get('estimated_effort', 'Medium'),
            'responsible_systems': story_details.get('responsible_systems', 'TBD'),
            'tagged_requirements': story_details.get('tagged_requirements', 'TBD')
        }
        
        # For demo purposes, just return success
        # In a real implementation, this would integrate with Jira API
        logger.info(f"Mock Jira submission: {submission_data}")
        
        return jsonify({
            "success": True,
            "message": "Story submitted to Jira successfully!",
            "jira_ticket": "DEMO-123"  # Mock ticket number
        })
        
    except Exception as e:
        logger.error(f"Error submitting to Jira: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/tabbed-upload-prd", methods=["POST"])
def tabbed_upload_prd():
    """Handle PRD upload and generate epics for tabbed layout (backward compatibility)."""
    try:
        logger.info("POST request to /tabbed-upload-prd (deprecated - use /tabbed-upload-files)")
        
        # Handle file uploads
        prd_file = request.files.get('prd_file')
        additional_file = request.files.get('additional_file')
        context_notes = request.form.get('context_notes', '')
        
        if not prd_file:
            return jsonify({"success": False, "error": "PRD file is required"})
        
        # Process PRD content
        prd_content = safe_read(prd_file)
        additional_content = ""
        
        if additional_file and additional_file.filename:
            additional_content = safe_read(additional_file)
        
        if not prd_content:
            return jsonify({"success": False, "error": "Could not extract content from PRD file"})
        
        # Store in session
        session['prd_content'] = prd_content
        session['additional_content'] = additional_content
        session['context_notes'] = context_notes
        
        # Create combined content
        combined_content = prd_content
        if additional_content:
            combined_content += f"\n\n--- Additional Documentation ---\n{additional_content}"
        if context_notes:
            combined_content += f"\n\n--- User Context ---\n{context_notes}"
        
        session['combined_content'] = combined_content
        
        # Generate epics using OpenAI
        enhanced_context = create_epic_generation_context(prd_content, additional_content, context_notes)
        
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert product manager and business analyst. Generate comprehensive epics that break down requirements into manageable chunks."},
                    {"role": "user", "content": enhanced_context}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            epics_text = response.choices[0].message.content
            epics = parse_epics_from_response(epics_text)
            
            if epics:
                session['generated_epics'] = epics
                logger.info(f"Generated {len(epics)} epics successfully")
                return jsonify({
                    "success": True,
                    "epics": epics,
                    "message": f"Successfully generated {len(epics)} epics"
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to parse epics from OpenAI response"
                })
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Failed to generate epics. Please check your OpenAI API key."
            })
            
    except Exception as e:
        logger.error(f"Error in PRD upload: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route("/clear-session", methods=["POST"])
def clear_session():
    """Clear all session data."""
    try:
        logger.info("POST request to /clear-session")
        session.clear()
        session.modified = True
        return jsonify({"success": True, "message": "Session cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))  # Different port to avoid conflicts
    debug = os.environ.get("FLASK_ENV", "development").lower() != "production"
    host = "0.0.0.0" if os.environ.get("FLASK_ENV") == "production" else "127.0.0.1"
    
    logger.info(f"Starting Tabbed Backend on port {port}")
    print(f"🚀 Tabbed Layout Backend starting...")
    print(f"   URL: http://localhost:{port}/tabbed-layout")
    print(f"   Health: http://localhost:{port}/health")
    print(f"   Debug: http://localhost:{port}/debug-info")
    
    app.run(host=host, port=port, debug=debug)
