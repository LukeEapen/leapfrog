
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
import csv  # Add CSV support for system mapping
import io  # Add io for StringIO support
import time  # Add time module for performance tracking
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
from openai import OpenAI
from werkzeug.utils import secure_filename
import docx
import PyPDF2
import io

# Additional imports for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    logging.warning("tiktoken not available - using fallback token counting")
    TIKTOKEN_AVAILABLE = False

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

# Token counting function
def count_tokens(text, model="gpt-4"):
    """Count tokens in text using tiktoken (with fallback)."""
    if not TIKTOKEN_AVAILABLE:
        # Fallback estimation: roughly 4 characters per token
        return len(text) // 4
    
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback estimation
        return len(text) // 4

def ask_assistant_from_file_optimized(code_filepath, user_prompt):
    """Optimized assistant interaction using direct chat completion."""
    start_time = time.time()
    logger.info(f"Starting optimized assistant interaction: {code_filepath}")
    
    try:
        # Read agent instructions from file
        agent_file_path = os.path.join("agents", code_filepath)
        
        with open(agent_file_path, 'r', encoding='utf-8') as file:
            system_instructions = file.read()
        
        logger.info(f"Loaded system instructions from {agent_file_path}")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Create chat completion
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4000,
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content.strip()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Assistant response completed in {elapsed_time:.2f} seconds")
        
        # Log token usage
        prompt_tokens = count_tokens(system_instructions + user_prompt, "gpt-4o")
        response_tokens = count_tokens(response_text, "gpt-4o")
        logger.info(f"Token usage - Prompt: {prompt_tokens:,}, Response: {response_tokens:,}, Total: {prompt_tokens + response_tokens:,}")
        
        return response_text
        
    except FileNotFoundError:
        logger.error(f"Agent file not found: {agent_file_path}")
        return f"Error: Agent configuration file '{code_filepath}' not found."
    except Exception as e:
        logger.error(f"Error in assistant interaction: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Error: {str(e)}"


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


    
@app.route("/tabbed-apply-story-details", methods=["POST"])
def tabbed_apply_story_details():
    """Persist updated story details from the chat panel."""
    try:
        data = request.get_json()
        story_id = data.get('story_id')
        updated_details = data.get('story_details')
        if not story_id or not updated_details:
            return jsonify({"success": False, "error": "Story ID and details required"}), 400

        # Get current epic context
        current_epic = session.get('current_epic', {})
        epic_id = current_epic.get('id')

        # Update only the story matching story_id and current epic
        user_stories = session.get('generated_user_stories', [])
        story_updated = False
        for story in user_stories:
            if story.get('id') == story_id and story.get('epic_id') == epic_id:
                story.update(updated_details)
                story_updated = True

        # Persist updated story details for agent and next steps
        session['generated_user_stories'] = user_stories
        session['story_details'] = updated_details
        session.modified = True

        # Optionally, update agent response cache or context if needed
        # (e.g., session['agent_response'] = ...)

        if story_updated:
            return jsonify({
                "success": True,
                "message": "Story details updated and persisted for selected epic.",
                "updated_story": updated_details,
                "epic_id": epic_id
            })
        else:
            return jsonify({
                "success": False,
                "error": "No matching story found for selected epic.",
                "epic_id": epic_id
            }), 404
    except Exception as e:
        logger.error(f"Error persisting story details: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

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
        
        # Store system mapping in the session for system mapping functionality
        if system_mapping_content:
            session['system_info'] = system_mapping_content
            session['system_info_filename'] = system_mapping_file.filename
            session['system_info_uploaded'] = datetime.now().isoformat()
            logger.info(f"System mapping stored in session: {system_mapping_file.filename}")
        
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
        
        # Get system information from session (CSV mapping data)
        system_info = session.get('system_info', '')
        
        prompt = f"""
        Based on the following PRD content and the specific epic, generate detailed user stories.

        PRD Content:
        {prd_content[:3000] if prd_content else "No additional PRD content available"}

        Epic Details:
        Title: {epic.get('title', '')}
        Description: {epic.get('description', '')}

        SYSTEM MAPPING INFORMATION:
        {system_info if system_info else 'No system mapping information provided. User stories will not include specific system mapping.'}

        Please generate 5-8 user stories that specifically support this epic. For each user story, provide:

        User Story 1: [Story Title]
        Description: As a [user type], I want [goal] so that [benefit]
        Acceptance Criteria:
        - [Specific, testable criterion 1]
        - [Specific, testable criterion 2]
        - [Specific, testable criterion 3]
        Priority: High/Medium/Low
        Effort: Small/Medium/Large
        Responsible Systems: [Based on the system mapping information provided above, map this user story to the appropriate systems. If system mapping is available, reference specific systems from the CSV data. If no mapping is available, provide logical system names.]

        User Story 2: [Story Title]
        Description: As a [user type], I want [goal] so that [benefit]
        Acceptance Criteria:
        - [Specific, testable criterion 1]
        - [Specific, testable criterion 2]
        - [Specific, testable criterion 3]
        Priority: High/Medium/Low
        Effort: Small/Medium/Large
        Responsible Systems: [Based on the system mapping information provided above, map this user story to the appropriate systems. If system mapping is available, reference specific systems from the CSV data. If no mapping is available, provide logical system names.]

        Continue this format for all user stories (aim for 5-8 stories).
        
        IMPORTANT: If system mapping information is provided above, use it to map each user story to the most appropriate systems based on the functionality described in the user story. Reference the actual system names from the CSV data when possible.
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert product manager and business analyst. Generate detailed, actionable user stories that align with the given epic and PRD requirements. When system mapping information is provided, ensure each user story is mapped to the most appropriate systems based on the functionality described. Use the CSV system mapping data to reference actual system names."},
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
            responsible_systems = "TBD"
            
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
                elif line.startswith('Responsible Systems:'):
                    responsible_systems_raw = line.replace('Responsible Systems:', '').strip()
                    responsible_systems = extract_primary_system(responsible_systems_raw)
                    current_section = None
                elif line.startswith('- ') and current_section == 'criteria':
                    acceptance_criteria.append(line[2:])
                elif current_section == 'description' and line and not line.startswith('Acceptance') and not line.startswith('Priority') and not line.startswith('Effort') and not line.startswith('Responsible'):
                    if description:
                        description += " " + line
                    else:
                        description = line
            
            if title_part and description:
                # Map responsible systems from CSV if available, otherwise use LLM output
                csv_mapped_systems = map_responsible_systems_from_csv(description, title_part)
                final_responsible_systems = csv_mapped_systems if csv_mapped_systems != "TBD" else responsible_systems
                
                # Ensure only one primary system is used
                primary_system = extract_primary_system(final_responsible_systems)
                
                story = {
                    'id': f'story_{i}',
                    'title': title_part,
                    'description': description,
                    'acceptance_criteria': '\n'.join([f"• {criteria}" for criteria in acceptance_criteria]) if acceptance_criteria else "Acceptance criteria to be defined",
                    'priority': priority,
                    'estimated_effort': effort,
                    'responsible_systems': primary_system,
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
        story_obj = data.get('story')

        user_stories = session.get('generated_user_stories', [])
        selected_story = None

        # Accept either story object or story_id
        if story_obj:
            selected_story = story_obj
            # If no id, try to find by title
            if not selected_story.get('id') and selected_story.get('title'):
                for story in user_stories:
                    if story.get('title') == selected_story.get('title'):
                        selected_story = story
                        break
        elif story_id:
            for story in user_stories:
                if story.get('id') == story_id:
                    selected_story = story
                    break
        else:
            return jsonify({"success": False, "error": "Story ID or story object is required"})

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
        current_epic = session.get('current_epic', {})
        
        # Extract story information
        story_name = user_story.get('title', '')
        story_description = user_story.get('description', '')
        
        # Generate acceptance criteria if not already present
        acceptance_criteria_text = user_story.get('acceptance_criteria', '')
        if not acceptance_criteria_text or acceptance_criteria_text == "Acceptance criteria to be defined":
            # Generate acceptance criteria using the same logic as three-section
            criteria_prompt = f"""Generate detailed acceptance criteria for this user story:

Story: {story_name}
Description: {story_description}

Return 3-5 specific, testable acceptance criteria in this format:
Given [context], when [action], then [expected result]

Focus on the core functionality and edge cases."""
            
            criteria_response = ask_assistant_from_file_optimized("poc2_agent4_acceptanceCriteria_gen", criteria_prompt)
            
            # Parse response into individual criteria
            criteria_lines = criteria_response.strip().split('\n')
            acceptance_criteria = []
            for line in criteria_lines:
                line = line.strip()
                if line and (line.startswith('Given') or line.startswith('•') or line.startswith('-')):
                    # Clean up formatting
                    if line.startswith('•') or line.startswith('-'):
                        line = line[1:].strip()
                    acceptance_criteria.append(line)
            
            if not acceptance_criteria:
                # Fallback criteria
                acceptance_criteria = [
                    f"Given a user accesses {story_name.lower()}, when they perform the required actions, then the system should respond appropriately",
                    f"Given invalid input is provided, when {story_name.lower()} is attempted, then appropriate error messages should be displayed",
                    f"Given a user completes {story_name.lower()}, when the operation is successful, then the system should provide confirmation",
                    f"Given a compliance rule applies, when {story_name.lower()} is processed, then all regulatory requirements must be met",
                    f"Given a data integrity constraint, when {story_name.lower()} is updated, then the system must validate and preserve data integrity",
                    f"Given a system error occurs, when {story_name.lower()} is executed, then the error must be logged and handled gracefully",
                    f"Given an edge case scenario, when {story_name.lower()} is triggered, then the system must behave as expected"
                ]
        
        # Extract tagged requirements from session or generate based on story
        tagged_requirements = session.get('tagged_requirements', [])
        if not tagged_requirements or not isinstance(tagged_requirements, list):
            tagged_requirements = [
                "REQ-001: User authentication and authorization",
                "REQ-002: Data security and privacy",
                "REQ-003: User interface and experience"
            ]
        
        # Generate traceability matrix
        traceability_prompt = f"""Create a traceability matrix for the following user story:

Story: {story_name}
Description: {story_description}
Acceptance Criteria: {acceptance_criteria_text}

Create a formatted traceability matrix showing relationships between requirements, story, and test cases."""
        
        traceability_response = ask_assistant_from_file_optimized("poc2_traceability_agent", traceability_prompt)
        
        story_details = {
            'title': story_name,
            'description': story_description,
            'acceptance_criteria': acceptance_criteria_text,
            'priority': user_story.get('priority', 'Medium'),
            'estimated_effort': user_story.get('estimated_effort', 'Medium'),
            'epic_title': current_epic.get('title', 'Unknown Epic'),
            'epic_description': current_epic.get('description', ''),
            'responsible_systems': extract_primary_system(user_story.get('responsible_systems', 'To be determined')),
            'tagged_requirements': tagged_requirements,
            'traceability_matrix': traceability_response
        }
        
        return story_details
        
    except Exception as e:
        logger.error(f"Error generating story details: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return fallback story details
        return {
            'title': user_story.get('title', ''),
            'description': user_story.get('description', ''),
            'acceptance_criteria': user_story.get('acceptance_criteria', 'Acceptance criteria to be defined'),
            'priority': user_story.get('priority', 'Medium'),
            'estimated_effort': user_story.get('estimated_effort', 'Medium'),
            'epic_title': session.get('current_epic', {}).get('title', 'Unknown Epic'),
            'epic_description': session.get('current_epic', {}).get('description', ''),
            'responsible_systems': extract_primary_system(user_story.get('responsible_systems', 'To be determined')),
            'tagged_requirements': ['REQ-001: Basic functionality requirement'],
            'traceability_matrix': 'Traceability matrix generation failed. Please try again.'
        }

@app.route("/tabbed-user-story-details-page", methods=["GET", "POST"])
def tabbed_user_story_details_page():
    """Render the user story details page for Jira submission from tabbed layout."""
    logger.info("Rendering user story details page for tabbed layout")
    
    try:
        # Get story details from session or form
        if request.method == "POST":
            story_id = request.form.get('selected_story_id')
            story_name = request.form.get('selected_story_name')
            story_description = request.form.get('selected_story_description')
            
            logger.info(f"POST data - ID: {story_id}, Name: {story_name}, Description: [{story_description}]")
        else:
            # GET request - use session data
            current_story = session.get('current_user_story', {})
            story_details = session.get('story_details', {})
            
            if not current_story:
                logger.warning("No current story in session for tabbed details page")
                return redirect(url_for('tabbed_layout'))
            
            story_id = current_story.get('id', '')
            story_name = current_story.get('title', '')
            story_description = current_story.get('description', '')
        
        if not story_name:
            logger.error("No story name provided for tabbed details page")
            return redirect(url_for('tabbed_layout'))
        
        # Get current epic context
        current_epic = session.get('current_epic', {})
        
        # Generate story details if needed
        if request.method == "POST" or not session.get('story_details'):
            # Find the story in our user stories
            user_stories = session.get('generated_user_stories', [])
            selected_story = None
            
            for story in user_stories:
                if story.get('id') == story_id or story.get('title') == story_name:
                    selected_story = story
                    break
            
            if not selected_story:
                # Create a basic story object
                selected_story = {
                    'id': story_id,
                    'title': story_name,
                    'description': story_description,
                    'priority': 'High',
                    'estimated_effort': 'Medium'
                }
            
            # Generate enhanced story details
            story_details = generate_story_details(selected_story)
            
            # Store in session
            session['current_user_story'] = selected_story
            session['story_details'] = story_details
        else:
            # Use existing story details from session
            story_details = session.get('story_details', {})
        
        # Extract data for template
        acceptance_criteria = story_details.get('acceptance_criteria', '')
        if isinstance(acceptance_criteria, list):
            acceptance_criteria = acceptance_criteria
        elif isinstance(acceptance_criteria, str):
            # Convert string to list if needed
            acceptance_criteria = [criteria.strip() for criteria in acceptance_criteria.split('\n') if criteria.strip()]
        
        tagged_requirements = story_details.get('tagged_requirements', [])
        if isinstance(tagged_requirements, str):
            tagged_requirements = [tagged_requirements]
        
        traceability_matrix = story_details.get('traceability_matrix', 'Not available')
        
        # Prepare template context
        template_context = {
            'epic_title': current_epic.get('title', ''),
            'epic_description': current_epic.get('description', ''),
            'user_story_name': story_name,
            'user_story_description': story_description,
            'acceptance_criteria': acceptance_criteria,
            'priority': story_details.get('priority', 'High'),
            'responsible_systems': extract_primary_system(story_details.get('responsible_systems', 'CAPS')),
            'tagged_requirements': tagged_requirements,
            'traceability_matrix': traceability_matrix
        }
        
        logger.info(f"Rendering user story details template with context: {template_context}")
        
        # Render the user story details template
        return render_template('poc2_user_story_details.html', **template_context)
        
    except Exception as e:
        logger.error(f"Error rendering user story details page: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Error loading user story details: {str(e)}", 500

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

@app.route("/chat_traceability", methods=["POST"])
def chat_traceability():
    """Handle chat requests for traceability matrix refinement and enhancement."""
    logger.info("Request received for traceability matrix chat")
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
            
        user_message = data.get("userMessage", "").strip()
        context = data.get("sessionContext", "{}")
        
        if not user_message:
            return jsonify({"success": False, "error": "Message is required"}), 400
        
        # Extract context information
        user_story_title = data.get("userStory", {}).get("title", "")
        user_story_description = data.get("userStory", {}).get("description", "")
        acceptance_criteria = data.get("userStory", {}).get("acceptanceCriteria", "")
        current_traceability = data.get("currentTraceability", "")
        
        logger.info(f"Traceability chat message: {user_message}")
        logger.info(f"Context - Story: {user_story_title}")
        
        # Build context string for AI
        context_text = f"""
Current User Story Context:
- Story Title: {user_story_title or 'Not specified'}
- Description: {user_story_description or 'Not specified'}
- Acceptance Criteria: {acceptance_criteria or 'Not specified'}
- Current Traceability Matrix: {current_traceability or 'Not available'}
"""
        
        # System prompt for traceability matrix chat
        system_prompt = f"""You are an expert systems analyst and requirements traceability specialist. You help create, analyze, and improve traceability matrices that map user stories to PRD requirements, design documents, and testing artifacts.

{context_text}

Your role is to:
1. Analyze and improve traceability matrices between user stories and PRD requirements
2. Identify missing traceability links and suggest improvements
3. Help create bidirectional traceability relationships
4. Ensure comprehensive coverage of requirements traceability
5. Suggest impact analysis approaches for requirement changes
6. Help with compliance and audit traceability documentation
7. Create structured, readable traceability matrices in table format

When enhancing traceability matrices, you should:
- Create clear mappings between user stories and specific PRD requirements
- Include requirement IDs, descriptions, and traceability relationships
- Suggest forward and backward traceability links
- Include test case traceability where relevant
- Format output as readable tables or structured text
- Identify gaps in requirement coverage
- Suggest additional traceability dimensions (design documents, test cases, compliance standards)

For traceability matrix outputs, use this format:
| User Story | PRD Requirement ID | Requirement Description | Traceability Type | Test Coverage |
|------------|-------------------|------------------------|-------------------|---------------|
| [Story ID] | [REQ-ID] | [Description] | [Forward/Backward/Bidirectional] | [Test Case IDs] |

Always provide structured, professional traceability information that would be suitable for project documentation and compliance audits."""

        user_prompt = f"User Request: {user_message}\n\nContext: Please analyze and enhance the traceability matrix for the current user story."

        # Use OpenAI to generate traceability guidance
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            logger.info("Traceability chat response generated successfully")
            return jsonify({
                "success": True,
                "message": ai_response,
                "userMessage": user_message
            })
            
        except Exception as openai_error:
            logger.error(f"OpenAI API error in traceability chat: {str(openai_error)}")
            # Return fallback response
            fallback_response = generate_traceability_fallback_response(user_message, user_story_title, current_traceability)
            return jsonify({
                "success": True,
                "message": fallback_response,
                "userMessage": user_message,
                "note": "AI service unavailable - using fallback response"
            })
        
    except Exception as e:
        logger.error(f"Error in traceability chat: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

def generate_traceability_fallback_response(user_message, user_story_title, current_traceability):
    """Generate a fallback response for traceability chat when OpenAI is unavailable."""
    lower_message = user_message.lower()
    
    if any(word in lower_message for word in ['improve', 'enhance', 'update', 'add']):
        return f"""📊 **Traceability Matrix Enhancement Suggestions**

For the user story "{user_story_title or 'Current Story'}", consider these improvements:

**📋 Requirement Mapping:**
• Map to specific PRD sections (e.g., REQ-AUTH-001, REQ-DATA-002)
• Include functional and non-functional requirements
• Add compliance requirements if applicable

**🔄 Bidirectional Traceability:**
• Forward: Requirements → User Stories → Test Cases
• Backward: Test Cases → User Stories → Requirements
• Impact analysis for requirement changes

**✅ Coverage Analysis:**
• Ensure no orphaned requirements exist
• Validate test coverage completeness

Would you like me to help create a specific traceability matrix table for this user story?"""
    
    elif any(word in lower_message for word in ['table', 'format', 'matrix']):
        return f"""📊 **Traceability Matrix Table Format**

Here's a recommended structure for your traceability matrix:

| User Story | PRD Requirement | Requirement Description | Traceability Type | Test Case |
|------------|----------------|------------------------|-------------------|-----------|
| {user_story_title or 'US-001'} | REQ-001 | User authentication and authorization | Forward | TC-001 |
| {user_story_title or 'US-001'} | REQ-002 | Data validation and input sanitization | Forward | TC-002 |
| {user_story_title or 'US-001'} | REQ-003 | Error handling and user feedback | Forward | TC-003 |

**Legend:**
- **Forward**: Requirement leads to User Story
- **Backward**: User Story validates Requirement  
- **Bidirectional**: Two-way relationship"""
    
    else:
        return f"""🔗 **Traceability Matrix Guidance**

I can help you improve the traceability matrix for "{user_story_title or 'your user story'}". Here are some common tasks:

**📝 Available Actions:**
• Create a new traceability matrix table
• Map requirements to specific PRD sections
• Add test case traceability links
• Enhance requirement descriptions
• Add compliance traceability
• Perform coverage gap analysis

**💡 Quick Tips:**
• Use clear requirement IDs (e.g., REQ-AUTH-001)
• Include both functional and non-functional requirements
• Document the traceability relationship type
• Link to specific test cases where possible

What specific aspect of the traceability matrix would you like to work on?"""

@app.route("/chat_requirements", methods=["POST"])
def chat_requirements():
    """Handle chat requests for tagged requirements refinement."""
    logger.info("Request received for requirements chat")
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
            
        user_message = data.get("userMessage", "").strip()
        
        if not user_message:
            return jsonify({"success": False, "error": "Message is required"}), 400
        
        # Extract context information
        user_story_title = data.get("userStory", {}).get("title", "")
        user_story_description = data.get("userStory", {}).get("description", "")
        current_requirements = data.get("currentRequirements", [])
        
        logger.info(f"Requirements chat message: {user_message}")
        
        # Build context for AI
        context_text = f"""
Current User Story:
- Title: {user_story_title or 'Not specified'}
- Description: {user_story_description or 'Not specified'}
- Current Tagged Requirements: {current_requirements if current_requirements else 'None specified'}
"""
        
        system_prompt = f"""You are an expert business analyst specializing in requirements engineering. You help identify, refine, and tag requirements for user stories.

{context_text}

Your role is to:
1. Analyze user stories and identify relevant functional and non-functional requirements
2. Tag requirements with clear, traceable identifiers
3. Ensure requirements are specific, measurable, and testable
4. Help categorize requirements (functional, security, performance, usability, etc.)
5. Suggest additional requirements that might be missing
6. Help improve requirement clarity and completeness

When providing requirements, format them as:
- REQ-[CATEGORY]-[NUMBER]: [Clear requirement description]

Categories can include: FUNC (Functional), SEC (Security), PERF (Performance), UI (User Interface), DATA (Data), etc.

Always provide actionable, specific requirements that directly relate to the user story."""

        user_prompt = f"User Request: {user_message}\n\nPlease help improve the tagged requirements for this user story."

        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            logger.info("Requirements chat response generated successfully")
            return jsonify({
                "success": True,
                "message": ai_response,
                "userMessage": user_message
            })
            
        except Exception as openai_error:
            logger.error(f"OpenAI API error in requirements chat: {str(openai_error)}")
            # Return fallback response
            fallback_response = f"""📋 **Requirements Analysis Guidance**

For the user story "{user_story_title or 'Current Story'}", consider these requirement categories:

**🔧 Functional Requirements:**
• REQ-FUNC-001: Core functionality specification
• REQ-FUNC-002: Business rule implementation
• REQ-FUNC-003: Data processing requirements

**🔒 Security Requirements:**
• REQ-SEC-001: Authentication and authorization
• REQ-SEC-002: Data protection and privacy
• REQ-SEC-003: Input validation and sanitization

**⚡ Performance Requirements:**
• REQ-PERF-001: Response time specifications
• REQ-PERF-002: Scalability requirements
• REQ-PERF-003: Resource utilization limits

**🎨 User Interface Requirements:**
• REQ-UI-001: User experience design
• REQ-UI-002: Accessibility compliance
• REQ-UI-003: Mobile responsiveness

Would you like me to help develop specific requirements for any of these categories?"""
            
            return jsonify({
                "success": True,
                "message": fallback_response,
                "userMessage": user_message,
                "note": "AI service unavailable - using fallback response"
            })
        
    except Exception as e:
        logger.error(f"Error in requirements chat: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

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
        
        # Check if this is an AJAX request
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        
        # Get story data
        epic_title = current_epic.get('title', '')
        user_story_name = current_story.get('title', current_story.get('name', ''))
        user_story_description = current_story.get('description', current_story.get('summary', ''))
        priority = current_epic.get('priority', current_story.get('priority', 'High'))
        responsible_systems = extract_primary_system(story_details.get('responsible_systems', 'CAPS'))
        
        # Extract acceptance criteria and tagged requirements
        acceptance_criteria = []
        tagged_requirements = []
        traceability_matrix = ''
        
        if story_details:
            # Get acceptance criteria
            if 'acceptance_criteria' in story_details and isinstance(story_details['acceptance_criteria'], list):
                acceptance_criteria = story_details['acceptance_criteria']
            elif 'acceptance_criteria' in current_story:
                if isinstance(current_story['acceptance_criteria'], list):
                    acceptance_criteria = current_story['acceptance_criteria']
                elif isinstance(current_story['acceptance_criteria'], str):
                    acceptance_criteria = [current_story['acceptance_criteria']]
            
            # Get tagged requirements
            if 'tagged_requirements' in story_details and isinstance(story_details['tagged_requirements'], list):
                tagged_requirements = story_details['tagged_requirements']
            
            # Get traceability matrix
            traceability_matrix = story_details.get('traceability_matrix', '')
        
        logger.info(f"Submitting Jira ticket for story: {user_story_name}")
        logger.info(f"Epic: {epic_title}")
        logger.info(f"Priority: {priority}")
        logger.info(f"Systems: {responsible_systems}")
        
        # Format the description for Jira
        jira_description = user_story_description
        
        # Add acceptance criteria to description
        if acceptance_criteria:
            jira_description += "\n\n*Acceptance Criteria:*\n"
            for i, criterion in enumerate(acceptance_criteria, 1):
                jira_description += f"• {criterion}\n"
        
        # Add tagged requirements to description
        if tagged_requirements:
            jira_description += "\n\n*Tagged Requirements:*\n"
            for i, requirement in enumerate(tagged_requirements, 1):
                jira_description += f"• {requirement}\n"
        
        # Add traceability matrix to description
        if traceability_matrix and traceability_matrix.strip() != 'Traceability mapping not available.':
            jira_description += "\n\n*Traceability Matrix (User Story → PRD Requirements):*\n"
            jira_description += f"{traceability_matrix}\n"
        
        # Add system information
        if responsible_systems:
            jira_description += f"\n\n*Responsible Systems:* {responsible_systems}"
        
        # Try to create Jira ticket using the Jira connector
        try:
            # Import and use Jira connector
            from jira import JIRA
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            
            # Jira configuration
            JIRA_SERVER = 'https://lalluluke.atlassian.net/'
            EMAIL = 'lalluluke@gmail.com'
            API_TOKEN = os.getenv("JIRA_API_TOKEN")
            
            if not API_TOKEN:
                logger.error("JIRA_API_TOKEN not found in environment variables")
                if is_ajax:
                    return jsonify({
                        "success": False,
                        "error": "Jira API token not configured. Please contact administrator."
                    })
                else:
                    return render_template('poc2_user_story_details.html', 
                                         error_message="Jira API token not configured. Please contact administrator.")
            
            # Connect to Jira
            jira = JIRA(server=JIRA_SERVER, basic_auth=(EMAIL, API_TOKEN))
            
            # Create issue dictionary (removed priority field due to Jira screen configuration)
            issue_dict = {
                'project': {'key': 'SCRUM'},  # Default project key
                'summary': user_story_name,
                'description': jira_description,
                'issuetype': {'name': 'Story'}
            }
            
            # Create the Jira issue
            new_issue = jira.create_issue(fields=issue_dict)
            
            logger.info(f"Successfully created Jira ticket: {new_issue.key}")
            
            # Store the ticket information for display
            success_data = {
                'success': True,
                'ticket_id': new_issue.key,
                'jira_url': f"{JIRA_SERVER}browse/{new_issue.key}",
                'message': f"Jira ticket {new_issue.key} created successfully!",
                'epic_title': epic_title,
                'user_story_name': user_story_name
            }
            
            if is_ajax:
                return jsonify(success_data)
            else:
                # Return success page with ticket details
                return render_template('jira_success.html', **success_data)
            
        except ImportError as e:
            logger.error(f"Jira library not available: {e}")
            error_msg = "Jira integration not available. Please install the required dependencies."
            if is_ajax:
                return jsonify({"success": False, "error": error_msg})
            else:
                return render_template('poc2_user_story_details.html', error_message=error_msg)
            
        except Exception as e:
            logger.error(f"Error creating Jira ticket: {e}")
            error_msg = f"Failed to create Jira ticket: {str(e)}"
            if is_ajax:
                return jsonify({"success": False, "error": error_msg})
            else:
                return render_template('poc2_user_story_details.html', error_message=error_msg)
    
    except Exception as e:
        logger.error(f"Error in Jira ticket submission: {str(e)}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"success": False, "error": f"Error submitting to Jira: {str(e)}"})
        else:
            return render_template('poc2_user_story_details.html', 
                                 error_message=f"Error submitting to Jira: {str(e)}")

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

# System Mapping Upload and Retrieval
@app.route("/upload-system-info", methods=["POST"])
def upload_system_info():
    """Upload and process system information file for user story mapping."""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        # Check file size (limit to 10MB)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            return jsonify({"success": False, "error": "File size too large (max 10MB)"}), 400
        
        # Read and process file content
        content = ""
        filename = file.filename.lower()
        
        try:
            if filename.endswith('.txt'):
                content = file.read().decode('utf-8')
            elif filename.endswith('.json'):
                json_data = json.load(file)
                content = json.dumps(json_data, indent=2)
            elif filename.endswith('.csv'):
                content = file.read().decode('utf-8')
            else:
                # Try to read as text for other formats
                content = file.read().decode('utf-8')
        except UnicodeDecodeError:
            return jsonify({"success": False, "error": "File encoding not supported. Please use UTF-8 encoded files."}), 400
        except json.JSONDecodeError:
            return jsonify({"success": False, "error": "Invalid JSON file format"}), 400
        
        if not content.strip():
            return jsonify({"success": False, "error": "File appears to be empty"}), 400
        
        # Store system information in session
        session['system_info'] = content
        session['system_info_filename'] = file.filename
        session['system_info_uploaded'] = datetime.now().isoformat()
        session.modified = True
        
        # Parse basic system information for preview
        lines = content.split('\n')[:10]  # First 10 lines for preview
        preview = '\n'.join(lines)
        if len(content.split('\n')) > 10:
            preview += "\n... (file continues)"
        
        logger.info(f"System information uploaded: {file.filename} ({file_size} bytes)")
        
        return jsonify({
            "success": True,
            "message": f"System information uploaded successfully: {file.filename}",
            "filename": file.filename,
            "size": file_size,
            "preview": preview[:500]  # Limit preview to 500 chars
        })
        
    except Exception as e:
        logger.error(f"Error uploading system info: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": "Failed to upload system information"}), 500

@app.route("/get-system-info", methods=["GET"])
def get_system_info():
    """Get current system information status."""
    try:
        system_info = session.get('system_info', '')
        filename = session.get('system_info_filename', '')
        uploaded_time = session.get('system_info_uploaded', '')
        
        return jsonify({
            "success": True,
            "has_system_info": bool(system_info),
            "filename": filename,
            "uploaded_time": uploaded_time,
            "preview": system_info[:500] if system_info else ""
        })
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@app.route("/clear-system-info", methods=["POST"])
def clear_system_info():
    """Clear uploaded system information."""
    try:
        if 'system_info' in session:
            del session['system_info']
        if 'system_info_filename' in session:
            del session['system_info_filename']
        if 'system_info_uploaded' in session:
            del session['system_info_uploaded']
        session.modified = True
        
        logger.info("System information cleared from session")
        return jsonify({"success": True, "message": "System information cleared"})
    except Exception as e:
        logger.error(f"Error clearing system info: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

def map_responsible_systems_from_csv(user_story_description, user_story_title):
    """Map responsible systems from uploaded CSV based on user story content. Returns only the most important single system."""
    try:
        system_info = session.get('system_info', '')
        if not system_info:
            return "TBD"
        
        # Parse CSV content
        csv_reader = csv.DictReader(io.StringIO(system_info))
        rows = list(csv_reader)
        
        if not rows:
            return "TBD"
        
        # Get column names (case-insensitive)
        columns = {col.lower().strip(): col for col in rows[0].keys()}
        
        # Look for system/component columns
        system_col = None
        description_col = None
        
        for col_lower, col_original in columns.items():
            if 'system' in col_lower or 'component' in col_lower or 'service' in col_lower:
                system_col = col_original
            if 'description' in col_lower or 'function' in col_lower or 'purpose' in col_lower:
                description_col = col_original
        
        if not system_col:
            # If no specific system column, use first column as system names
            system_col = list(rows[0].keys())[0]
        
        # Search for matching systems based on content with scoring
        system_matches = []
        search_text = f"{user_story_title} {user_story_description}".lower()
        
        for row in rows:
            system_name = row.get(system_col, '').strip()
            if not system_name:
                continue
            
            match_score = 0
            
            # Check if system name appears in user story (highest score)
            if system_name.lower() in search_text:
                match_score += 10
            
            # Check for partial matches with system name
            system_words = system_name.lower().split()
            for word in system_words:
                if len(word) > 2 and word in search_text:
                    match_score += 5
            
            # Check if description/function matches
            if description_col:
                system_desc = row.get(description_col, '').lower()
                if system_desc:
                    desc_words = [word for word in system_desc.split() if len(word) > 3]
                    for keyword in desc_words:
                        if keyword in search_text:
                            match_score += 2
            
            if match_score > 0:
                system_matches.append((system_name, match_score))
        
        if system_matches:
            # Sort by score (descending) and return the highest scoring system
            system_matches.sort(key=lambda x: x[1], reverse=True)
            return system_matches[0][0]  # Return only the best match
        else:
            # If no matches found, return the first system as fallback
            first_system = rows[0].get(system_col, '').strip()
            return first_system if first_system else "TBD"
            
    except Exception as e:
        logger.error(f"Error mapping systems from CSV: {str(e)}")
        return "TBD"
            
    except Exception as e:
        logger.error(f"Error mapping systems from CSV: {str(e)}")
        return "TBD"

def extract_primary_system(responsible_systems_str):
    """Extract the primary/most important system from a comma-separated list of systems."""
    if not responsible_systems_str or responsible_systems_str.strip() in ['TBD', 'To be determined']:
        return 'CAPS'  # Default to CAPS as primary system
    
    # Split by comma and get the first (most important) system
    systems = [s.strip() for s in responsible_systems_str.split(',') if s.strip()]
    
    if not systems:
        return 'CAPS'
    
    # Return the first system as it's typically the most important
    primary_system = systems[0]
    
    # Clean up any formatting issues
    primary_system = primary_system.replace('System:', '').replace('Component:', '').strip()
    
    return primary_system

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
