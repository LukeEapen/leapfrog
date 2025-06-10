########################
"""
new-prd-workflow.py
A Flask-based web application for generating Product Requirements Documents (PRDs) using Gemini (Google Generative AI).
The workflow guides users through a multi-step process, leveraging multiple AI agents to synthesize product overviews,
feature analyses, and detailed requirements, and compiles the results into a downloadable Word document.

Key Features:
-------------
- User authentication and secure session management.
- Multi-step input workflow for capturing industry, sector, geography, intent, and features.
- Asynchronous and parallel invocation of multiple Gemini agents for content generation.
- Caching and retry logic for robust agent interactions.
- Data storage abstraction supporting both Redis and file-based backends.
- Extraction and validation of references from AI-generated outputs.
- Automated cleanup of expired session data.
- Dynamic generation of a styled Word document (docx) with all PRD sections and references.
- Rate limiting and error handling for API endpoints.

Main Components:
----------------
- Flask routes for login, multi-step PRD creation, and document download.
- Agent orchestration functions for calling Gemini agents (sync and async).
- Data storage and retrieval utilities with session key management.
- Markdown-to-Word conversion utilities with custom styling and hyperlink support.
- Reference extraction and validation from AI outputs.
- Input validation using Marshmallow schemas.
- Logging configuration for monitoring and debugging.

Environment Variables:
----------------------
- GEMINI_API_KEY: API key for Gemini.
- REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD: Redis configuration.
- FLASK_SECRET_KEY: Flask session secret.
- ADMIN_USERNAME, ADMIN_PASSWORD: Admin credentials.

Usage:
------
Run the application and navigate to the root URL to begin the PRD creation workflow.
Follow the steps to input product context, review and edit AI-generated content, and download the final PRD document.
"""

# IMPORTS AND SETUP
########################

# Standard library imports
import os
import re
import json
import uuid
import time
import asyncio
import logging
import tempfile
import logging.config
import random
from io import BytesIO
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor

# Security and authentication imports
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Third-party framework and API imports
import google.generativeai as genai
import redis
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, send_file, jsonify
)
from dotenv import load_dotenv
from marshmallow import Schema, fields, ValidationError

# Document processing imports
from bs4 import BeautifulSoup
from markdown2 import markdown
from docx.shared import Pt, RGBColor
from docx import Document
import docx
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def run_chat_agent(prompt_file_path, user_input, temperature=0.2, top_p=1.0, max_tokens=1000):
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    prompt = f"{system_prompt}\n\nUser Input:\n{user_input}"

    logger.info(f"[GEMINI CALL] Calling Gemini API with prompt file '{prompt_file_path}' | user_input length: {len(user_input)}")

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens
        }
    )

    # Log token usage if available
    usage = getattr(response, "usage_metadata", None)
    if usage:
        prompt_tokens = usage.prompt_token_count
        completion_tokens = usage.candidates_token_count
        total_tokens = prompt_tokens + completion_tokens if prompt_tokens and completion_tokens else None
        logger.info(f"[GEMINI USAGE] Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total tokens: {total_tokens}")
        # If you want to estimate cost, you can multiply total_tokens by your rate here
        # logger.info(f"[GEMINI COST ESTIMATE] (Set your rate per 1K tokens here)")
    else:
        logger.info("[GEMINI USAGE] Usage metadata not available in response.")

    return response.text

########################
# REDIS CONFIGURATION
########################

try:
    redis_test = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_DB', 0)),
        password=os.getenv('REDIS_PASSWORD'),
        decode_responses=True
    )
    redis_test.ping()
    redis_client = redis_test
    USING_REDIS = True
except redis.ConnectionError as e:
    logging.warning(f"Redis connection failed: {e}. Using file storage.")
    TEMP_DIR = tempfile.gettempdir()
    USING_REDIS = False

########################
# FLASK APP CONFIGURATION
########################

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')

app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=1)
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

executor = ThreadPoolExecutor(max_workers=10)

# AGENT PROMPT FILES
AGENT_FILES = {
    'agent_1_1': 'agents/agent_1_1',
    'agent_2': 'agents/agent_2',
    'agent_3': 'agents/agent_3',
    'agent_4_1': 'agents/agent_4_1',
    'agent_4_2': 'agents/agent_4_2',
    'agent_4_3': 'agents/agent_4_3',
    'agent_4_4': 'agents/agent_4_4',
    'agent_4_5': 'agents/agent_4_5',
    'agent_4_6': 'agents/agent_4_6'
}

########################
# LOGGING CONFIGURATION
########################

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'app.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

########################
# DATA STORAGE FUNCTIONS
########################

def store_data(data):
    session_id = str(uuid.uuid4())
    if USING_REDIS:
        redis_client.setex(session_id, 3600, json.dumps(data))
    else:
        with open(os.path.join(TEMP_DIR, f'prd_session_{session_id}.json'), 'w') as f:
            json.dump(data, f)
    return session_id

def get_data(session_id):
    if USING_REDIS:
        data = redis_client.get(session_id)
        return json.loads(data) if data else None
    else:
        path = os.path.join(TEMP_DIR, f'prd_session_{session_id}.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    return None

########################
# MONITORING & PERFORMANCE
########################
def monitor_agent_performance(func):
    @wraps(func)
    def wrapper(agent_key, input_text, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(agent_key, input_text, *args, **kwargs)
            duration = time.time() - start_time
            logging.info(f"Agent {agent_key} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"Agent {agent_key} failed after {duration:.2f}s: {str(e)}")
            raise
    return wrapper

def verify_credentials(username, password):
    return (username == os.getenv('ADMIN_USERNAME') and
            password == os.getenv('ADMIN_PASSWORD'))

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if verify_credentials(request.form.get('username'), request.form.get('password')):
            session['logged_in'] = True
            return redirect(url_for('page1'))
        return render_template('page0_login.html', error=True)
    return render_template('page0_login.html', error=False)

@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        inputs = {key: request.form[key] for key in ['industry', 'sector', 'geography', 'intent', 'features']}
        MAX_FILE_SIZE_KB = 100
        if 'context_file' in request.files:
            file = request.files['context_file']
            if file and file.filename:
                file.seek(0, os.SEEK_END)
                file_length = file.tell()
                file.seek(0)
                if file_length > MAX_FILE_SIZE_KB * 1024:
                    logging.warning("Uploaded file too large")
                    return "Uploaded file exceeds the 100KB limit.", 400
                try:
                    filename = file.filename.lower()
                    if filename.endswith('.txt'):
                        file_content = file.read().decode('utf-8')
                    elif filename.endswith('.docx'):
                        doc = docx.Document(file)
                        file_content = "\n".join([para.text for para in doc.paragraphs])
                    else:
                        file_content = "Unsupported file format."
                    inputs['context_file'] = file_content
                except Exception as e:
                    logging.error(f"File upload error: {str(e)}")
                    return "Error processing file upload", 400

        session.update(inputs)

        context_parts = [f"{k.replace('_', ' ').title()}: {v}" for k, v in inputs.items()
                         if k != 'context_file']

        if 'context_file' in inputs:
            context_parts.append(f"\nAdditional Context:\n{inputs['context_file']}")

        context = "\n".join(context_parts)
        session_id = store_data({
            "inputs": inputs,
            "status": "processing"
        })
        session['data_key'] = session_id

        if USING_REDIS:
            redis_client.delete(session_id)
        else:
            path = os.path.join(TEMP_DIR, f'prd_session_{session_id}.json')
            if os.path.exists(path):
                os.remove(path)

        def run_agents():
            try:
                a11 = run_chat_agent(AGENT_FILES['agent_1_1'], context)
                a2 = run_chat_agent(AGENT_FILES['agent_2'], context)
                a3 = run_chat_agent(AGENT_FILES['agent_3'], context)
                logging.info(f"Agent 1.1 Output: {a11}")
                logging.info(f"Agent 2 Output: {a2}")
                final_data = {
                    "product_overview": a11,
                    "feature_overview": a2,
                    "highest_order": a3,
                    "status": "complete"
                }
                if USING_REDIS:
                    redis_client.setex(session_id, 3600, json.dumps(final_data))
                else:
                    with open(os.path.join(TEMP_DIR, f'prd_session_{session_id}.json'), 'w') as f:
                        json.dump(final_data, f)
                logging.info(f"[AGENTS DONE] Page 1 background agents complete for session {session_id}")
            except Exception as e:
                logging.error(f"Error during agent execution: {e}")
                fail_data = {"status": "error", "message": str(e)}
                if USING_REDIS:
                    redis_client.setex(session_id, 3600, json.dumps(fail_data))
                else:
                    with open(os.path.join(TEMP_DIR, f'prd_session_{session_id}.json'), 'w') as f:
                        json.dump(fail_data, f)

        executor.submit(run_agents)

        return redirect('/page2')

    return render_template('page1_input.html')

@app.route('/page2', methods=['GET', 'POST'])
def page2():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    session_id = session.get('data_key', '')
    data = get_data(session_id) or {}

    start_time = time.time()
    timeout = 120

    while time.time() - start_time < timeout:
        current_data = get_data(session_id) or {}
        if current_data.get("status") == "complete":
            data = current_data
            break
        elif current_data.get("status") == "error":
            return f"<h3>❌ Agent processing failed:</h3><pre>{current_data.get('message')}</pre>", 500
        time.sleep(1)
    else:
        return "⏳ Processing took too long. Please refresh the page shortly.", 504

    if request.method == 'POST':
        data.update({
            "product_overview": request.form.get("product_overview", ""),
            "feature_overview": request.form.get("feature_overview", "")
        })
        if USING_REDIS:
            redis_client.setex(session_id, 3600, json.dumps(data))
        return redirect('/page3')

    return render_template("page2_agents.html",
        agent11_output=data.get('product_overview', ''),
        agent2_output=data.get('feature_overview', '')
    )

@app.route('/update_content', methods=['POST'])
def update_content():
    if not session.get('logged_in'):
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        data = request.get_json()
        content_type = data.get('type')
        new_content = data.get('content')

        if not content_type or not new_content:
            return jsonify({'error': 'Missing required fields'}), 400

        stored_data = get_data(session.get('data_key', '')) or {}

        if content_type == 'product':
            existing_content = stored_data.get('product_overview', '')
            full_prompt = f"""You are editing the Product Overview section. Keep existing relevant content unless changes are requested.

            Current content:
            {existing_content}

            User instruction:
            {new_content}
            """
            new_response = run_chat_agent(AGENT_FILES['agent_1_1'], full_prompt)
            stored_data['product_overview'] = new_response
        elif content_type == 'feature':
            existing_content = stored_data.get('feature_overview', '')
            full_prompt = f"""You are editing the Feature Overview section. Maintain useful content unless instructed otherwise.

            Current content:
            {existing_content}

            User instruction:
            {new_content}
            """
            new_response = run_chat_agent(AGENT_FILES['agent_2'], full_prompt)
            stored_data['feature_overview'] = new_response
        elif content_type == 'highest_order':
            existing_content = stored_data.get('combined_outputs', {}).get('highest_order', '')
            full_prompt = f"""Here is the current High-Level Requirements section.

            {existing_content}

            Please revise it based on this instruction:
            {new_content}
            """
            new_response = run_chat_agent(AGENT_FILES['agent_3'], full_prompt)
            stored_data['combined_outputs']['highest_order'] = new_response
        elif content_type.startswith('agent_4_'):
            existing_content = stored_data.get('combined_outputs', {}).get(content_type, '')
            full_prompt = f"""You are updating the following section of a Product Requirements Document. Keep all useful information intact, and only modify based on the user request. Be conservative with deletions.

            {existing_content}

            User instruction:
            {new_content}
            """

            new_response = run_chat_agent(AGENT_FILES[content_type], full_prompt)
            stored_data['combined_outputs'][content_type] = new_response

        if USING_REDIS:
            redis_client.setex(session['data_key'], 3600, json.dumps(stored_data))
        else:
            with open(os.path.join(TEMP_DIR, f'prd_session_{session["data_key"]}.json'), 'w') as f:
                json.dump(stored_data, f)

        return jsonify({'success': True, 'response': new_response})

    except Exception as e:
        logging.error(f"Error updating content: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/page3', methods=['GET', 'POST'])
def page3():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    data = get_data(session.get('data_key', '')) or {}

    if request.method == 'POST':
        user_inputs = data.get("inputs", {})
        feature_overview = data.get("feature_overview", "")
        combined_input = f"""
        # Original User Inputs
        Industry: {user_inputs.get('industry', '')}
        Sector: {user_inputs.get('sector', '')}
        Geography: {user_inputs.get('geography', '')}
        Intent: {user_inputs.get('intent', '')}
        Features: {user_inputs.get('features', '')}

        # Feature Overview (Agent 2 Analysis)
        {feature_overview}
        """

        logging.info(f"[PAGE3] Combined input for agents: {combined_input}")

        keys = [k for k in AGENT_FILES if k.startswith("agent_4_")]

        async def call_agents_parallel_gemini(agent_calls):
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    None, run_chat_agent, AGENT_FILES[k], combined_input
                )
                for k, combined_input in agent_calls
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(call_agents_parallel_gemini([
            (k, combined_input) for k in keys
        ]))
        outputs = dict(zip(keys, results))

        data['combined_outputs'] = outputs
        session['combined_outputs'] = outputs

        if USING_REDIS:
            redis_client.setex(session['data_key'], 3600, json.dumps(data))
        else:
            with open(os.path.join(TEMP_DIR, f'prd_session_{session["data_key"]}.json'), 'w') as f:
                json.dump(data, f)

        return redirect('/page4')

    return render_template('page3_prompt_picker.html', highest_order=data.get('highest_order', ''))

@app.route('/page4')
def page4():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    try:
        data = get_data(session.get('data_key', ''))
        if not data:
            logger.error("No data found for session key")
            return "No data found", 404

        outputs = {
            'product_overview': data.get('product_overview', ''),
            'feature_overview': data.get('feature_overview', ''),
            'agent_4_1': data.get('combined_outputs', {}).get('agent_4_1', ''),
            'agent_4_2': data.get('combined_outputs', {}).get('agent_4_2', ''),
            'agent_4_3': data.get('combined_outputs', {}).get('agent_4_3', ''),
            'agent_4_4': data.get('combined_outputs', {}).get('agent_4_4', ''),
            'agent_4_5': data.get('combined_outputs', {}).get('agent_4_5', '')
        }
        logger.info(json.dumps(outputs, indent=2))

        logger.info("[PAGE4] Rendering with outputs:")
        for key, value in outputs.items():
            logger.info(f"{key}: {value[:100]}...")

        return render_template('page4_final_output.html', outputs=outputs)

    except Exception as e:
        logger.error(f"Error in page4: {str(e)}")
        return str(e), 500

# ... (rest of your utility/document functions remain unchanged) ...

@app.route("/review_prd", methods=["POST"])
def review_prd():
    """
    Agent 4.6: PRD Review and Quality Assurance Agent

    This endpoint processes requests to review a complete PRD using Agent 4.6.
    The agent analyzes all PRD sections for:
    - Completeness of requirements
    - Clarity and consistency
    - Missing critical information
    - Improvement suggestions

    Args:
        sections (dict): PRD sections to review, received as JSON in request body

    Returns:
        JSON response containing:
        - Array of issues found
        - Each issue includes: section name, identified problem, and suggestion for improvement
    """
    data = request.get_json()
    prd_sections = data.get("sections", {})

    full_prd = "\n\n".join(
        f"{key.replace('_', ' ').title()}:\n{value}"
        for key, value in prd_sections.items()
        if value.strip()
    )

    review_prompt = (
        "As the PRD Review and Quality Assurance Agent, analyze this PRD content. "
        "For each section, evaluate completeness, clarity, and identify any gaps. "
        "Provide specific, actionable improvements. "
        "Format response as JSON array with fields: section, issue, suggestion.\n\n"
        f"{full_prd}"
    )

    response = run_chat_agent(AGENT_FILES['agent_4_6'], review_prompt)
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = [{
            "section": "ReviewAgent",
            "issue": "Invalid JSON response format",
            "suggestion": "Please try the review again"
        }]

    return jsonify({"issues": parsed})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7003))
    print(f"Starting Flask server on port {port} ...")
    app.run(host="0.0.0.0", port=port, debug=True)