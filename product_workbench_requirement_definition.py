

from flask import Flask, jsonify, request, session
app = Flask(__name__)

# Ensure route is registered at top level
@app.route('/api/upload-prd-to-vector-db', methods=['POST'])
def upload_prd_to_vector_db():
    """
    API endpoint to upload the PRD to the Vector DB and return a link.
    Expects a POST request with PRD data in JSON format.
    """
    try:
        session_id = session.get('data_key', None)
        if not session_id:
            return jsonify({'success': False, 'error': 'No session key found'}), 400

        prd_data = get_data(session_id)
        if not prd_data:
            return jsonify({'success': False, 'error': 'No PRD data found'}), 404

        from docx import Document
        from io import BytesIO
        doc = Document()
        doc.add_heading("Product Requirements Document", level=1)
        doc.add_paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        doc.add_paragraph()
        doc.add_heading("Product Overview", level=2)
        doc.add_paragraph(prd_data.get("product_overview", ""))
        doc.add_heading("Feature Overview", level=2)
        doc.add_paragraph(prd_data.get("feature_overview", ""))
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        import requests
        files = {'prd_file': ('PRD_Draft.docx', buffer, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')}
        logger.info("Uploading PRD to Vector DB at /api/upload-to-vector-db...")
        try:
            response = requests.post('http://localhost:5001/api/upload-to-vector-db', files=files)
            logger.info(f"Upload response status: {response.status_code}")
            logger.info(f"Upload response text: {response.text}")
            result = response.json()
            if result.get('success'):
                prd_link = result.get('url')
                logger.info(f"PRD uploaded to Vector DB: {prd_link}")
                return jsonify({'success': True, 'link': prd_link})
            else:
                logger.error(f"Vector DB upload failed: {result.get('error')}")
                return jsonify({'success': False, 'error': result.get('error')}), 500
        except Exception as e:
            logger.error(f"Vector DB upload failed: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        logger.error(f"PRD upload failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


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
import random  # Add at top with other imports
from io import BytesIO
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor

# Security and authentication imports
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Third-party framework and API imports
import openai
import redis
from flask import (
    render_template, request, redirect, 
    url_for, session, send_file, jsonify
)
from dotenv import load_dotenv
from marshmallow import Schema, fields, ValidationError

# Document processing imports
from bs4 import BeautifulSoup
from markdown2 import markdown
from docx.shared import Pt, RGBColor  # Add RGBColor hereE
from docx import Document
import docx  # <-- Add this import for docx.Document usage
from docx.shared import Pt
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from vector_db_api import vector_db_api

########################
# REDIS CONFIGURATION
########################

# Redis import with fallback
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
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app.secret_key = os.getenv('FLASK_SECRET_KEY')

# Configure secure session settings
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=1)
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

executor = ThreadPoolExecutor(max_workers=10)
########################
# OPENAI ASSISTANT IDs
########################

# Map of assistant IDs to their roles
ASSISTANTS = {
    #'agent_1': 'asst_EvIwemZYiG4cCmYc7GnTZoQZ',
    'agent_1_1': 'asst_sW7IMhE5tQ78Ylx0zQkh6YnZ', # Agent 1.1 - Product Overview Synthesizer – System Instructions
    'agent_2'  : 'asst_t5hnaKy1wPvD48jTbn8Mx45z',   # Agent 2: Feature Overview Generator – System Instructions
    'agent_3'  : 'asst_EqkbMBdfOpUoEUaBPxCChVLR',   # Agent 3: Highest-Order Requirements Agent
    'agent_24_1': 'asst_Ed8s7np19IPmjG5aOpMAYcPM', # Agent 4.1: Product Requirements / User Stories Generator - System Instructions
    'agent_4_2': 'asst_CLBdcKGduMvSBM06MC1OJ7bF', # Agent 4.2: Operational Business Requirements Generator – System Instructions
    'agent_4_3': 'asst_61ITzgJTPqkQf4OFnnMnndNb', # Agent 4.3: Capability-Scoped Non-Functional Requirements Generator – System Instructions
    'agent_4_4': 'asst_pPFGsMMqWg04OSHNmyQ5oaAy', # Agent 4.4: Data Attribute Requirement Generator – System Instructions
    'agent_4_5': 'asst_wwgc1Zbl5iknlDtcFLOuTIjd',  # Agent 4.5: LRC: Legal, Regulatory, and Compliance Synthesizer – System Instructions
    'agent_4_6': 'asst_JOtY81FnKEkrhgcJmuJSDyip'
}

########################
# LOGGING CONFIGURATION
########################

# Define logging configuration
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
# Apply logging configuration
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

########################
# DATA STORAGE FUNCTIONS
########################

def store_data(data):
    """Store session data with unique ID in Redis or file system.
    
    Args:
        data (dict): Data to store, must be JSON serializable
        
    Returns:
        str: Unique session ID for retrieving data
        
    Raises:
        ValueError: If data is empty or cannot be serialized
    """
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    if USING_REDIS:
        redis_client.setex(session_id, 3600, json.dumps(data))
    else:
        with open(os.path.join(TEMP_DIR, f'prd_session_{session_id}.json'), 'w') as f:
            json.dump(data, f)
    return session_id

def get_data(session_id):
    """
    Retrieve session data for a given session_id from either Redis or file storage.

    Args:
        session_id (str): The unique identifier for the session.

    Returns:
        dict or None: The session data as a dictionary if found, otherwise None.

    Notes:
        - If USING_REDIS is True, attempts to fetch data from Redis.
        - If USING_REDIS is False or data is not found in Redis, attempts to fetch data from a JSON file in TEMP_DIR.
    """
    if USING_REDIS:
        # Try Redis first
        data = redis_client.get(session_id)
        return json.loads(data) if data else None
    else:
        # Check file storage
        path = os.path.join(TEMP_DIR, f'prd_session_{session_id}.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    return None

########################
# MONITORING & PERFORMANCE
########################
def monitor_agent_performance(func):
    """Decorator to track agent execution time and log performance.
    
    Wraps agent calls to:
    - Measure execution time
    - Log successful completions
    - Capture and log failures
    - Maintain performance metrics
    
    Args:
        func: The agent function to monitor
        
    Returns:
        wrapper: Decorated function with monitoring
    """
    @wraps(func)
    def wrapper(agent_id, input_text, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(agent_id, input_text, *args, **kwargs)
            duration = time.time() - start_time
            logging.info(f"Agent {agent_id} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"Agent {agent_id} failed after {duration:.2f}s: {str(e)}")
            raise
    return wrapper

async def wait_for_run_completion(thread_id, run_id, timeout=90, poll_interval=0.5):
    """Wait for OpenAI assistant run to complete with timeout.
    
    Args:
        thread_id (str): OpenAI thread identifier
        run_id (str): OpenAI run identifier
        timeout (int): Maximum seconds to wait (default: 90)
        poll_interval (float): Seconds between status checks (default: 0.5)
        
    Returns:
        dict: Completed run status
        
    Raises:
        TimeoutError: If run exceeds timeout
        RuntimeError: If run fails
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = openai.beta.threads.runs.retrieve(run_id=run_id, thread_id=thread_id)
        if status.status == "completed":
            return status
        elif status.status == "failed":
            raise RuntimeError(f"Run {run_id} failed.")
        await asyncio.sleep(poll_interval)
    raise TimeoutError(f"Run {run_id} did not complete in {timeout} seconds.")


@monitor_agent_performance
def call_agent(agent_id, input_text):
    """
    Calls an OpenAI assistant agent with the specified input text and returns the agent's response.
    Args:
        agent_id (str): The ID of the assistant agent to call.
        input_text (str): The input text to send to the agent.
    Returns:
        str: The response from the agent, or an error message if the call fails.
    Raises:
        ValueError: If thread, message, or run creation fails, or if no messages are returned.
        Exception: For any other unexpected errors during the agent call process.
    Logs:
        - Start and end of the agent call.
        - The agent's response.
        - Any errors encountered during the process.
    """
    try:
        logging.info(f"[CALL START] Calling agent {agent_id}")
        thread_key = f"thread_{agent_id}"
        if thread_key not in session:
            thread = openai.beta.threads.create()
            session[thread_key] = thread.id
        else:
            thread = openai.beta.threads.retrieve(session[thread_key])
        
        if not thread or not thread.id:
            raise ValueError("Failed to create thread")
            
        message = openai.beta.threads.messages.create(
            thread_id=thread.id, 
            role="user", 
            content=input_text
        )
        if not message:
            raise ValueError("Failed to create message")
            
        run = openai.beta.threads.runs.create(
            thread_id=thread.id, 
            assistant_id=agent_id
        )
        if not run:
            raise ValueError("Failed to create run")

        start_time = time.time()
        while time.time() - start_time < 60:  # 60 second timeout
            status = openai.beta.threads.runs.retrieve(
                thread_id=thread.id, 
                run_id=run.id
            )
            if status.status == "completed":
                break
            time.sleep(0.25)
            
        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        if not messages.data:
            raise ValueError("No messages returned")
            
        response = messages.data[0].content[0].text.value
        
        # Log the agent's response
        logging.info(f"""
        [AGENT RESPONSE] Agent {agent_id}:
        ----------------------------------------
        {response}
        ----------------------------------------
        """)
            
        return response

    except Exception as e:
        logging.error(f"[ERROR] Agent {agent_id} failed: {e}")
        return f"Error: {str(e)}"

@monitor_agent_performance
async def call_agent_async(agent_id, input_text):
    """
    Asynchronously calls an OpenAI agent with the specified agent ID and input text, manages the conversation thread,
    waits for the agent's response, and returns the response text.
    Args:
        agent_id (str): The unique identifier of the OpenAI agent (assistant) to call.
        input_text (str): The input message to send to the agent.
    Returns:
        str: The response text from the agent, or an error message if the call fails.
    Raises:
        Exception: Any exception encountered during the agent call is caught and logged, and an error message is returned.
    """
    try:
        logging.info(f"[CALL START] Calling agent {agent_id}")
        thread_key = f"thread_{agent_id}"
        if thread_key not in session:
            thread = openai.beta.threads.create()
            session[thread_key] = thread.id
        else:
            thread = openai.beta.threads.retrieve(session[thread_key])

        message = openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=input_text
        )

        run = openai.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=agent_id
        )

        # Use async polling
        await wait_for_run_completion(thread.id, run.id)

        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        response = messages.data[0].content[0].text.value

        logging.info(f"[AGENT RESPONSE] Agent {agent_id}:\n{response}")
        return response

    except Exception as e:
        logging.error(f"[ERROR] Agent {agent_id} failed: {e}")
        return f"Error: {str(e)}"


# Update call_agents_parallel to use retry logic
async def call_agents_parallel(agent_calls):
    """
    Executes multiple agent calls in parallel using asyncio.

    Args:
        agent_calls (list of tuple): A list of tuples, each containing an agent_id and input_text
            to be passed to the agent.

    Returns:
        list: A list of results returned by each agent call, in the same order as the input.

    Raises:
        Exception: Propagates any exceptions raised by the agent calls.

    Note:
        This function assumes that `call_agent_with_cache` is an async function that takes
        (agent_id, input_text) as arguments.
    """
    tasks = [
        call_agent_with_cache(agent_id, input_text)
        for agent_id, input_text in agent_calls
    ]
    return await asyncio.gather(*tasks)

agent_semaphore = asyncio.Semaphore(3)  # Limit concurrent calls

async def call_agent_with_limit(agent_id, input_text):
    """
    Asynchronously calls an agent with a concurrency limit.

    Args:
        agent_id (str): The identifier of the agent to call.
        input_text (str): The input text to send to the agent.

    Returns:
        Any: The result returned by the agent.

    Note:
        This function enforces a concurrency limit using an asynchronous semaphore.
    """
    async with agent_semaphore:
        return await call_agent_with_cache(agent_id, input_text)
def verify_credentials(username, password):
    """Verify user credentials against environment variables."""
    return (username == os.getenv('ADMIN_USERNAME') and 
            password == os.getenv('ADMIN_PASSWORD'))

@app.route('/', methods=['GET', 'POST'])
def login():
    """
    Handles user login functionality.

    If the request method is POST, verifies the provided username and password.
    - On successful authentication, sets the session as logged in and redirects to 'page1'.
    - On failure, renders the login page with an error message.

    If the request method is not POST, renders the login page without an error message.
    """
    if request.method == 'POST':
        if verify_credentials(request.form.get('username'), request.form.get('password')):
            session['logged_in'] = True
            return redirect(url_for('page1'))
        return render_template('page0_login.html', error=True)
    return render_template('page0_login.html', error=False)


@app.route('/page1', methods=['GET', 'POST'])
def page1():
    """
    Handles the logic for the first page of the workflow, including user authentication, form input processing, file upload handling, session data storage, and asynchronous agent execution.
    Workflow:
    - Checks if the user is logged in; redirects to login if not.
    - On POST request:
        - Collects form inputs: 'industry', 'sector', 'geography', 'intent', 'features'.
        - Handles optional file upload ('context_file'), reads and decodes its content, and adds it to the inputs.
        - Updates the session with input data.
        - Constructs a context string from the inputs and file content.
        - Stores the input data and status in a session store (Redis or file).
        - Defines and submits a background task to run multiple agents in parallel using the constructed context.
        - Stores agent outputs or error information in the session store.
        - Redirects the user to the next page ('/page2').
    - On GET request:
        - Renders the input form template ('page1_input.html').
    """
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':             # Get form inputs
        inputs = {key: request.form[key] for key in ['industry', 'sector', 'geography', 'intent', 'features']}
        MAX_FILE_SIZE_KB = 100
        # Handle file upload
        if 'context_file' in request.files:
                file = request.files['context_file']
                if file and file.filename:
                    file.seek(0, os.SEEK_END)
                    file_length = file.tell()
                    file.seek(0)  # Reset stream pointer

                    if file_length > MAX_FILE_SIZE_KB * 1024:
                        logging.warning("Uploaded file too large")
                        return "Uploaded file exceeds the 100KB limit.", 400
                    try:
                        filename = file.filename.lower()
                        if filename.endswith('.txt'):
                            # Read as UTF-8 text
                            file_content = file.read().decode('utf-8')
                        elif filename.endswith('.docx'):
                            # Read as docx (do not decode)
                            doc = docx.Document(file)
                            file_content = "\n".join([para.text for para in doc.paragraphs])
                        else:
                            file_content = "Unsupported file format."

                        # Add file content to inputs
                        inputs['context_file'] = file_content
                    except Exception as e:
                        logging.error(f"File upload error: {str(e)}")
                        return "Error processing file upload", 400

        session.update(inputs)

       # Update context to include file content
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

        # Ensure old data is cleared
        if USING_REDIS:
            redis_client.delete(session_id)
        else:
            path = os.path.join(TEMP_DIR, f'prd_session_{session_id}.json')
            if os.path.exists(path):
                os.remove(path)

        def run_agents():
            try:
                #a1 = call_agent(ASSISTANTS['agent_1'], context)
                a11, a2, a3 = asyncio.run(call_agents_parallel([
                    (ASSISTANTS['agent_1_1'], context),
                    (ASSISTANTS['agent_2'], context),
                    (ASSISTANTS['agent_3'], context)
                ]))
                logging.info(f"Agent 1.1 Output: {a11}")
                logging.info(f"Agent 2 Output: {a2}")
                final_data = {
                    #"agent_1_output": a1,
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
    """
    Handles the logic for the second page of the workflow.
    - Checks if the user is logged in; redirects to login if not.
    - Retrieves session-specific data and waits for a background job to complete, with a timeout.
    - Handles error and timeout cases during background processing.
    - On POST requests, updates the data with form inputs and persists it (optionally to Redis).
    - Renders the 'page2_agents.html' template with agent outputs for GET requests.
    Returns:
        - Redirects to login if not authenticated.
        - Error message and 500 status if agent processing fails.
        - Timeout message and 504 status if processing takes too long.
        - Redirects to page 3 on successful POST.
        - Renders the page with agent outputs on GET.
    """
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    session_id = session.get('data_key', '')
    data = get_data(session_id) or {}

    # Wait for background job to complete
    start_time = time.time()
    timeout = 120  # seconds

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
    """
    Handles updating content for a user's session based on the provided content type and new content.
    This endpoint requires the user to be authenticated. It expects a JSON payload with the following fields:
        - type: The type of content to update ('product' or 'feature').
        - content: The new content to process.
    Depending on the content type, the function re-runs the appropriate agent to generate a new overview, updates the stored session data, and persists the changes either in Redis or as a local file.
    Returns:
        - JSON response with {'success': True, 'response': <new_response>} on success.
        - JSON error message and appropriate HTTP status code on failure.
    """
    if not session.get('logged_in'):
        return jsonify({'error': 'Not authenticated'}), 401
        
    try:
        data = request.get_json()
        content_type = data.get('type')
        new_content = data.get('content')
        
        if not content_type or not new_content:
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Get stored data
        stored_data = get_data(session.get('data_key', '')) or {}
        
        # Update content based on type
        if content_type == 'product':
            # Re-run Agent 1.1 with new content
            existing_content = stored_data.get('product_overview', '')
            full_prompt = f"""You are editing the Product Overview section. Keep existing relevant content unless changes are requested.

            Current content:
            {existing_content}

            User instruction:
            {new_content}
            """
            new_response = call_agent(ASSISTANTS['agent_1_1'], full_prompt)
            stored_data['product_overview'] = new_response
        elif content_type == 'feature':
            # Re-run Agent 2 with new content
            existing_content = stored_data.get('feature_overview', '')
            full_prompt = f"""You are editing the Feature Overview section. Maintain useful content unless instructed otherwise.

            Current content:
            {existing_content}

            User instruction:
            {new_content}
            """
            new_response = call_agent(ASSISTANTS['agent_2'], full_prompt)
            stored_data['feature_overview'] = new_response
        elif content_type == 'highest_order':
            existing_content = stored_data.get('combined_outputs', {}).get('highest_order', '')
            full_prompt = f"""Here is the current High-Level Requirements section.

            {existing_content}

            Please revise it based on this instruction:
            {new_content}
            """
            new_response = call_agent(ASSISTANTS['agent_3'], full_prompt)
            stored_data['combined_outputs']['highest_order'] = new_response
        elif content_type.startswith('agent_4_'):
                agent_id = ASSISTANTS.get(content_type)
                if not agent_id:
                    return jsonify({'error': f'Unknown agent for {content_type}'}), 400
                existing_content = stored_data.get('combined_outputs', {}).get(content_type, '')
                full_prompt = f"""You are updating the following section of a Product Requirements Document. Keep all useful information intact, and only modify based on the user request. Be conservative with deletions.

                {existing_content}

                User instruction:
                {new_content}
                """

                new_response = call_agent(agent_id, full_prompt)
                stored_data['combined_outputs'][content_type] = new_response   

        # Store updated data
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
    """
    Handles the logic for the third page of the workflow, including user authentication, data retrieval, 
    processing user and agent inputs, parallel invocation of agent assistants, and storing combined outputs.
    Workflow:
    - Checks if the user is logged in; redirects to login if not.
    - Retrieves session data and prepares a combined input string from user inputs and agent analysis.
    - On POST requests:
        - Gathers relevant data and formats it for agent processing.
        - Runs multiple agent assistants in parallel using asyncio.
        - Stores the combined outputs in the session and persistent storage (Redis or file).
        - Redirects to the next page in the workflow.
    - On GET requests:
        - Renders the prompt picker template with the highest order data.
    Returns:
        - Redirects to login if not authenticated.
        - Redirects to page 4 after processing POST request.
        - Renders 'page3_prompt_picker.html' template on GET request.
    """
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    data = get_data(session.get('data_key', '')) or {}

    if request.method == 'POST':
        # Get user inputs and agent outputs
        user_inputs = data.get("inputs", {})
        feature_overview = data.get("feature_overview", "")
        # Combine all relevant inputs into structured format
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

        keys = [k for k in ASSISTANTS if k.startswith("agent_4_")]
        # Use asyncio to call all agents in parallel
        results = asyncio.run(call_agents_parallel([
            (ASSISTANTS[k], combined_input) for k in keys
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

_agent_response_cache = {}

async def call_agent_with_cache(agent_id, input_text):
    """
    Asynchronously calls an agent with the given input text, utilizing a cache to avoid redundant calls.
    Args:
        agent_id (str): The unique identifier for the agent to be called.
        input_text (str): The input text to be processed by the agent.
    Returns:
        Any: The result returned by the agent, either from cache or from a fresh call.
    Notes:
        - If the response for the given agent_id and input_text is already cached, the cached response is returned.
        - Otherwise, the agent is called and the response is cached for future use.
    """
    cache_key = f"{agent_id}_{hash(input_text)}"
    if cache_key in _agent_response_cache:
        logging.info(f"[CACHE HIT] Returning cached response for {agent_id}")
        return _agent_response_cache[cache_key]
    
    result = await call_agent_with_retry(agent_id, input_text)
    _agent_response_cache[cache_key] = result
    return result

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/page4')
@require_auth
def page4():
    """
    Handles the logic for rendering the fourth page of the workflow.
    - Checks if the user is logged in; redirects to login if not.
    - Retrieves session-specific data using a session key.
    - If data is missing, logs an error and returns a 404 response.
    - Extracts relevant output fields from the data for rendering.
    - Logs the output values (truncated for readability).
    - Renders the 'page4_final_output.html' template with the extracted outputs.
    - Handles and logs any exceptions, returning a 500 response on error.
    Returns:
        Response: Rendered HTML template with outputs, or an error message with appropriate HTTP status code.
    """
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    try:
        data = get_data(session.get('data_key', ''))
        if not data:
            logger.error("No data found for session key")
            return "No data found", 404  # Or render a friendly template

        outputs = {
            'product_overview': data.get('product_overview', ''),
            'feature_overview': data.get('feature_overview', ''),
           # 'agent_4_1': data.get('combined_outputs', {}).get('agent_4_1', ''),
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




def add_hyperlink(paragraph, url, text=None):
    """
    Adds a clickable hyperlink to a python-docx Paragraph object.
    Args:
        paragraph (docx.text.paragraph.Paragraph): The paragraph to which the hyperlink will be added.
        url (str): The URL that the hyperlink will point to.
        text (str, optional): The display text for the hyperlink. If None, the URL will be used as the display text.
    Returns:
        docx.text.paragraph.Paragraph: The modified paragraph with the hyperlink appended.
    Note:
        This function uses lxml and python-docx internals to create a hyperlink, as python-docx does not natively support hyperlinks.
    """
    if text is None:
        text = url
    part = paragraph.part
    r_id = part.relate_to(
        url, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True)
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)

    new_run = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    color = OxmlElement('w:color')
    color.set(qn('w:val'), '0000FF')
    rPr.append(color)
    underline = OxmlElement('w:u')
    underline.set(qn('w:val'), 'single')
    rPr.append(underline)
    new_run.append(rPr)

    new_run_text = OxmlElement('w:t')
    new_run_text.text = text
    new_run.append(new_run_text)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)
    return paragraph

def validate_reference(ref: str) -> bool:
    """Validate extracted reference."""
    if not ref or len(ref.strip()) < 5:
        return False
        
    # Check for common invalid patterns
    invalid_patterns = [
        r'^[\W_]+$',                    # Only special characters
        r'^(?:http)?[:/]+$',            # Incomplete URLs
        r'^[Rr]eference:\s*$',             # Empty reference labels
        r'^\d+$'                        # Just numbers
    ]
    
    return not any(re.match(pattern, ref.strip()) for pattern in invalid_patterns)

def extract_references_from_outputs(outputs):
    """
    Extracts and returns a sorted list of unique references found in the provided outputs.
    This function scans the values of the `outputs` dictionary, searching for references using a variety of regular expression patterns. It supports extraction of references in formats such as Markdown links, raw URLs, DOI, ISBN, and common citation styles. The extracted references are cleaned of extraneous punctuation and filtered for minimum length to reduce noise.
    Args:
        outputs (dict): A dictionary where each value is a string potentially containing references.
    Returns:
        list: A sorted list of unique, cleaned reference strings extracted from the outputs.
    """
    """Extract references from combined outputs."""
    all_refs = set()
    ref_patterns = [
        r'\[([^\]]+)\]\(([^\)]+)\)',                     # Markdown links
        r'Reference:\s+(.+?)(?=\n|$)',                   # Reference lines
        r'Source:\s+(.+?)(?=\n|$)',                      # Source lines
        r'(?:https?://\S+)',                             # Raw URLs
        r'(?<=Reference:).*?[\d]{4}.*?(?=\n|$)',        # Citations with years
        r'\((?:[^()]*?\d{4}[^()]*?)\)',                 # Parenthetical citations
        r'(?<=DOI:)\s*[\d.]+\/[\w.-]+',                 # DOI references
        r'(?:ISBN(?:-1[03])?:?\s*(?:[\d-]+))'          # ISBN references
    ]
    
    def extract_from_text(text):
        """
        Extracts references from the given text using predefined regular expression patterns.

        Iterates over a set of reference patterns (`ref_patterns`), finds all matches in the input text,
        and collects unique references. Handles markdown links differently by extracting the URL part
        if available, otherwise adds the entire match. For other patterns, adds the full matched string.
        Logs a warning if an error occurs during extraction.

        Args:
            text (str): The input text from which to extract references.

        Returns:
            set: A set of unique references extracted from the text.
        """
        refs = set()
        for pattern in ref_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                try:
                    # Handle markdown links differently
                    if '[' in pattern:
                        # Check if we have enough groups before accessing
                        if len(match.groups()) >= 2:
                            refs.add(match.group(2))  # Add URL from markdown link
                        else:
                            refs.add(match.group(0))  # Add entire match if not enough groups
                    else:
                        # For other patterns, add the full match
                        refs.add(match.group().strip())
                except (IndexError, AttributeError) as e:
                    logging.warning(f"Error extracting reference: {e}")
                    continue
        return refs

    # Extract from all outputs
    for output in outputs.values():
        if isinstance(output, str):
            try:
                refs = extract_from_text(output)
                all_refs.update(refs)
            except Exception as e:
                logging.error(f"Error processing output text: {e}")
                continue
    
    # Clean and sort references
    cleaned_refs = sorted(list({
        ref.strip(' .,[]()\"\'')  # Clean up punctuation
        for ref in all_refs
        if ref and len(ref.strip()) > 5  # Minimum length to filter noise
    }))
    
    logging.info(f"[REFERENCES] Extracted {len(cleaned_refs)} references")
    return cleaned_refs

limiter = Limiter(app=app, key_func=get_remote_address)

# Replace with the new route
@app.route('/generate_word_doc', methods=['POST'])
@limiter.limit("5 per minute")
def generate_word_doc():
    """
    Generates a Product Requirements Document (PRD) as a Word file from session data.
    This function performs the following steps:
    1. Checks user authentication via session.
    2. Retrieves and validates PRD data from the session.
    3. Initializes a Word document with custom styles.
    4. Maps and processes PRD sections, adding them to the document.
    5. Handles and logs errors for individual sections.
    6. Adds a references section to the document.
    7. Streams the generated document as a downloadable .docx file.
    Returns:
        Flask response: A Word document file if successful, or a JSON error response with appropriate HTTP status code.
    """
    """Generate Word document from PRD content."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        # Get data and validate
        data = get_data(session.get("data_key", ""))
        if not data:
            logging.error("No data found in session")
            return jsonify({'error': 'No session data found'}), 404

        # Initialize document with styles
        doc = Document()
        initialize_document_styles(doc)

        # Map all sections
        sections = {
            "Product Overview": data.get("product_overview", ""),
            "Feature Overview": data.get("feature_overview", ""),
         #   "Product Requirements": data.get("combined_outputs", {}).get("agent_4_1", ""),
            "Functional Requirements": data.get("combined_outputs", {}).get("agent_4_2", ""),
            "Non-Functional Requirements": data.get("combined_outputs", {}).get("agent_4_3", ""),
            "Data Requirements": data.get("combined_outputs", {}).get("agent_4_4", ""),
            "Legal & Compliance Requirements": data.get("combined_outputs", {}).get("agent_4_5", "")
        }

        # Add title and metadata
        doc.add_heading("Product Requirements Document", level=1)
        doc.add_paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        doc.add_paragraph()

        # Process each section
        for title, content in sections.items():
            if content and content.strip():
                try:
                    process_section(doc, title, content)
                except Exception as e:
                    logging.error(f"Error processing section {title}: {str(e)}")
                    doc.add_paragraph(f"Error in section {title}: {str(e)}", style="Error")

        # Add references
        add_references_section(doc, data.get("combined_outputs", {}))

        # Save to buffer
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        logging.info("Document generated successfully")
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"PRD_Draft_{time.strftime('%Y%m%d')}.docx",
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )

    except Exception as e:
        logging.error(f"Document generation failed: {str(e)}")
        return jsonify({
            'error': 'Document generation failed',
            'message': str(e)
        }), 500
    

def add_references_section(doc, outputs):
    """
    Adds a "References" section to a document using references extracted from the provided outputs.
    Args:
        doc: The document object to which the references section will be added.
        outputs: The outputs from which references are to be extracted.
    Raises:
        Logs an error if any exception occurs during the process.
    The function extracts references, adds a page break and a heading for the references section,
    and then adds each reference as a bullet point. If a reference is a URL, it is added as a hyperlink.
    """
    """Add references section to document."""
    try:
        refs = extract_references_from_outputs(outputs)
        if not refs:
            return

        doc.add_page_break()
        doc.add_heading("References", level=2)
        
        for ref in refs:
            p = doc.add_paragraph(style="List Bullet")
            if ref.startswith(("http://", "https://")):
                add_hyperlink(p, ref)
            else:
                p.add_run(ref)
                
    except Exception as e:
        logging.error(f"Error adding references: {str(e)}")

def process_section(doc, title, raw_markdown):
    """
    Processes a single section of a document by adding a heading, converting Markdown content to HTML,
    parsing the HTML, and processing its elements for inclusion in the document.
    Args:
        doc: The document object to which the section will be added.
        title (str): The title of the section to be added as a heading.
        raw_markdown (str): The raw Markdown content of the section.
    Raises:
        Logs any exceptions encountered during processing and adds an error note to the document.
    """
    """Process a single section of the document."""
    try:
        doc.add_heading(title, level=2)
        
        # Convert markdown to HTML
        html_content = markdown(
            raw_markdown,
            extras=[
                "fenced-code-blocks",
                "tables",
                "header-ids",
                "break-on-newline"
            ]
        )
        
        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Process elements
        for el in soup.find_all(["p", "li", "h3", "h4", "pre", "code"]):
            process_element(doc, el)
            
    except Exception as e:
        logging.error(f"Error processing section {title}: {str(e)}")
        # Add error note in document
        doc.add_paragraph(f"Error processing section: {str(e)}", style="Error")

def process_element(doc, el):
    """
    Processes a single HTML element and adds its content to a Word document paragraph.
    Args:
        doc: A python-docx Document object to which the paragraph will be added.
        el: A BeautifulSoup Tag object representing the HTML element to process.
    Behavior:
        - Extracts and strips text from the HTML element.
        - If the text matches a requirement label pattern (e.g., "F123. Description"), adds the label in bold followed by the description.
        - If the text matches a labeled line pattern (e.g., "Label: Description"), adds the label in bold followed by the description.
        - Otherwise, adds the text as a regular paragraph. If the element is a list item ("li"), applies the "List Bullet" style.
        - Sets the paragraph's space after to 6 points.
    Returns:
        None
    """
    """Process a single HTML element."""
    text = el.get_text(strip=True)
    if not text:
        return

    para = doc.add_paragraph()
    
    # Handle requirement labels
    if match := re.match(r"^(F\d+|NFR\d+|DR\d+|LR\d+)\. (.+)", text):
        label, desc = match.groups()
        run = para.add_run(f"{label}. ")
        run.bold = True
        para.add_run(desc)
    # Handle labeled lines
    elif match := re.match(r"^([A-Z][\w\s]+?): (.+)", text):
        label, desc = match.groups()
        run = para.add_run(f"{label}: ")
        run.bold = True
        para.add_run(desc)
    # Regular text
    else:
        para.add_run(text)
        if el.name == "li":
            para.style = "List Bullet"
    
    para.paragraph_format.space_after = Pt(6)

@app.errorhandler(Exception)
def handle_exception(e):
    """
    Handles unhandled exceptions by logging the error and returning a standardized JSON response.

    Args:
        e (Exception): The exception instance that was raised.

    Returns:
        tuple: A tuple containing a JSON response with error details and the HTTP status code 500.
    """
    """Handle all unhandled exceptions."""
    logging.error(f"Unhandled exception: {str(e)}")
    return jsonify({
        'error': 'An unexpected error occurred',
        'message': str(e)
    }), 500

@app.errorhandler(404)
def not_found_error(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(401)
def unauthorized_error(e):
    """Handle 401 errors."""
    return redirect(url_for('login'))

def process_section(doc, title, raw_markdown):
    """Process a single section of the document."""
    try:
        doc.add_heading(title, level=2)
        
        # Convert markdown to HTML
        html_content = markdown(
            raw_markdown,
            extras=[
                "fenced-code-blocks",
                "tables",
                "header-ids",
                "break-on-newline"
            ]
        )
        
        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Process elements
        for el in soup.find_all(["p", "li", "h3", "h4", "pre", "code"]):
            process_element(doc, el)
            
    except Exception as e:
        logging.error(f"Error processing section {title}: {str(e)}")
        # Add error note in document
        doc.add_paragraph(f"Error processing section: {str(e)}", style="Error")

def initialize_document_styles(doc):
    """
    Initializes custom paragraph styles for a Word document.
    Adds the following styles to the document:
    - 'Error': Red, italicized text for error messages.
    - 'Requirement': Calibri font, size 11pt for requirement sections.
    Args:
        doc (docx.document.Document): The Word document object to which styles will be added.
    Returns:
        docx.document.Document: The document object with the new styles applied.
    """
    """Initialize document styles."""
    styles = doc.styles
    
    # Error style
    error_style = styles.add_style('Error', WD_STYLE_TYPE.PARAGRAPH)
    error_style.font.color.rgb = RGBColor(255, 0, 0)  # Now RGBColor will be defined
    error_style.font.italic = True
    
    # Requirement style
    req_style = styles.add_style('Requirement', WD_STYLE_TYPE.PARAGRAPH)
    req_style.font.name = 'Calibri'
    req_style.font.size = Pt(11)
    
    return doc

def cleanup_expired_sessions():
    """
    Removes expired session files from the temporary directory if Redis is not used.
    This function iterates through files in the TEMP_DIR directory, identifying files
    that start with 'prd_session_'. If a file's last modification time is more than
    one hour ago, it attempts to delete the file. The function logs the number of
    files successfully removed and any failures encountered during the process.
    If Redis is used for session management, the function exits early since Redis
    handles expiration automatically. All errors are logged for troubleshooting.
    Raises:
        Logs exceptions and errors encountered during file removal or directory access.
    """
    """Clean up expired session data with better error handling."""
    if USING_REDIS:
        return  # Redis handles expiration
    # TODO: Implement file cleanup logic here
        
    try:
        current_time = time.time()
        cleaned = 0
        failed = 0
        
        for filename in os.listdir(TEMP_DIR):
            if not filename.startswith('prd_session_'):
                continue
                
            filepath = os.path.join(TEMP_DIR, filename)
            try:
                if current_time - os.path.getmtime(filepath) > 3600:
                    os.remove(filepath)
                    cleaned += 1
            except OSError as e:
                failed += 1
                logging.error(f"Failed to remove {filepath}: {e}")
                
        logging.info(f"Session cleanup: {cleaned} removed, {failed} failed")
        
    except Exception as e:
        logging.error(f"Session cleanup failed: {e}")

class UserInputSchema(Schema):
    """
    Schema for validating user input data.

    Attributes:
        industry (str): The industry to which the user input pertains. Required.
        sector (str): The sector within the specified industry. Required.
        geography (str): The geographical region relevant to the input. Required.
        intent (str): The user's intent or purpose for the input. Required.
        features (str): The features or attributes specified by the user. Required.
    """
    industry = fields.Str(required=True)
    sector = fields.Str(required=True)
    geography = fields.Str(required=True)
    intent = fields.Str(required=True)
    features = fields.Str(required=True)

def validate_form_input(form_data):
    """
    Validates and sanitizes form input data against a predefined schema.
    This function first strips whitespace from all input values, then validates the sanitized data using the `UserInputSchema`.
    It enforces additional rules:
    - Each input value must be at least 3 characters and no more than 1000 characters long.
    - Input values must not contain any of the following characters: `<`, `>`, `{`, `}`, `[`, `]`.
    Args:
        form_data (dict): Dictionary containing form input data as key-value pairs.
    Returns:
        tuple:
            - bool: True if validation succeeds, False otherwise.
            - dict: Validated data if successful, or a dictionary of error messages if validation fails.
    Raises:
        None. All exceptions are caught and logged internally.
    """
    """Validate form input against schema with improved sanitization."""
    try:
        # Sanitize input before validation
        sanitized_data = {
            key: value.strip() for key, value in form_data.items()
        }
        
        schema = UserInputSchema()
        validated_data = schema.load(sanitized_data)
        
        # Enhanced validation rules
        for key, value in validated_data.items():
            if len(value) < 3:
                return False, {key: ["Input too short (minimum 3 characters)"]}
            if len(value) > 1000:
                return False, {key: ["Input too long (maximum 1000 characters)"]}
            if any(char in value for char in '<>{}[]'):
                return False, {key: ["Invalid characters detected"]}
                
        return True, validated_data
        
    except ValidationError as e:
        logging.warning(f"Validation error: {e.messages}")
        return False, e.messages
    except Exception as e:
        logging.error(f"Unexpected validation error: {str(e)}")
        return False, {"error": "Internal validation error"}
    
def extend_session():
    """
    Extends the user's session lifetime if they are currently logged in.

    If the user is logged in, this function sets the session to be permanent,
    updates the application's permanent session lifetime to one hour, and marks
    the session as modified to ensure the changes are saved.
    """
    """Extend session lifetime if user is active."""
    if session.get('logged_in'):
        session.permanent = True
        app.permanent_session_lifetime = timedelta(hours=1)
        session.modified = True

@app.before_request
def before_request():
    """
    Performs pre-request operations such as extending the user session and occasionally cleaning up expired sessions.

    This function should be called before handling each request. It ensures the session is kept active and, with a small probability (1% per call), triggers cleanup of expired sessions to manage resources efficiently.
    """
    extend_session()
    # Add session cleanup check every 100 requests
    if random.random() < 0.01:  # 1% chance
        cleanup_expired_sessions()

########################
# OPENAI AGENT INTERACTION
########################
@monitor_agent_performance
async def call_agent_with_retry(agent_id, input_text, max_retries=3):
    """Call OpenAI assistant with exponential backoff retry logic.
    
    Args:
        agent_id (str): OpenAI assistant identifier
        input_text (str): Input prompt for the agent
        max_retries (int): Maximum retry attempts (default: 3)
        
    Returns:
        str: Agent response text
        
    Raises:
        RuntimeError: If all retry attempts fail
    """
    for attempt in range(max_retries):
        try:
            # Create thread with initial message
            thread = openai.beta.threads.create(messages=[{"role": "user", "content": input_text}])
            if not thread or not thread.id:
                raise ValueError("Thread creation failed")

            # Start agent run
            run = openai.beta.threads.runs.create(thread_id=thread.id, assistant_id=agent_id)

            # Use the async event loop to wait non-blockingly
            status = await wait_for_run_completion(thread.id, run.id)

            # Get response messages
            messages = openai.beta.threads.messages.list(thread_id=thread.id)
            response = messages.data[0].content[0].text.value if messages.data else ""
            return response

        except Exception as e:
            print(f"[Retry {attempt+1}/{max_retries}] Error with agent {agent_id}: {e}")
            if attempt == max_retries - 1:
                return f"Error: {str(e)}"
            await asyncio.sleep(1.5)
            
def store_session_data(data):
    """
    Stores large session data externally (in Redis or as a file) and saves only a reference key in the session.
    Args:
        data (dict): The session data to be stored. Must be serializable to JSON.
    Returns:
        str: The unique session ID used as the storage key.
    Raises:
        ValueError: If the input data is empty or not serializable to JSON.
        Exception: If storage fails due to unforeseen errors.
    Behavior:
        - Serializes the input data to JSON.
        - Attempts to store the data in Redis if enabled; falls back to file storage on failure.
        - Stores only the generated session ID in the session under 'data_key'.
        - Logs all major actions and errors.
    """
    """Store large session data in Redis/files and keep only key in session."""
    try:
        if not data:
            raise ValueError("Cannot store empty data")
            
        session_id = str(uuid.uuid4())
        storage_key = f"prd_data_{session_id}"
        
        # Convert data to JSON
        try:
            json_data = json.dumps(data)
        except (TypeError, ValueError) as e:
            logging.error(f"JSON serialization failed: {e}")
            raise ValueError("Invalid data format")
        
        # Store main data with error handling
        if USING_REDIS:
            try:
                redis_client.setex(storage_key, 3600, json_data)
            except redis.RedisError as e:
                logging.error(f"Redis storage failed: {e}")
                # Fallback to file storage
                with open(os.path.join(TEMP_DIR, f'prd_session_{session_id}.json'), 'w') as f:
                    f.write(json_data)
        else:
            with open(os.path.join(TEMP_DIR, f'prd_session_{session_id}.json'), 'w') as f:
                f.write(json_data)
                
        # Store minimal data in session
        session['data_key'] = session_id
        logging.info(f"Session data stored with ID: {session_id}")
        return session_id
        
    except Exception as e:
        logging.error(f"Failed to store session data: {e}")
        raise

def get_session_data():
    """
    Retrieve session data from storage with validation.
    Attempts to retrieve session data using a session key. First tries to fetch data from Redis if enabled,
    falling back to file-based storage if Redis is unavailable or retrieval fails. Handles and logs errors
    during retrieval and deserialization. Returns the session data as a Python object if successful, or None
    if retrieval fails or no session key is found.
    """
    """Retrieve session data from storage with validation."""
    try:
        session_id = session.get('data_key')
        if not session_id:
            logging.warning("No session key found")
            return None
            
        storage_key = f"prd_data_{session_id}"
        
        # Try Redis first
        if USING_REDIS:
            try:
                data = redis_client.get(storage_key)
                if data:
                    return json.loads(data)
            except (redis.RedisError, json.JSONDecodeError) as e:
                logging.error(f"Redis retrieval failed: {e}")
        
        # Fallback to file storage
        path = os.path.join(TEMP_DIR, f'prd_session_{session_id}.json')
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                logging.error(f"File retrieval failed: {e}")
                
        return None
        
    except Exception as e:
        logging.error(f"Session data retrieval failed: {e}")
        return None
        
    return get_data(session_id)

@app.route('/chat_with_agent', methods=['POST'])
def chat_with_agent():
    if not session.get('logged_in'):
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        data = request.get_json()
        message = data.get('message', '')
        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # 👇 THIS is where it talks to Agent 1.1
        reply = call_agent(ASSISTANTS['agent_1_1'], message)

        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/chat_agent3", methods=["POST"])
@require_auth
def chat_agent3():
    try:
        data = request.get_json()
        user_msg = data.get("message", "")
        session_id = session.get("data_key", "")
        current_data = get_data(session_id)

        # Call Agent 3
        agent_reply = call_agent(ASSISTANTS['agent_3'], user_msg)

        # Update Highest Order section
        if current_data:
            current_data["highest_order"] = agent_reply
            if USING_REDIS:
                redis_client.setex(session_id, 3600, json.dumps(current_data))
            else:
                with open(os.path.join(TEMP_DIR, f'prd_session_{session_id}.json'), 'w') as f:
                    json.dump(current_data, f)

        return jsonify({"reply": agent_reply})

    except Exception as e:
        logging.error(f"Chat Agent 3 error: {str(e)}")
        return jsonify({"error": str(e)}), 500
     

@app.route("/review_prd", methods=["POST"])
def review_prd():
    """
    Agent 4.6: PRD Review and Quality Assurance Agent
    
    This endpoint processes requests to review a complete PRD using Agent 4.6 (asst_JOtY81FnKEkrhgcJmuJSDyip).
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

  # Construct review prompt for Agent 4.6
    review_prompt = (
        "As the PRD Review and Quality Assurance Agent, analyze this PRD content. "
        "For each section, evaluate completeness, clarity, and identify any gaps. "
        "Provide specific, actionable improvements. "
        "Format response as JSON array with fields: section, issue, suggestion.\n\n"
        f"{full_prd}"
    )

    # Call Agent 4.6 (PRD Review Agent)
    response = call_agent(ASSISTANTS['agent_4_6'], review_prompt)
   # response = call_openai_agent("agent_4_6", review_prompt)  # Assumes this helper exists
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = [{
            "section": "ReviewAgent", 
            "issue": "Invalid JSON response format", 
            "suggestion": "Please try the review again"
        }]

    return jsonify({"issues": parsed})
    
# Add to main
if __name__ == '__main__':
    cleanup_expired_sessions()
    app.register_blueprint(vector_db_api)
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)

