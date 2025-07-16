

import os
import openai
import time
import logging
from dotenv import load_dotenv
from flask import Flask, send_from_directory, redirect, url_for, request, jsonify

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')

# ...existing code...

# API endpoint to synthesize functions from user story and legacy English
@app.route('/api/synthesize-functions', methods=['POST'])
def api_synthesize_functions():
    try:
        data = request.get_json()
        user_story = data.get('user_story', '')
        legacy_english = data.get('legacy_english', '')
        if not user_story and not legacy_english:
            return jsonify({'error': 'No input provided'}), 400

        # Load system instructions
        instructions_path = os.path.join(os.path.dirname(__file__), 'agents', 'function_synthesizer_instructions.txt')
        with open(instructions_path, 'r', encoding='utf-8') as f:
            system_instructions = f.read()

        # Compose prompt
        prompt = f"""{system_instructions}\n\nUser Story:\n{user_story}\n\nLegacy English Description:\n{legacy_english}\n\nSynthesize functions as instructed."""

        # Call OpenAI GPT-3.5 with temperature 0.2 using new API
        chat_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        output = chat_response.choices[0].message.content
        return jsonify({'result': output})
    except Exception as e:
        logging.error(f"Error in synthesize-functions: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Route map for workflow
ROUTES = [
    ('/', 'legacy_code_parser_agent.html'),
    ('/user-story-decomposer', 'user_story_decomposer_agent.html'),
    ('/function-synthesizer', 'function_synthesizer_agent.html'),
    ('/design-decomposer', 'design_decomposer_agent.html'),
    ('/review-prompt', 'review_prompt_agent.html'),
    ('/service-builder', 'service_builder_agent.html'),
    ('/modernization-validator', 'modernization_validator_agent.html'),
]

# Dynamically create routes for each page
for route, html_file in ROUTES:
    def make_route(html_file):
        def route_func():
            return send_from_directory(app.template_folder, html_file)
        return route_func
    app.add_url_rule(route, endpoint=html_file, view_func=make_route(html_file))


# Optional: redirect /start to first page
@app.route('/start')
def start():
    return redirect(url_for('legacy_code_parser_agent.html'))


# Assistant IDs
assistant_id_agent_1 = "asst_P2HdYTBuZtBZqHGZizbsii1P"
assistant_id_agent_3 = "asst_bHzpgDT7IB6Bb80GpDmhOxcW"  # Replace with Agent 3's ID

def call_agent(assistant_id, message):
    """
    Calls an OpenAI assistant with a given message and returns the assistant's reply.
    This function creates a new conversation thread, sends a user message, runs the assistant,
    polls for completion, and retrieves the assistant's response. It logs the process and
    measures the elapsed time for the assistant to complete the task.
    Args:
        assistant_id (str): The unique identifier of the OpenAI assistant to call.
        message (str): The message to send to the assistant.
    Returns:
        str: The assistant's reply as a string.
    Raises:
        Exception: If the assistant run fails, is cancelled, expires, or any other error occurs during the process.
    """
    logging.info(f"Calling assistant with ID: {assistant_id} and message: {message}")
    start_time = time.time()
    try:
        # Step 1: Create a thread (conversation container)
        thread = openai.beta.threads.create()

        # Step 2: Add user message
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=message
        )

        # Step 3: Run the assistant
        run = openai.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        # Step 4: Poll for completion
        while True:
            run_status = openai.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_status.status == "completed":
                break
            elif run_status.status in ["failed", "cancelled", "expired"]:
                raise Exception(f"Run status: {run_status.status}")
            time.sleep(1)

        # Step 5: Get assistant's reply
        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        for msg in reversed(messages.data):
            if msg.role == "assistant":
                end_time = time.time()
                elapsed_time = end_time - start_time
                logging.info(f"Assistant {assistant_id} completed in {elapsed_time:.2f} seconds")
                return msg.content[0].text.value.strip()

    except Exception as e:
        logging.error(f"Error in call_agent for assistant {assistant_id}: {str(e)}")
        raise

@app.route('/upload', methods=['POST'])
def upload_file():
    import traceback
    try:
        if 'file' not in request.files:
            logging.error('No file part in request')
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            logging.error('No selected file')
            return jsonify({'error': 'No selected file'}), 400

        # Save file to uploads folder
        uploads_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)

        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            file_content = f.read()

        # Load system instructions for legacy code parser
        instructions_path = os.path.join(os.path.dirname(__file__), 'agents', 'legacy_code_parser_instructions.txt')
        with open(instructions_path, 'r', encoding='utf-8') as f:
            system_instructions = f.read()

        # Compose prompt for GPT-3.5
        prompt = f"""{system_instructions}\n\nLegacy Code:\n{file_content}\n\nTranslate and output as instructed."""

        # Call OpenAI GPT-3.5 with temperature 0.2 using new API
        chat_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": file_content}
            ],
            temperature=0.41,
            max_tokens=1024
        )
        output = chat_response.choices[0].message.content

        # Log success
        logging.info(f"Processed file: {file.filename} with Legacy Code Parser Agent")
        return jsonify({
            'response_agent_1': output
        })
    except Exception as e:
        logging.error(f"Error in /upload: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500


# Only run the server after all routes are defined
from jira_user_stories import fetch_user_stories

# API endpoint to get user stories from JIRA
@app.route('/api/jira-user-stories')
def api_jira_user_stories():
    try:
        project_key = request.args.get('project_key', 'SCRUM')
        stories = fetch_user_stories(project_key)
        return jsonify({'stories': stories})
    except Exception as e:
        logging.error(f"Error fetching user stories from JIRA: {str(e)}")
        return jsonify({'error': str(e)}), 500

# API endpoint to run User Story Decomposer Agent
@app.route('/api/decompose-user-story', methods=['POST'])
def api_decompose_user_story():
    try:
        data = request.get_json()
        user_story = data.get('user_story', '')
        if not user_story:
            return jsonify({'error': 'No user story provided'}), 400

        # Load system instructions
        instructions_path = os.path.join(os.path.dirname(__file__), 'agents', 'user_story_decomposer_instructions.txt')
        with open(instructions_path, 'r', encoding='utf-8') as f:
            system_instructions = f.read()

        # Compose prompt for GPT-3.5
        prompt = f"""{system_instructions}\n\nUser Story:\n{user_story}\n\nDecompose and output as instructed."""

        # Call OpenAI GPT-3.5 with temperature 0.2 using new API
        chat_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_story}
            ],
            temperature=0.2,
            max_tokens=512
        )
        output = chat_response.choices[0].message.content
        return jsonify({'result': output})
    except Exception as e:
        logging.error(f"Error in decompose-user-story: {str(e)}")
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True, port=5050)
