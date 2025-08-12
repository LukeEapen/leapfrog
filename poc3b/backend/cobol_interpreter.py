import os
"""
cobol_interpreter.py

A Flask web application for processing COBOL code files using OpenAI assistants.
The application allows users to upload COBOL source files, which are then processed
sequentially by two OpenAI assistants (Agent 1 and Agent 3). The responses from both
agents are returned as a JSON response.

Features:
- File upload endpoint for COBOL source files.
- Integration with OpenAI's API to call assistants for code analysis or transformation.
- Logging of assistant interactions and error handling.
- Environment variable management using dotenv.

Endpoints:
- '/' : Renders the main HTML page for file upload.
- '/upload' : Accepts file uploads, processes the file with two OpenAI assistants, and returns their responses.

Configuration:
- Requires an OPENAI_API_KEY in the environment.
- Uploads are saved to an 'uploads' directory in the current working directory.

Usage:
Run the script to start the Flask server. Upload a COBOL file via the web interface or API to receive processed results from the assistants.
"""
import openai
from flask import Flask, request, jsonify, render_template
import time
import logging
from dotenv import load_dotenv

app = Flask(__name__)

# Set the upload folder path
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

@app.route('/')
def index():
    return render_template('legacy.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles file upload via HTTP request, saves the uploaded file to the server, reads its contents,
    and processes it by sequentially calling two agents. Returns the responses from both agents as JSON.
    Returns:
        Response: A JSON response containing the outputs from both agents if successful,
                  or an error message with appropriate HTTP status code if an error occurs.
    """
    import traceback
    try:
        if 'file' not in request.files:
            logging.error('No file part in request')
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            logging.error('No selected file')
            return jsonify({'error': 'No selected file'}), 400

        # Ensure the upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Read the file and process it
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            cobol_code = f.read()

        # Call Agent 1
        response_agent_1 = call_agent(assistant_id_agent_1, cobol_code)

        # Call Agent 3 with the response of Agent 1
        response_agent_3 = call_agent(assistant_id_agent_3, response_agent_1)

        # Return both responses
        return jsonify({
            'response_agent_1': response_agent_1,
            'response_agent_3': response_agent_3
        })
    except Exception as e:
        logging.error(f"Error in /upload: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))  # Use Render's port or default
    app.run(host="0.0.0.0", port=port)