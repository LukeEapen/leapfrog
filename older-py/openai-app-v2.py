import openai
from flask import Flask, request, jsonify, render_template
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
import logging
import os
import sys
import codecs
import threading
import tiktoken
from typing import Optional
from dotenv import load_dotenv

# Configure UTF-8 for Windows
if sys.platform.startswith('win'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_responses.log', encoding='utf-8')
    ]
)

app = Flask(__name__)
thread_local = threading.local()

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Assistant configurations
ASSISTANTS = {
    'agent_1': ("asst_EvIwemZYiG4cCmYc7GnTZoQZ", "Prompt Structuring Agent"),
    'agent_2': ("asst_EkihtJQe9qFiztRdRXPhiy2G", "Requirements Generator"),
    'agent_3': ("asst_Si7JAfL2Ov80wvcly6GKLJcN", "Validator Agent"),
    'agent_5': ("asst_r29PjUzVwfd6XiiYH3ueV41P", "Legal & Compliance Analyst"),
    'agent_6': ("asst_WG96Jp4VMrLJfUE4RMkGFMjf", "NFR Specialist"),
    'agent_7': ("asst_VWzoRLWbeZJS8I2IwtOcsLMp", "Platform Architect"),
    'agent_8': ("asst_sufR6Spw8EBDDoAzqQJN9iJt", "Requirement Functional - Legacy Parity Agent")
}

def truncate_tokens(text: str, max_tokens: int = 3000) -> tuple[str, Optional[int]]:
    """Truncate text to max_tokens and return truncated text and token count"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = encoding.encode(text)
        token_count = len(tokens)
        
        if token_count <= max_tokens:
            return text, token_count
            
        truncated_tokens = tokens[:max_tokens]
        truncated_text = encoding.decode(truncated_tokens)
        return truncated_text, token_count
        
    except Exception as e:
        logging.warning(f"Token counting failed: {str(e)}")
        return text, None

def call_agent(assistant_id: str, message: str, agent_name: str, timeout: int = 45):
    """Enhanced OpenAI assistant caller with improved polling and timeout"""
    start_time = time.time()
    logging.info(f"[START] {agent_name} (ID: {assistant_id})")

    try:
        thread = openai.beta.threads.create()
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=message
        )

        run = openai.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        # Implement exponential backoff polling
        poll_interval = 0.5  # Start with 0.5 second
        max_interval = 2.0   # Max 2 seconds between polls
        elapsed = 0

        while elapsed < timeout:
            status = openai.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            
            if status.status == "completed":
                messages = openai.beta.threads.messages.list(thread_id=thread.id)
                response = messages.data[0].content[0].text.value
                total_time = time.time() - start_time
                logging.info(f"[{agent_name}] completed in {total_time:.2f}s")
                return response
            
            elif status.status == "failed":
                raise Exception(f"Run failed for {agent_name}")
            
            # Exponential backoff with max limit
            time.sleep(min(poll_interval, max_interval))
            poll_interval *= 1.5
            elapsed = time.time() - start_time

        raise TimeoutError(f"Operation timed out after {timeout} seconds")

    except Exception as e:
        logging.error(f"[ERROR] {agent_name}: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/api/agent_1', methods=['POST'])
def run_agent_1():
    """Handle Agent 1 requests"""
    question = request.json.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    output = call_agent(ASSISTANTS['agent_1'][0], question, ASSISTANTS['agent_1'][1])
    return jsonify({'output': output})

@app.route('/api/agent_2', methods=['POST'])
def run_agent_2():
    """Handle Agent 2 requests with token truncation and longer timeout"""
    input_text = request.json.get('input', '').strip()
    if not input_text:
        return jsonify({'error': 'No input provided'}), 400

    # Truncate tokens if needed
    truncated_text, token_count = truncate_tokens(input_text, max_tokens=3000)
    if token_count and token_count > 3000:
        logging.warning(f"Input truncated from {token_count} to 3000 tokens")

    output = call_agent(
        ASSISTANTS['agent_2'][0], 
        truncated_text, 
        ASSISTANTS['agent_2'][1],
        timeout=60
    )
    return jsonify({
        'output': output,
        'truncated': token_count > 3000 if token_count else False,
        'original_tokens': token_count
    })

@app.route('/api/agent_3', methods=['POST'])
def run_agent_3():
    """Handle Agent 3 requests"""
    input_text = request.json.get('input', '').strip()
    if not input_text:
        return jsonify({'error': 'No input provided'}), 400
        
    output = call_agent(ASSISTANTS['agent_3'][0], input_text, ASSISTANTS['agent_3'][1])
    return jsonify({'output': output})

@app.route('/api/agent_4', methods=['POST'])
def run_agent_4():
    """Call agents 5-8 in parallel with improved handling"""
    input_text = request.json.get('input', '').strip()
    if not input_text:
        return jsonify({'error': 'No input provided'}), 400

    def run(agent_key):
        assistant_id, name = ASSISTANTS[agent_key]
        result = call_agent(assistant_id, input_text, name, timeout=45)
        return f"**{name}**\n{result}"

    agent_keys = ['agent_8', 'agent_5', 'agent_6', 'agent_7']

    # Use thread pool with timeout
    with ThreadPoolExecutor(max_workers=4) as executor:
        try:
            futures = [executor.submit(run, key) for key in agent_keys]
            results = [f.result(timeout=50) for f in futures]  # 50s total timeout
            return jsonify({'output': "\n\n".join(results)})
        except Exception as e:
            logging.error(f"Parallel processing error: {str(e)}")
            return jsonify({'error': str(e)}), 500

@app.route('/agenticAI')
def index():
    """Serve the main UI"""
    return render_template('index-openai.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5007))
    print(f"\nStarting server...")
    print(f"Access the UI at: http://127.0.0.1:{port}/agenticAI")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=True,
        use_reloader=True
    )