import openai
from flask import Flask, request, jsonify, render_template
import time
import logging
import os
import sys
import codecs
from dotenv import load_dotenv

# Force UTF-8 encoding for Windows console
if sys.platform.startswith('win'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_responses.log', encoding='utf-8')
    ]
)

app = Flask(__name__)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Assistant IDs with descriptions
ASSISTANTS = {
    'agent_1': ("asst_EvIwemZYiG4cCmYc7GnTZoQZ", "Prompt Structuring Agent"),
    'agent_2': ("asst_EkihtJQe9qFiztRdRXPhiy2G", "Requirements Generator"),
    'agent_3': ("asst_Si7JAfL2Ov80wvcly6GKLJcN", "Validator Agent"),
    'agent_5': ("asst_r29PjUzVwfd6XiiYH3ueV41P", "Legal & Compliance Analyst"),
    'agent_6': ("asst_WG96Jp4VMrLJfUE4RMkGFMjf", "NFR Specialist"),
    'agent_7': ("asst_VWzoRLWbeZJS8I2IwtOcsLMp", "Platform Architect"),
    'agent_8': ("asst_sufR6Spw8EBDDoAzqQJN9iJt", "Requirement Functional - Legacy Parity Agent")
}

def print_agent_response(agent_name: str, response: str, elapsed_time: float = None):
    """Print agent response with ASCII-safe formatting"""
    separator = "=" * 80
    logging.info(f"\n{separator}")
    logging.info(f"[AGENT] {agent_name}")
    if elapsed_time:
        logging.info(f"[TIME] {elapsed_time:.2f} seconds")
    logging.info(f"[RESPONSE]\n{response}")
    logging.info(f"{separator}\n")

def call_agent(assistant_id: str, message: str, agent_name: str):
    """Call OpenAI assistant with enhanced logging"""
    start_time = time.time()
    logging.info(f"[START] {agent_name} (ID: {assistant_id})")

    try:
        # Create thread and send message
        thread = openai.beta.threads.create()
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=message
        )

        # Run the assistant
        run = openai.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        # Poll for completion
        while True:
            run_status = openai.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_status.status == "completed":
                break
            elif run_status.status == "failed":
                raise Exception(f"Run failed for {agent_name}")
            time.sleep(1)

        # Get response
        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        response = messages.data[0].content[0].text.value
        elapsed_time = time.time() - start_time
        
        print_agent_response(agent_name, response, elapsed_time)
        return response

    except Exception as e:
        logging.error(f"[ERROR] {agent_name}: {str(e)}")
        raise

@app.route('/api/query_agents', methods=['POST'])
def query_agents():
    """Handle agent pipeline with sequential and parallel processing"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        logging.info("[START] New query processing")
        logging.info(f"[INPUT] {question}")

        # Sequential processing
        agent_1_response = call_agent(
            ASSISTANTS['agent_1'][0], 
            question, 
            "Prompt Structuring Agent"
        )
        
        agent_2_response = call_agent(
            ASSISTANTS['agent_2'][0], 
            agent_1_response, 
            "Requirements Generator"
        )
        
        agent_3_response = call_agent(
            ASSISTANTS['agent_3'][0], 
            agent_2_response, 
            "Validator Agent"
        )

        # Parallel processing
        parallel_responses = {}
        for agent_key in ['agent_5', 'agent_6', 'agent_7', 'agent_8']:
            assistant_id, agent_desc = ASSISTANTS[agent_key]
            parallel_responses[agent_key] = call_agent(
                assistant_id,
                agent_2_response,
                agent_desc
            )

        # Combine parallel responses
        agent_4_response = "\n\n".join([
            f"**{ASSISTANTS[agent][1]} Response:**\n{response}"
            for agent, response in parallel_responses.items()
        ])

        logging.info("[COMPLETE] Query processing finished")
        
        return jsonify({
            'agent_1': agent_1_response,
            'agent_2': agent_2_response,
            'agent_3': agent_3_response,
            'agent_4': agent_4_response
        })

    except Exception as e:
        logging.error(f"[ERROR] Processing failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/agenticAI')
def index():
    """Render the main UI template."""
    return render_template('index-new.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logging.info("[SERVER] Starting on port %d", port)
    app.run(host="0.0.0.0", port=port)