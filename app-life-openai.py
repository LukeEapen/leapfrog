import openai
from flask import Flask, request, jsonify, render_template
import time
import logging
import os
import sys
import codecs
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

if sys.platform.startswith('win'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_responses.log', encoding='utf-8')
    ]
)

app = Flask(__name__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

ASSISTANTS = {
    'agent_1': ("asst_EvIwemZYiG4cCmYc7GnTZoQZ", "Prompt Structuring Agent"),
    'agent_2': ("asst_EkihtJQe9qFiztRdRXPhiy2G", "Requirements Generator"),
    'agent_3': ("asst_Si7JAfL2Ov80wvcly6GKLJcN", "Validator Agent"),
    'agent_5': ("asst_r29PjUzVwfd6XiiYH3ueV41P", "Legal & Compliance Analyst"),
    'agent_6': ("asst_WG96Jp4VMrLJfUE4RMkGFMjf", "NFR Specialist"),
    'agent_7': ("asst_VWzoRLWbeZJS8I2IwtOcsLMp", "Platform Architect"),
    'agent_8': ("asst_sufR6Spw8EBDDoAzqQJN9iJt", "Requirement Functional - Legacy Parity Agent")
}

def call_agent(assistant_id: str, message: str, agent_name: str):
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

        while True:
            status = openai.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if status.status == "completed":
                break
            elif status.status == "failed":
                raise Exception(f"Run failed for {agent_name}")
            time.sleep(0.25)

        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        response = messages.data[0].content[0].text.value
        elapsed = time.time() - start_time

        logging.info(f"[{agent_name}] completed in {elapsed:.2f}s")
        return response

    except Exception as e:
        logging.error(f"[ERROR] {agent_name}: {str(e)}")
        return f"[ERROR] {str(e)}"

@app.route('/api/query_agents', methods=['POST'])
def query_agents():
    """Handle multi-phase agent processing with improved validation"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        phase = data.get('phase', 'agent_1')  # Default to agent_1 if no phase specified
        question = data.get('question', '').strip()

        logging.info(f"[START] Processing {phase}")
        logging.info(f"[INPUT] {question[:100]}...")  # Log truncated input

        # Phase-specific processing with validation
        if phase == 'agent_1':
            if not question:
                return jsonify({'error': 'Question is required for agent_1'}), 400
            response = call_agent(ASSISTANTS['agent_1'][0], question, ASSISTANTS['agent_1'][1])
            return jsonify({'agent_1': response})

        elif phase == 'agent_2':
            agent_1_output = data.get('agent_1_output', '').strip()
            if not agent_1_output:
                return jsonify({'error': 'agent_1_output is required for agent_2'}), 400
            response = call_agent(ASSISTANTS['agent_2'][0], agent_1_output, ASSISTANTS['agent_2'][1])
            return jsonify({'agent_2': response})

        elif phase == 'agent_3':
            agent_2_output = data.get('agent_2_output', '').strip()
            if not agent_2_output:
                return jsonify({'error': 'agent_2_output is required for agent_3'}), 400
            response = call_agent(ASSISTANTS['agent_3'][0], agent_2_output, ASSISTANTS['agent_3'][1])
            return jsonify({'agent_3': response})

        elif phase == 'agent_4':
            agent_2_output = data.get('agent_2_output', '').strip()
            if not agent_2_output:
                return jsonify({'error': 'agent_2_output is required for agent_4'}), 400

            def run_agent(agent_key):
                assistant_id, name = ASSISTANTS[agent_key]
                try:
                    response = call_agent(assistant_id, agent_2_output, name)
                    return f"**{name} Response:**\n{response}"
                except Exception as e:
                    logging.error(f"Error in {name}: {str(e)}")
                    return f"**{name} Response:**\nError: {str(e)}"

            # Process Agent 8 first, then others
            agent_keys = ['agent_8', 'agent_5', 'agent_6', 'agent_7']
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                agent_4_results = list(executor.map(run_agent, agent_keys))

            return jsonify({'agent_4': "\n\n".join(agent_4_results)})

        else:
            return jsonify({'error': f'Invalid phase: {phase}'}), 400

    except Exception as e:
        error_msg = str(e)
        logging.error(f"[ERROR] {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/agenticAI')
def index():
    return render_template('index-life-openai.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
