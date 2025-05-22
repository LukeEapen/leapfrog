import asyncio
from openai import AsyncOpenAI
from flask import Flask, request, jsonify, render_template
import time
import logging
import os
import sys
import codecs
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Configure UTF-8 for Windows
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

# Initialize Flask and OpenAI
app = Flask(__name__)
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

async def call_agent_async(assistant_id: str, message: str, agent_name: str, timeout: int = 45):
    """Async function to call OpenAI assistant with timeout"""
    start_time = time.time()
    logging.info(f"[START] {agent_name}")

    try:
        # Create thread and message
        thread = await client.beta.threads.create()
        await client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=message
        )

        # Start the run
        run = await client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        # Poll with timeout
        elapsed = 0
        poll_interval = 0.5
        
        while elapsed < timeout:
            run_status = await client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            
            if run_status.status == "completed":
                messages = await client.beta.threads.messages.list(thread_id=thread.id)
                response = messages.data[0].content[0].text.value
                total_time = time.time() - start_time
                logging.info(f"[COMPLETE] {agent_name} in {total_time:.2f}s")
                return response
            
            elif run_status.status == "failed":
                raise Exception(f"Run failed for {agent_name}")
            
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Operation timed out after {timeout} seconds")

    except Exception as e:
        logging.error(f"[ERROR] {agent_name}: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/api/query_agents', methods=['POST'])
async def query_agents():
    """Handle agent queries with async processing"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        phase = data.get('phase', 'agent_1')
        logging.info(f"[PHASE] Processing {phase}")

        if phase == 'agent_1':
            question = data.get('question', '').strip()
            if not question:
                return jsonify({'error': 'Question required'}), 400
            
            response = await call_agent_async(
                ASSISTANTS['agent_1'][0],
                question,
                ASSISTANTS['agent_1'][1]
            )
            return jsonify({'agent_1': response})

        elif phase == 'agent_2':
            agent_1_output = data.get('agent_1_output', '').strip()
            if not agent_1_output:
                return jsonify({'error': 'Agent 1 output required'}), 400
            
            response = await call_agent_async(
                ASSISTANTS['agent_2'][0],
                agent_1_output,
                ASSISTANTS['agent_2'][1],
                timeout=60  # Longer timeout for Agent 2
            )
            return jsonify({'agent_2': response})

        elif phase == 'agent_3':
            agent_2_output = data.get('agent_2_output', '').strip()
            if not agent_2_output:
                return jsonify({'error': 'Agent 2 output required'}), 400
            
            response = await call_agent_async(
                ASSISTANTS['agent_3'][0],
                agent_2_output,
                ASSISTANTS['agent_3'][1]
            )
            return jsonify({'agent_3': response})

        elif phase == 'agent_4':
            agent_2_output = data.get('agent_2_output', '').strip()
            if not agent_2_output:
                return jsonify({'error': 'Agent 2 output required'}), 400

            # Process agents in parallel
            tasks = []
            for agent_key in ['agent_8', 'agent_5', 'agent_6', 'agent_7']:
                assistant_id, name = ASSISTANTS[agent_key]
                task = call_agent_async(assistant_id, agent_2_output, name)
                tasks.append(task)

            # Wait for all agents to complete
            responses = await asyncio.gather(*tasks)
            
            # Format responses maintaining order
            formatted_responses = [
                f"**{ASSISTANTS[key][1]} Response:**\n{response}"
                for key, response in zip(['agent_8', 'agent_5', 'agent_6', 'agent_7'], responses)
            ]

            return jsonify({'agent_4': "\n\n".join(formatted_responses)})

        else:
            return jsonify({'error': f'Invalid phase: {phase}'}), 400

    except Exception as e:
        logging.error(f"[ERROR] {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/agenticAI')
def index():
    """Serve the main UI"""
    return render_template('index-life-openai.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 6003))
    print(f"\nStarting server...")
    print(f"Access the UI at: http://127.0.0.1:{port}/agenticAI")
    
    # Use asyncio-compatible server
    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )