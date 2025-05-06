import openai
from flask import Flask, request, jsonify, render_template
import time
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)


#Personal platform.openai.com
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Assistant IDs
assistant_id_agent_1 = "asst_EvIwemZYiG4cCmYc7GnTZoQZ"
assistant_id_agent_2 = "asst_EkihtJQe9qFiztRdRXPhiy2G"
assistant_id_agent_3 = "asst_Si7JAfL2Ov80wvcly6GKLJcN"
assistant_id_agent_4 = "asst_4IDFX2ublv3tJ1L4L7yVDPLr"


def call_agent(assistant_id, message):
    logging.info(f"Calling agent {assistant_id} with message: {message}")
    start_time = time.time()

    try:
        # Create a thread and send the message
        thread = openai.beta.threads.create()
        logging.debug(f"Thread created with ID: {thread.id}")
        
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=message
        )
        logging.debug(f"Message sent to thread {thread.id}")

        # Start the run
        run = openai.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
        logging.debug(f"Run started for agent {assistant_id} with run ID: {run.id}")

        # Poll for the run status
        while True:
            run_status = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            logging.debug(f"Run status for agent {assistant_id}: {run_status.status}")
            
            if run_status.status == "completed":
                logging.info(f"Run completed for agent {assistant_id}")
                break
            elif run_status.status == "failed":
                logging.error(f"Run failed for agent {assistant_id}")
    
                # Log more detail if available
                if hasattr(run_status, 'last_error') and run_status.last_error:
                    logging.error(f"Last error for agent {assistant_id}: {run_status.last_error}")
                else:
                    logging.debug(f"Full run status object: {run_status}")

                raise Exception(f"Run failed for agent {assistant_id}")
            time.sleep(1)

        # Retrieve messages from the thread
        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        for msg in messages.data:
            if msg.role == "assistant":
                end_time = time.time()
                elapsed_time = end_time - start_time
                logging.info(f"Agent {assistant_id} completed in {elapsed_time:.2f} seconds")
                return msg.content[0].text.value

        logging.error(f"No assistant response found for agent {assistant_id}")
        raise Exception(f"No response from agent {assistant_id}")

    except Exception as e:
        logging.error(f"Error in call_agent for agent {assistant_id}: {str(e)}")
        raise

@app.route('/api/query_agents', methods=['POST'])
def query_agents():
    data = request.json
    question = data.get('question')
    logging.info("Received question for agents: %s", question)

    try:
        # Call Agent 1
        logging.info("Starting Agent 1")
        start_time_agent_1 = time.time()
        agent_1_response = call_agent(assistant_id_agent_1, question)
        end_time_agent_1 = time.time()
        logging.info(f"Agent 1 completed in {end_time_agent_1 - start_time_agent_1:.2f} seconds")

        # Call Agent 2 with Agent 1's response
        logging.info("Starting Agent 2")
        start_time_agent_2 = time.time()
        agent_2_response = call_agent(assistant_id_agent_2, agent_1_response)
        end_time_agent_2 = time.time()
        logging.info(f"Agent 2 completed in {end_time_agent_2 - start_time_agent_2:.2f} seconds")

        # Call Agent 3 with Agent 2's response
        logging.info("Starting Agent 3")
        start_time_agent_3 = time.time()
        agent_3_response = call_agent(assistant_id_agent_3, agent_2_response)
        end_time_agent_3 = time.time()
        logging.info(f"Agent 3 completed in {end_time_agent_3 - start_time_agent_3:.2f} seconds")

        # Combine responses from Agent 2 and Agent 3 and send to Agent 4
        logging.info("Starting Agent 4")
        combined_response = f"Agent 2 Response: {agent_2_response}\nAgent 3 Response: {agent_3_response}"
        start_time_agent_4 = time.time()
        #agent_4_response = call_agent(assistant_id_agent_4, combined_response)
        agent_4_response = "COMING SOON WITH PRD FORMAT"
        end_time_agent_4 = time.time()
        logging.info(f"Agent 4 completed in {end_time_agent_4 - start_time_agent_4:.2f} seconds")

        return jsonify({
            'agent_1': agent_1_response,
            'agent_2': agent_2_response,
            'agent_3': agent_3_response,
            'agent_4': agent_4_response
        })
    except Exception as e:
        logging.error("Error during agent processing: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/agenticAI')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's port or default
    app.run(host="0.0.0.0", port=port)