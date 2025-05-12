from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import asyncio
from typing import TypedDict, Dict, Optional, Any
import logging

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent_interactions.log')
    ]
)
logger = logging.getLogger(__name__)

# Define state schema with output keys
class GraphState(TypedDict):
    input: str
    output_1: Optional[str]
    output_2: Optional[str]
    output_3: Optional[str]
    output_5: Optional[str]
    output_6: Optional[str]
    output_7: Optional[str]
    output_8: Optional[str]

# Load environment and setup
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=openai_api_key)
app = Flask(__name__)

# Assistant IDs mapping
assistant_ids = {
    "agent_1": "asst_EvIwemZYiG4cCmYc7GnTZoQZ",  # Prompt Structuring Agent
    "agent_2": "asst_EkihtJQe9qFiztRdRXPhiy2G",  # Requirements Generator
    "agent_3": "asst_Si7JAfL2Ov80wvcly6GKLJcN",  # Validator Agent
    "agent_5": "asst_SBfAJLv7rEmYfpaiZNeo4M4R",  # Regulatory Analyst
    "agent_6": "asst_mhK4I0m573exEx1bEeU7R5rO",  # NFR Specialist
    "agent_7": "asst_kPNCCx49PI7pVaTT7dQR1196",  # Platform Architect
    "agent_8": "asst_HrrIeoEnSklSIB04MIWXFoCy"   # Legacy Integration Lead
}

async def query_openai_assistant(assistant_id: str, user_input: str) -> str:
    """Query OpenAI Assistant with detailed logging."""
    try:
        if not user_input:
            logger.warning(f"Empty input provided to assistant {assistant_id}")
            return "Error: No input provided"

        logger.info(f"Creating thread for assistant {assistant_id}")
        thread = await client.beta.threads.create()
        
        logger.info(f"Sending message to thread {thread.id}")
        await client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=str(user_input)
        )
        
        logger.info(f"Starting run with assistant {assistant_id}")
        run = await client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant_id
        )
        
        messages = await client.beta.threads.messages.list(thread_id=thread.id)
        if not messages.data:
            logger.warning(f"No response received from assistant {assistant_id}")
            return "No response received"
            
        response = str(messages.data[0].content[0].text.value)
        logger.info(f"Response received from assistant {assistant_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in query_openai_assistant: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

def assistant_node_factory(agent_key: str, include_input: bool = False):
    """Create a LangGraph node for OpenAI Assistant interaction."""
    output_key = f"output_{agent_key[-1]}"
    
    async def _invoke(state: Dict[str, Any]) -> Dict[str, str]:
        try:
            logger.info(f"Processing node {agent_key}")
            user_input = state.get("input", "")
            
            if not user_input and not include_input:
                prev_num = int(agent_key[-1]) - 1
                prev_key = f"output_{prev_num}"
                user_input = state.get(prev_key, "")
                logger.info(f"Using previous output from {prev_key}")
            
            logger.info(f"Input for {agent_key}: {user_input[:200]}...")
            content = await query_openai_assistant(assistant_ids[agent_key], str(user_input))
            logger.info(f"Output from {agent_key}: {content[:200]}...")
            
            output = {output_key: content}
            if include_input:
                output["input"] = user_input
            return output
            
        except Exception as e:
            logger.error(f"Error in node {agent_key}: {str(e)}", exc_info=True)
            return {output_key: f"Error in processing: {str(e)}"}
    
    return RunnableLambda(_invoke)

# Initialize graph with type hints
graph = StateGraph(GraphState)

# Add nodes
for agent_key in assistant_ids.keys():
    graph.add_node(
        agent_key,
        assistant_node_factory(
            agent_key,
            include_input=(agent_key == "agent_1")
        )
    )

# Define graph structure
graph.set_entry_point("agent_1")
graph.add_edge("agent_1", "agent_2")
graph.add_edge("agent_2", "agent_3")

# Parallel processing branches
for agent in ["agent_5", "agent_6", "agent_7", "agent_8"]:
    graph.add_edge("agent_3", agent)
    graph.add_edge(agent, END)

compiled_graph = graph.compile()

@app.route("/api/query_agents", methods=["POST"])
def query_agents():
    """Handle agent queries with comprehensive logging."""
    try:
        logger.info("Received new query request")
        data = request.get_json()
        if not data or "question" not in data:
            logger.warning("No question provided in request")
            return jsonify({"error": "No question provided"}), 400
            
        question = data.get("question", "").strip()
        if not question:
            logger.warning("Empty question provided")
            return jsonify({"error": "Empty question provided"}), 400
            
        logger.info(f"Processing question: {question[:200]}...")
        initial_state = {"input": question}
        result = asyncio.run(compiled_graph.ainvoke(initial_state))
        
        # Format agent 4's combined output
        agent_4_output = "\n\n".join([
            f"Legal & Compliance:\n{result.get('output_5', 'No response')}",
            f"Non-Functional Requirements:\n{result.get('output_6', 'No response')}",
            f"Architecture:\n{result.get('output_7', 'No response')}",
            f"Legacy Integration:\n{result.get('output_8', 'No response')}"
        ])
        
        response = {
            "agent_1": result.get("output_1", "No response"),
            "agent_2": result.get("output_2", "No response"),
            "agent_3": result.get("output_3", "No response"),
            "agent_4": agent_4_output
        }
        
        logger.info("Successfully processed query")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in query_agents: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/agenticAI")
def index():
    """Render the main UI template."""
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)