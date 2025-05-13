from flask import Flask, request, jsonify, render_template
import asyncio
import logging
import sys
from openai import AsyncOpenAI
from langgraph.graph import StateGraph, END
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_responses.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Flask setup
app = Flask(__name__)

# Load environment variables
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Assistant IDs with descriptions
assistant_ids = {
    "agent_1": ("asst_EvIwemZYiG4cCmYc7GnTZoQZ", "Prompt Structuring Agent"),
    "agent_2": ("asst_EkihtJQe9qFiztRdRXPhiy2G", "Requirements Generator"),
    "agent_3": ("asst_Si7JAfL2Ov80wvcly6GKLJcN", "Validator Agent"),
    "agent_5": ("asst_SBfAJLv7rEmYfpaiZNeo4M4R", "Legal & Compliance Analyst"),
    "agent_6": ("asst_mhK4I0m573exEx1bEeU7R5rO", "NFR Specialist"),
    "agent_7": ("asst_kPNCCx49PI7pVaTT7dQR1196", "Platform Architect"),
    "agent_8": ("asst_HrrIeoEnSklSIB04MIWXFoCy", "Legacy Integration Lead")
}

async def query_openai_assistant(assistant_id: str, input_text: str) -> str:
    """Query OpenAI assistant with timeout handling"""
    try:
        thread = await client.beta.threads.create()
        await client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=input_text
        )
        
        run = await client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        timeout = 30  # 30 seconds timeout
        start_time = asyncio.get_event_loop().time()

        while True:
            if asyncio.get_event_loop().time() - start_time > timeout:
                return "Timeout: Assistant took too long to respond"

            run = await client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            
            if run.status == "completed":
                messages = await client.beta.threads.messages.list(thread_id=thread.id)
                return messages.data[0].content[0].text.value
            elif run.status in ("failed", "cancelled"):
                return f"Run failed: {run.status}"
                
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Error in assistant {assistant_id}: {str(e)}")
        return f"Error: {str(e)}"

async def _invoke(agent_key: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke an agent with the given state"""
    payload = state["question"] if isinstance(state, dict) else state
    content = await query_openai_assistant(assistant_ids[agent_key][0], payload)
    return {**state, agent_key: content}

async def agent3_condition(state: Dict[str, Any]) -> str:
    """Determine if processing should continue after agent 3"""
    if asyncio.iscoroutine(state):
        state = await state
    return "approved" if state.get("user_approved") else END

def build_graph() -> StateGraph:
    """Build the agent processing graph"""
    builder = StateGraph(state_schema=Dict[str, Any])
    
    # Add nodes
    for agent_key in ["agent_1", "agent_2", "agent_3"]:
        builder.add_node(agent_key, lambda x, key=agent_key: asyncio.run(_invoke(key, x)))
    
    # Add parallel processing nodes
    for agent_key in ["agent_5", "agent_6", "agent_7", "agent_8"]:
        builder.add_node(agent_key, lambda x, key=agent_key: asyncio.run(_invoke(key, x)))

    # Define graph structure
    builder.set_entry_point("agent_1")
    builder.add_edge("agent_1", "agent_2")
    builder.add_edge("agent_2", "agent_3")
    
    # Add parallel processing paths
    for agent_key in ["agent_5", "agent_6", "agent_7", "agent_8"]:
        builder.add_edge("agent_3", agent_key)
        builder.add_edge(agent_key, END)

    return builder.compile()

@app.route("/agenticAI")
def serve_index():
    """Serve the main UI"""
    return render_template("index-human.html")

@app.route("/api/query_agents", methods=["POST"])
async def query_agents():
    """Handle agent queries with proper error handling"""
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "No question provided"}), 400
            
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Empty question provided"}), 400
            
        initial_state = {"question": question, "user_approved": True}
        
        graph = build_graph()
        result = await graph.ainvoke(initial_state)
        
        # Format agent 4's combined output
        agent_4_output = "\n\n".join([
            f"**{assistant_ids[f'agent_{i}'][1]}:**\n{result.get(f'agent_{i}', 'No response')}"
            for i in [5, 6, 7, 8]
        ])
        
        return jsonify({
            "agent_1": result.get("agent_1", "No response"),
            "agent_2": result.get("agent_2", "No response"),
            "agent_3": result.get("agent_3", "No response"),
            "agent_4": agent_4_output
        })
        
    except Exception as e:
        logger.error(f"Error in query_agents: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    app.run(host="0.0.0.0", port=port, debug=True)