from flask import Flask, request, jsonify, render_template
import asyncio
import logging
import sys
from openai import AsyncOpenAI
from langgraph.graph import StateGraph, END
from typing import Dict, Any, Annotated
from dotenv import load_dotenv
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_responses.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

assistant_ids = {
    "agent_1": ("asst_EvIwemZYiG4cCmYc7GnTZoQZ", "Prompt Structuring Agent"),
    "agent_2": ("asst_EkihtJQe9qFiztRdRXPhiy2G", "Requirements Generator"),
    "agent_3": ("asst_Si7JAfL2Ov80wvcly6GKLJcN", "Validator Agent"),
    "agent_5": ("asst_r29PjUzVwfd6XiiYH3ueV41P", "Legal & Compliance Analyst"),
    "agent_6": ("asst_WG96Jp4VMrLJfUE4RMkGFMjf", "NFR Specialist"),
    "agent_7": ("asst_VWzoRLWbeZJS8I2IwtOcsLMp", "Platform Architect"),
    "agent_8": ("asst_sufR6Spw8EBDDoAzqQJN9iJt", "Requirement Functional - Legacy Parity Agent")
}

async def query_openai_assistant(assistant_id: str, input_text: str) -> str:
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

        timeout = 30
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
    # Determine input for each agent
    if agent_key == "agent_1":
        input_text = state.get("question", "")
    elif agent_key == "agent_2":
        input_text = state["responses"].get("agent_1", "")
    elif agent_key == "agent_3":
        input_text = state["responses"].get("agent_2", "")
    elif agent_key in ["agent_5", "agent_6", "agent_7", "agent_8"]:
        input_text = state["responses"].get("agent_3", "")
    else:
        input_text = state.get("question", "")

    if not input_text.strip():
        content = "No input provided to this agent."
    else:
        content = await query_openai_assistant(assistant_ids[agent_key][0], input_text)

    new_responses = dict(state.get("responses", {}))
    new_responses[agent_key] = content
    return {
        "question": state.get("question", ""),
        "responses": new_responses
    }

def build_graph() -> StateGraph:
    builder = StateGraph(
        state_schema={
            "question": str,
            "responses": Annotated[dict, "accumulate"]
        }
    )

    async def agent1(x): return await _invoke("agent_1", x)
    async def agent2(x): return await _invoke("agent_2", x)
    async def agent3(x): return await _invoke("agent_3", x)
    async def agent5(x): return await _invoke("agent_5", x)
    async def agent6(x): return await _invoke("agent_6", x)
    async def agent7(x): return await _invoke("agent_7", x)
    async def agent8(x): return await _invoke("agent_8", x)

    builder.add_node("agent_1", agent1)
    builder.add_node("agent_2", agent2)
    builder.add_node("agent_3", agent3)
    builder.add_node("agent_5", agent5)
    builder.add_node("agent_6", agent6)
    builder.add_node("agent_7", agent7)
    builder.add_node("agent_8", agent8)

    builder.set_entry_point("agent_1")
    builder.add_edge("agent_1", "agent_2")
    builder.add_edge("agent_2", "agent_3")
    for key in ["agent_5", "agent_6", "agent_7", "agent_8"]:
        builder.add_edge("agent_3", key)
        builder.add_edge(key, END)

    return builder.compile()

@app.route("/agenticAI")
def serve_index():
    return render_template("index-human.html")

@app.route("/api/query_agents", methods=["POST"])
async def query_agents():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Empty question provided"}), 400

        initial_state = {"question": question, "responses": {}}
        graph = build_graph()
        result = await graph.ainvoke(initial_state)
        responses = result["responses"]

        agent_4_output = "\n\n".join([
            f"**{assistant_ids[f'agent_{i}'][1]}:**\n{responses.get(f'agent_{i}', 'No response')}"
            for i in [5, 6, 7, 8]
        ])

        return jsonify({
            "agent_1": responses.get("agent_1", "No response"),
            "agent_2": responses.get("agent_2", "No response"),
            "agent_3": responses.get("agent_3", "No response"),
            "agent_4": agent_4_output
        })

    except Exception as e:
        logger.error(f"Error in query_agents: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    app.run(host="0.0.0.0", port=port, debug=True)