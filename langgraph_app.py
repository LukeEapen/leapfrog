from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import asyncio
from typing import TypedDict, Dict, Optional

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Define the state schema with output keys
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
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, openai_api_key=openai_api_key)
app = Flask(__name__)

def agent_node_factory(instruction: str, key: str, include_input: bool = False):
    """Creates an agent node with proper error handling and type safety."""
    output_key = f"output_{key[-1]}"  # Convert agent_1 to output_1
    async def _invoke(state: Dict) -> Dict:
        try:
            user_input = state.get("input", "")
            response = await llm.ainvoke([
                HumanMessage(content=f"{instruction}\n\n{user_input}")
            ])
            output = {output_key: str(response.content)}
            if include_input:
                output["input"] = user_input
            return output
        except Exception as e:
            print(f"Error in agent {key}: {str(e)}")
            return {output_key: f"Error processing request: {str(e)}"}
    return RunnableLambda(_invoke)

# Agent instructions
instructions = {
    "agent_1": "Convert vague input into structured directive prompts.",
    "agent_2": "Generate exhaustive, testable system requirements.",
    "agent_3": "Validate requirements for completeness, clarity, and auditability.",
    "agent_5": "Analyze LRC implications in generated requirements.",
    "agent_6": "Add non-functional requirements based on financial standards.",
    "agent_7": "Ensure architectural alignment with modernization goals.",
    "agent_8": "Assess legacy integration dependencies and gaps."
}

# Initialize graph with type hints
graph = StateGraph(GraphState)

# Add nodes
for agent_key, instruction in instructions.items():
    graph.add_node(
        agent_key,
        agent_node_factory(
            instruction,
            agent_key,
            include_input=(agent_key == "agent_1")
        )
    )

# Define sequential flow
graph.set_entry_point("agent_1")
graph.add_edge("agent_1", "agent_2")
graph.add_edge("agent_2", "agent_3")

# Define parallel processing branches
for agent in ["agent_5", "agent_6", "agent_7", "agent_8"]:
    graph.add_edge("agent_3", agent)
    graph.add_edge(agent, END)

# Compile graph
compiled_graph = graph.compile()

@app.route("/api/query_agents", methods=["POST"])
async def query_agents():
    """Handle agent queries with proper error handling."""
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "No question provided"}), 400

        question = data["question"]
        initial_state = {"input": question}
        
        result = await compiled_graph.ainvoke(initial_state)
        
        # Combine parallel outputs for agent_4
        agent_4_output = "\n\n".join([
            f"Legal & Compliance: {result.get('output_5', 'No response')}",
            f"Non-Functional Requirements: {result.get('output_6', 'No response')}",
            f"Architecture: {result.get('output_7', 'No response')}",
            f"Legacy Integration: {result.get('output_8', 'No response')}"
        ])
        
        return jsonify({
            "agent_1": result.get("output_1", "No response"),
            "agent_2": result.get("output_2", "No response"),
            "agent_3": result.get("output_3", "No response"),
            "agent_4": agent_4_output
        })

    except Exception as e:
        print(f"Error in query_agents: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/agenticAI")
def index():
    """Render the main UI template."""
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    app.run(host="0.0.0.0", port=port, debug=True)