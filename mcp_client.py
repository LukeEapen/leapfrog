# mcp_client.py
import requests, uuid

def call_agent_via_mcp(agent_id: str, input_text: str) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "call_llm_agent",
        "params": {
            "agent_id": agent_id,
            "input": input_text
        }
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post("http://localhost:4000/mcp", json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    if "error" in result:
        raise RuntimeError(f"MCP error: {result['error']}")
    return result["result"]
