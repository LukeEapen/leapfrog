# use_mcp_agent.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
import os

app = FastAPI()

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class CallLLMAgentParams(BaseModel):
    agent_id: str
    input: str

@app.post("/mcp")
async def mcp_handler(request: Request):
    body = await request.json()
    method = body.get("method")
    params = body.get("params")
    
    if method == "call_llm_agent":
        agent_id = params.get("agent_id")
        user_input = params.get("input")
        prompt_file_path = f"{agent_id}"
        result = await run_chat_agent(prompt_file_path, user_input)
        return {
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": result
        }

    return {"jsonrpc": "2.0", "id": body["id"], "error": "Unknown method"}


async def run_chat_agent(prompt_file_path, user_input, temperature=0.2, top_p=1.0, max_tokens=1000):
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )

    return response.choices[0].message.content
