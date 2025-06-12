#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting MCP..."
uvicorn use_mcp_agent:app --port 4000

echo "Starting app..."
python new-prd-workflow-file-v2.py