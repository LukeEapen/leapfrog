#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting MCP..."
python -m uvicorn use_mcp_agent:app --host 0.0.0.0 --port 4000

echo "Starting app..."
python new-prd-workflow-file-v2.py