from flask import Blueprint, jsonify, request
import os

vector_db_api = Blueprint('vector_db_api', __name__)

# Dummy PRD list for demonstration. Replace with actual Vector DB query.
PRD_LIST = [
    {"name": "PRD Example 1", "url": "https://vectordb.example.com/prd1"},
    {"name": "PRD Example 2", "url": "https://vectordb.example.com/prd2"}
]

@vector_db_api.route('/api/vector-db-prds', methods=['GET'])
def get_vector_db_prds():
    # TODO: Replace with actual Vector DB query logic
    return jsonify({"prds": PRD_LIST})

@vector_db_api.route('/api/upload-to-vector-db', methods=['POST'])
def upload_to_vector_db():
    # TODO: Implement actual upload logic
    prd_file = request.files.get('prd_file')
    if not prd_file:
        return jsonify({"success": False, "error": "No PRD file provided."}), 400
    # Simulate upload and return a dummy link
    uploaded_url = f"https://vectordb.example.com/{prd_file.filename}"
    return jsonify({"success": True, "url": uploaded_url})
