
# ...existing code...

# Place this route after vector_db_api is defined

# ...existing code...

# Place this route after vector_db_api is defined

from flask import Blueprint, jsonify, request
import os

vector_db_api = Blueprint('vector_db_api', __name__)

# Dummy PRD list for demonstration. Replace with actual Vector DB query.
PRD_LIST = [
    {"name": "PRD Example 1", "url": "https://vectordb.example.com/prd1"},
    {"name": "PRD Example 2", "url": "https://vectordb.example.com/prd2"}
]

@vector_db_api.route('/vector-db/files/', methods=['GET'])
def list_vector_db_files():
    # List all files in the uploads folder
    try:
        files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"files": [], "error": str(e)}), 500
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


UPLOAD_FOLDER = os.path.join(os.getcwd(), 'vector_db', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@vector_db_api.route('/api/upload-to-vector-db', methods=['POST'])
def upload_to_vector_db():
    prd_file = request.files.get('prd_file')
    if not prd_file:
        return jsonify({"success": False, "error": "No PRD file provided."}), 400
    # Save the file locally
    save_path = os.path.join(UPLOAD_FOLDER, prd_file.filename)
    prd_file.save(save_path)
    # Return a link to the file served by Flask
    file_url = f"/vector-db/files/{prd_file.filename}"
    full_url = request.host_url.rstrip('/') + file_url
    return jsonify({"success": True, "url": full_url})

@vector_db_api.route('/vector-db/files/<filename>', methods=['GET'])
def serve_prd_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"success": False, "error": "File not found."}), 404
    from flask import send_file
    return send_file(file_path, as_attachment=True)
