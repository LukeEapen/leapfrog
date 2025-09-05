from flask import Blueprint, jsonify, request, render_template, send_file
import os

# Single, consolidated blueprint
vector_db_api = Blueprint('vector_db_api', __name__)

# Root for uploaded PRDs (kept under repo workspace by default)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'vector_db', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Optional static list extension (kept empty by default)
PRD_LIST = []

@vector_db_api.route('/api/vector-db-prds', methods=['GET'])
def get_vector_db_prds():
    """
    Returns a list of PRDs available in the uploads folder as name/url pairs.
    URL is a full absolute link pointing to the Flask-served file endpoint.
    """
    try:
        files = []
        if os.path.isdir(UPLOAD_FOLDER):
            files = [
                f for f in os.listdir(UPLOAD_FOLDER)
                if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))
            ]

        base = request.host_url.rstrip('/')
        prds = [
            {
                "name": fname,
                "url": f"{base}/vector-db/files/{fname}"
            }
            for fname in files
        ]

        # Backward-compat: include any statically configured PRDs if they look like absolute paths under our host
        for p in PRD_LIST:
            if isinstance(p, dict) and p.get('url', '').startswith(base):
                prds.append(p)

        return jsonify({"prds": prds})
    except Exception as e:
        return jsonify({"prds": [], "error": str(e)}), 500

@vector_db_api.route('/vector-db/viewer', methods=['GET'])
def vector_db_viewer():
    """Render a simple searchable Vector DB viewer page."""
    return render_template('vector_db_viewer.html')

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
    return send_file(file_path, as_attachment=True)

# Optional: list files endpoint returning names and absolute URLs
@vector_db_api.route('/vector-db/files/', methods=['GET'])
def list_vector_db_files():
    try:
        base = request.host_url.rstrip('/')
        files = [
            {
                'name': f,
                'url': f"{base}/vector-db/files/{f}"
            }
            for f in os.listdir(UPLOAD_FOLDER)
            if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))
        ]
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'files': [], 'error': str(e)}), 500
