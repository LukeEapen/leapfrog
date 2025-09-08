# app.py - Flask Microservice Entry Point
from flask import Flask, request, send_from_directory, jsonify, Response
import os, datetime
from flask_cors import CORS

# Import your blueprints/routes here
# from routes.user_routes import user_bp
# from models import db

app = Flask(__name__)
CORS(app)

# Configurations (update as needed)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'  # Or your DB URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions (uncomment if using SQLAlchemy, etc.)
# db.init_app(app)

# Register blueprints/routes
# app.register_blueprint(user_bp)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    # Serve the legacy code parser agent UI directly (disable caching to avoid stale inline JS)
    resp = send_from_directory(os.path.join(BASE_DIR, 'frontend'), 'legacy_code_parser_agent.html')
    try:
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
    except Exception:
        pass
    return resp

CHAT_HISTORY = []  # simple in-memory store

@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({'error':'Content-Type must be application/json'}), 415
    data = request.get_json(silent=True) or {}
    user_msg = (data or {}).get('message','').strip()
    if not user_msg:
        return jsonify({'error':'Empty message'}), 400
    # Placeholder AI response – echo with timestamp
    response_text = f"Echo: {user_msg} ({datetime.datetime.utcnow().isoformat()}Z)"
    CHAT_HISTORY.append({'role':'user','content':user_msg})
    CHAT_HISTORY.append({'role':'assistant','content':response_text})
    return jsonify({'response':response_text,'history':CHAT_HISTORY[-12:]})

@app.route('/apply-business-breakdown', methods=['POST'])
def apply_business_breakdown():
    if not request.is_json:
        return jsonify({'error':'Content-Type must be application/json'}), 415
    data = request.get_json(silent=True) or {}
    new_html = (data or {}).get('html','').strip()
    if not new_html:
        return jsonify({'error':'No html provided'}), 400
    # For now, just acknowledge (client will replace Section 2 content)
    return jsonify({'status':'ok','applied': True})

@app.route('/frontend/<path:filename>')
def frontend_assets(filename):
    return send_from_directory(os.path.join(BASE_DIR,'frontend'), filename)

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/chat'):
        return jsonify({'error':'/chat not found'}), 404
    return e

@app.errorhandler(405)
def method_not_allowed(e):
    if request.path.startswith('/chat'):
        return jsonify({'error':'Method not allowed; use POST'}), 405
    return e

# Add error handlers, CLI commands, etc. as needed

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
