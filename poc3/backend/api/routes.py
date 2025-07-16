from flask import Blueprint, jsonify

api_blueprint = Blueprint('api', __name__)

# Mock endpoints for each agent in the architecture
@api_blueprint.route('/legacy-code-parser', methods=['GET'])
def legacy_code_parser():
    return jsonify({"agent": "Legacy Code Parser Agent", "status": "ok", "data": {"functions": ["funcA", "funcB"]}})

@api_blueprint.route('/user-story-decomposer', methods=['GET'])
def user_story_decomposer():
    return jsonify({"agent": "User Story Decomposer", "status": "ok", "data": {"stories": ["story1", "story2"]}})

@api_blueprint.route('/function-synthesizer', methods=['GET'])
def function_synthesizer():
    return jsonify({"agent": "Function Synthesizer Agent", "status": "ok", "data": {"functions": ["synthFunc1"]}})

@api_blueprint.route('/design-decomposer', methods=['GET'])
def design_decomposer():
    return jsonify({"agent": "Design Decomposer Agent", "status": "ok", "data": {"designs": ["design1"]}})

@api_blueprint.route('/review-prompt', methods=['GET'])
def review_prompt():
    return jsonify({"agent": "Review Prompt Agent", "status": "ok", "data": {"review": "Looks good!"}})

@api_blueprint.route('/service-builder', methods=['GET'])
def service_builder():
    return jsonify({"agent": "Service Builder Agent", "status": "ok", "data": {"service": "ServiceX"}})

@api_blueprint.route('/modernization-validator', methods=['GET'])
def modernization_validator():
    return jsonify({"agent": "Modernization Validator Agent", "status": "ok", "data": {"validation": "Passed"}})

# Approval and chat endpoints (mock)
@api_blueprint.route('/approval', methods=['POST'])
def approval():
    return jsonify({"status": "approved"})

@api_blueprint.route('/chat', methods=['POST'])
def chat():
    return jsonify({"reply": "This is a mock chat response."})
