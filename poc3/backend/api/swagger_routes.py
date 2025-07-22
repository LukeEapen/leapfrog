from flask import Blueprint, request, jsonify
import os
import json

swagger_bp = Blueprint('swagger', __name__)

SWAGGER_PATH = os.path.join(os.path.dirname(__file__), '..', 'api', 'swagger.json')

def load_swagger():
    if not os.path.exists(SWAGGER_PATH):
        return {}
    with open(SWAGGER_PATH, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def save_swagger(swagger):
    with open(SWAGGER_PATH, 'w', encoding='utf-8') as f:
        json.dump(swagger, f, indent=2)

@swagger_bp.route('/api/add-business-logic-to-swagger', methods=['POST'])
def add_business_logic_to_swagger():
    data = request.get_json()
    business_logic = data.get('businessLogic')
    pseudocode = data.get('pseudocode')
    # Load current swagger
    swagger = load_swagger()
    # Add to a custom extension field (x-business-logic)
    if 'x-business-logic' not in swagger:
        swagger['x-business-logic'] = []
    swagger['x-business-logic'].append({
        'businessLogic': business_logic,
        'pseudocode': pseudocode
    })
    save_swagger(swagger)
    return jsonify({'success': True, 'message': 'Business logic and pseudocode added to Swagger.'})
