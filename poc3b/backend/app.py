
import traceback
import os
import openai
import time
import logging
import traceback
import json
from dotenv import load_dotenv
from flask import Flask, send_from_directory, redirect, url_for, request, jsonify
from api.swagger_routes import swagger_bp

from flask import Flask, request, jsonify, send_from_directory
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
app.register_blueprint(swagger_bp)



# --- Refactored MSBuilder Orchestration ---
import concurrent.futures

# Serve swagger.json for frontend
@app.route('/api/swagger.json', methods=['GET'])
def serve_swagger_json():
    swagger_path = os.path.join(os.path.dirname(__file__), 'api', 'swagger.json')
    if not os.path.exists(swagger_path):
        return jsonify({'error': 'Swagger file not found'}), 404
    with open(swagger_path, 'r', encoding='utf-8') as f:
        try:
            return jsonify(json.load(f))
        except Exception as e:
            return jsonify({'error': str(e)}), 500
# --- Multi-Agent Orchestration for Complete Microservice Project ---
from flask import jsonify, request

# Ensure component assets (Help widget, etc.) are served even if default static routing fails
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

@app.route('/static/components/<path:filename>')
def serve_static_components(filename):
    components_dir = os.path.join(FRONTEND_DIR, 'components')
    return send_from_directory(components_dir, filename)

@app.route('/components/<path:filename>')
def serve_components(filename):
    components_dir = os.path.join(FRONTEND_DIR, 'components')
    return send_from_directory(components_dir, filename)

def generate_complete_microservice_project(project_structure, context):
    files = {}
    swagger = context.get('swagger', {})
    business_logic_map = context.get('business_logic_map', {})  # Map of file_path to business logic
    pseudocode_map = context.get('pseudocode_map', {})  # Map of file_path to pseudocode

    def get_relevant_swagger(file_path):
        # Extract only relevant part of swagger for this file (endpoint/model)
        # For demo, return full swagger; in production, filter by file_path
        return swagger

    def build_chunked_prompt(file_path):
        relevant_swagger = get_relevant_swagger(file_path)
        business_logic = business_logic_map.get(file_path, '')
        pseudocode = pseudocode_map.get(file_path, '')
        system_instructions = "You are generating a production-ready microservice file. Inject all business logic and rules. Include pseudocode as comments at the top. Do not leave any stubs or placeholders."
        prompt = f"""
# Pseudocode for {file_path}:
{pseudocode}

# Actual implementation:
{system_instructions}

Swagger Spec:
{json.dumps(relevant_swagger, indent=2)}

Business Logic:
{business_logic}

Implement all endpoints, models, and business logic relevant to this file. Output only the code for this file.
"""
        return prompt

    def generate_file_code(file_path):
        prompt = build_chunked_prompt(file_path)
        try:
            chat_response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a senior Python Flask microservice developer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4096
            )
            code = chat_response.choices[0].message.content.strip()
            if code.startswith('```'):
                code = code.split('\n', 1)[-1]
                if code.endswith('```'):
                    code = code[:-3]
            return code
        except Exception as e:
            logging.error(f"OpenAI API error for file {file_path}: {str(e)}\n{traceback.format_exc()}")
            return ''

    def is_incomplete(code):
        if not code or code.strip() == '':
            return True
        stub_keywords = [
            'pass', '...', 'raise NotImplementedError', 'TODO', 'to be implemented', 'stub', 'placeholder', 'return None'
        ]
        code_lower = code.lower()
        for kw in stub_keywords:
            if kw in code_lower:
                return True
        return False

    def populate_file(tree, parent_path=''):
        for key, value in tree.items():
            path = parent_path + '/' + key if parent_path else key
            if value == 'file':
                code = generate_file_code(path)
                # Validate and retry if incomplete
                if is_incomplete(code):
                    code = generate_file_code(path)  # Retry once
                if is_incomplete(code):
                    code = f"# Auto-generated stub for {path}. Please implement."
                files[path] = code
            elif isinstance(value, dict):
                populate_file(value, path)

    populate_file(project_structure)
    return {
        'files': files,
        'project_structure': project_structure,
        'build_status': 'Complete',
        'message': 'All files populated by agents with business logic and pseudocode.'
    }
    # Remove duplicate/old code block after refactor

# --- API Endpoint to Generate Complete Microservice Project ---
@app.route('/api/generate-complete-microservice', methods=['POST'])
def api_generate_complete_microservice():
    try:
        data = request.get_json(force=True)
        project_structure = data.get('project_structure', {})
        context = data.get('context', {})
        result = generate_complete_microservice_project(project_structure, context)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

def is_stub_or_empty(code):
    # Simple check for stubs or empty code
    if not code or code.strip() == '':
        return True
    stub_keywords = [
        'pass', '...', 'raise NotImplementedError', 'TODO', 'to be implemented', 'stub', 'placeholder', 'return None'
    ]
    code_lower = code.lower()
    for kw in stub_keywords:
        if kw in code_lower:
            return True
    return False

def build_file_prompt(file_path, swagger, business_logic, system_instructions):
    html_hint = "\nIf this is an HTML file (e.g., index.html), generate a simple, modern, visually appealing HTML page that could serve as a landing or status page for the microservice."
    backend_hint = "\nIf this is a backend file (e.g., in routes/, models/, or utils/), ensure it contains all required code for endpoints, models, business logic, and error handling as described in the Swagger and business logic. Do not leave any file empty or as a stub."
    return (
        f"{system_instructions}\n\n"
        f"You are generating the file: {file_path} for a Python Flask microservice.\n"
        f"The Swagger spec is:\n{json.dumps(swagger, indent=2)}\n"
        f"The business logic is:\n{business_logic}\n"
        f"Implement all endpoints, models, and business logic relevant to this file. Do not leave any stubs or placeholders. Output only the code for this file."
        f"{html_hint if file_path.endswith('.html') else ''}"
        f"{backend_hint if (file_path.endswith('.py') and file_path != 'app.py') or '/routes/' in file_path or '/models/' in file_path or '/utils/' in file_path else ''}"
    )

def walk_project_structure(structure, parent_path=''):
    files = []
    for key, value in structure.items():
        path = os.path.join(parent_path, key) if parent_path else key
        if value == 'file':
            files.append(path.replace('\\', '/'))
        elif isinstance(value, dict):
            files.extend(walk_project_structure(value, path))
    return files

def call_llm_for_file(file_path, swagger, business_logic, system_instructions):
    # Chunked/multi-pass generation for large files
    prompt = build_file_prompt(file_path, swagger, business_logic, system_instructions)
    try:
        # First, try normal generation
        chat_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4096
        )
        code = chat_response.choices[0].message.content.strip()
        # Remove code block markers if present
        if code.startswith('```'):
            code = code.split('\n', 1)[-1]
            if code.endswith('```'):
                code = code.rsplit('```', 1)[0]
        # If code is incomplete and file is large, chunk by class/function
        if is_stub_or_empty(code) and file_path.endswith('.py'):
            # Try to extract relevant classes/functions from Swagger
            schemas = swagger.get('components', {}).get('schemas', {})
            paths = swagger.get('paths', {})
            chunks = []
            # For models
            if '/models/' in file_path:
                for model, defn in schemas.items():
                    chunk_prompt = (
                        f"Generate the Python dataclass for model '{model}' as described in the Swagger spec. Include all fields, types, and docstrings."
                        f"\nSwagger schema:\n{json.dumps(defn, indent=2)}"
                    )
                    chunk_code = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_instructions},
                            {"role": "user", "content": chunk_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1024
                    ).choices[0].message.content.strip()
                    chunks.append(chunk_code)
            # For routes
            elif '/routes/' in file_path:
                for path, methods in paths.items():
                    for method, op in methods.items():
                        chunk_prompt = (
                            f"Generate the Flask route for endpoint '{path}' [{method.upper()}] as described in the Swagger spec. Include request/response validation and business logic."
                            f"\nSwagger operation:\n{json.dumps(op, indent=2)}"
                        )
                        chunk_code = openai.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": system_instructions},
                                {"role": "user", "content": chunk_prompt}
                            ],
                            temperature=0.1,
                            max_tokens=1024
                        ).choices[0].message.content.strip()
                        chunks.append(chunk_code)
            # For other backend files, fallback to normal prompt
            if chunks:
                code = '\n\n'.join(chunks)
        return code
    except Exception as e:
        logging.error(f"OpenAI API error for file {file_path}: {str(e)}\n{traceback.format_exc()}")
        return ''

@app.route('/api/msbuilder-generate', methods=['POST'])
def api_msbuilder_generate():
    try:
        data = request.get_json()
        swagger = data.get('swagger', None)
        business_logic = data.get('business_logic', '')  # Accept business logic from previous stage
        if not swagger:
            return jsonify({'error': 'No Swagger document provided'}), 400

        # Load MSBuilder instructions
        instructions_path = os.path.join(os.path.dirname(__file__), 'agents', 'msbuilder_instructions.txt')
        with open(instructions_path, 'r', encoding='utf-8') as f:
            system_instructions = f.read()

        # Step 1: Ask LLM for project structure (single call)
        structure_prompt = (
            f"{system_instructions}\n\nSwagger Document:\n{json.dumps(swagger, indent=2)}\n\n"
            "Generate only the project_structure JSON tree (no code, no files, just the structure as in the example). "
            "In addition to all backend files, include a simple index.html file in the root or static/ folder."
        )
        chat_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": structure_prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        structure_output = chat_response.choices[0].message.content.strip()
        # Remove code block markers if present
        if structure_output.startswith('```'):
            structure_output = structure_output.split('\n', 1)[-1]
            if structure_output.endswith('```'):
                structure_output = structure_output.rsplit('```', 1)[0]
        structure_output = structure_output.strip()
        project_structure = json.loads(structure_output)
        # --- PATCH: Move README.md, app.py, requirements.txt to root if present anywhere ---
        def move_to_root(structure, filenames):
            # Recursively search and move files to root
            to_move = {}
            def recurse(obj, parent=None, key=None):
                if isinstance(obj, dict):
                    for k in list(obj.keys()):
                        if k in filenames:
                            to_move[k] = obj[k]
                            del obj[k]
                        elif isinstance(obj[k], dict):
                            recurse(obj[k], obj, k)
                return obj
            recurse(structure)
            for fname, val in to_move.items():
                structure[fname] = val
            return structure
        project_structure = move_to_root(project_structure, ["README.md", "app.py", "requirements.txt"])
        file_paths = walk_project_structure(project_structure)
        files = {}

        # --- PATCH: Ensure all required backend model/route files are present in project_structure and file_paths ---
        required_models = set()
        required_routes = set()
        # From Swagger schemas
        for schema_name in swagger.get('components', {}).get('schemas', {}):
            required_models.add(schema_name)
        # From Swagger paths
        for path, ops in swagger.get('paths', {}).items():
            base = path.strip('/').split('/')[0]
            if base:
                required_routes.add(base)
                # Heuristic: route name is also a model name
                required_models.add(''.join(word.capitalize() for word in base.split('_')))

        def ensure_file_in_structure(structure, folder, filename):
            if folder not in structure or not isinstance(structure[folder], dict):
                structure[folder] = {}
            structure[folder][filename] = 'file'

        for model in required_models:
            fname = model[0].lower() + ''.join(['_' + c.lower() if c.isupper() else c for c in model[1:]]) + '.py'
            ensure_file_in_structure(project_structure, 'models', fname)
        for route in required_routes:
            ensure_file_in_structure(project_structure, 'routes', f'{route}_routes.py')

        file_paths = walk_project_structure(project_structure)
        file_paths = walk_project_structure(project_structure)
        # Ensure Swagger completeness for all backend files before code generation
        def ensure_swagger_completeness_for_files(file_paths, swagger):
            schemas = swagger.setdefault('components', {}).setdefault('schemas', {})
            paths = swagger.setdefault('paths', {})
            import re
            def snake_to_pascal(s):
                return ''.join(word.capitalize() for word in s.replace('.py', '').split('_'))
            def pascal_to_snake(name):
                return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

            # Add/patch models with more realistic placeholder fields
            for file_path in file_paths:
                if '/models/' in file_path and file_path.endswith('.py'):
                    file_base = file_path.split('/')[-1]
                    model_name = snake_to_pascal(file_base)
                    if model_name not in schemas:
                        # Add placeholder fields for demo; in production, extract from business logic
                        schemas[model_name] = {
                            'type': 'object',
                            'properties': {
                                'id': {'type': 'integer', 'description': 'Auto-generated id'},
                                'name': {'type': 'string', 'description': f'Name of the {model_name}'},
                                'created_at': {'type': 'string', 'format': 'date-time', 'description': 'Creation timestamp'}
                            },
                            'description': f'Auto-generated schema for {model_name}'
                        }

            # Add/patch CRUD endpoints for each route
            for file_path in file_paths:
                if '/routes/' in file_path and file_path.endswith('.py'):
                    base = re.sub(r'_routes.py$', '', file_path.split('/')[-1])
                    model_name = snake_to_pascal(base)
                    path_str = f'/{base}'
                    if path_str not in paths:
                        # Add CRUD endpoints
                        paths[path_str] = {
                            'get': {
                                'summary': f'Get list of {model_name}',
                                'responses': {
                                    '200': {
                                        'description': f'List of {model_name}',
                                        'content': {
                                            'application/json': {
                                                'schema': {
                                                    'type': 'array',
                                                    'items': {'$ref': f'#/components/schemas/{model_name}'}
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            'post': {
                                'summary': f'Create a new {model_name}',
                                'requestBody': {
                                    'required': True,
                                    'content': {
                                        'application/json': {
                                            'schema': {'$ref': f'#/components/schemas/{model_name}'}
                                        }
                                    }
                                },
                                'responses': {
                                    '201': {
                                        'description': f'{model_name} created',
                                        'content': {
                                            'application/json': {
                                                'schema': {'$ref': f'#/components/schemas/{model_name}'}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    # Add detail endpoints (GET/PUT/DELETE by id)
                    detail_path = f'/{base}/{{id}}'
                    if detail_path not in paths:
                        paths[detail_path] = {
                            'get': {
                                'summary': f'Get a {model_name} by id',
                                'parameters': [
                                    {
                                        'name': 'id',
                                        'in': 'path',
                                        'required': True,
                                        'schema': {'type': 'integer'}
                                    }
                                ],
                                'responses': {
                                    '200': {
                                        'description': f'{model_name} details',
                                        'content': {
                                            'application/json': {
                                                'schema': {'$ref': f'#/components/schemas/{model_name}'}
                                            }
                                        }
                                    },
                                    '404': {'description': f'{model_name} not found'}
                                }
                            },
                            'put': {
                                'summary': f'Update a {model_name} by id',
                                'parameters': [
                                    {
                                        'name': 'id',
                                        'in': 'path',
                                        'required': True,
                                        'schema': {'type': 'integer'}
                                    }
                                ],
                                'requestBody': {
                                    'required': True,
                                    'content': {
                                        'application/json': {
                                            'schema': {'$ref': f'#/components/schemas/{model_name}'}
                                        }
                                    }
                                },
                                'responses': {
                                    '200': {
                                        'description': f'{model_name} updated',
                                        'content': {
                                            'application/json': {
                                                'schema': {'$ref': f'#/components/schemas/{model_name}'}
                                            }
                                        }
                                    },
                                    '404': {'description': f'{model_name} not found'}
                                }
                            },
                            'delete': {
                                'summary': f'Delete a {model_name} by id',
                                'parameters': [
                                    {
                                        'name': 'id',
                                        'in': 'path',
                                        'required': True,
                                        'schema': {'type': 'integer'}
                                    }
                                ],
                                'responses': {
                                    '204': {'description': f'{model_name} deleted'},
                                    '404': {'description': f'{model_name} not found'}
                                }
                            }
                        }
            return swagger

        swagger = ensure_swagger_completeness_for_files(file_paths, swagger)
        # Pass system_instructions to all helper functions and thread workers
        def agent_for_file(file_path):
            # Two-pass: models, routes, and other backend files
            if file_path.endswith('.cbl') or file_path.endswith('.COBOL'):
                return 'legacy_code_parser'
            elif file_path.endswith('.xml'):
                return 'msproject'
            elif '/models/' in file_path and file_path.endswith('.py'):
                return 'models_pass'
            elif '/routes/' in file_path and file_path.endswith('.py'):
                return 'routes_pass'
            else:
                return 'msbuilder'

        def call_llm_for_file(agent, file_path, swagger, business_logic, system_instructions):
            # Compose a specialized prompt for each agent
            def try_llm_with_prompt(prompt):
                try:
                    chat_response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_instructions},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=4096
                    )
                    code = chat_response.choices[0].message.content.strip()
                    if code.startswith('```'):
                        code = code.split('\n', 1)[-1]
                        if code.endswith('```'):
                            code = code.rsplit('```', 1)[0]
                    return code
                except Exception as e:
                    logging.error(f"Agent {agent} API error for file {file_path}: {str(e)}\n{traceback.format_exc()}")
                    return ''

            # Aggressive multi-pass prompt escalation for backend files
            if agent == 'legacy_code_parser':
                prompt = (
                    f"{system_instructions}\n\nYou are generating the file: {file_path} for a Python Flask microservice.\n"
                    f"The Swagger spec is:\n{json.dumps(swagger, indent=2)}\n"
                    f"The business logic is:\n{business_logic}\n"
                    f"Translate the legacy COBOL code to Python or document as needed. Output only the code for this file."
                )
                return try_llm_with_prompt(prompt)
            elif agent == 'msproject':
                prompt = (
                    f"{system_instructions}\n\nSwagger Document:\n{json.dumps(swagger, indent=2)}\n\n"
                    "Generate only the msproject.xml file as Gantt-compatible XML, based on the endpoints and logic. Output only the XML."
                )
                return try_llm_with_prompt(prompt)
            elif agent in ('models_pass', 'routes_pass', 'msbuilder'):
                # 1st attempt: normal prompt
                if agent == 'models_pass':
                    prompt = (
                        f"{system_instructions}\n\nYou are generating the file: {file_path} for a Python Flask microservice.\n"
                        f"The Swagger spec is:\n{json.dumps(swagger, indent=2)}\n"
                        f"The business logic is:\n{business_logic}\n"
                        f"Implement all data models, classes, and business logic for this file as described in the Swagger spec. Include all fields, types, validation, and docstrings. Do not leave any stubs or placeholders. Output only the code for this file."
                    )
                elif agent == 'routes_pass':
                    prompt = (
                        f"{system_instructions}\n\nYou are generating the file: {file_path} for a Python Flask microservice.\n"
                        f"The Swagger spec is:\n{json.dumps(swagger, indent=2)}\n"
                        f"The business logic is:\n{business_logic}\n"
                        f"Implement all endpoints, routes, request/response validation, error handling, and business logic for this file as described in the Swagger spec. Do not leave any stubs or placeholders. Output only the code for this file."
                    )
                else:
                    prompt = (
                        f"{system_instructions}\n\nYou are generating the file: {file_path} for a Python Flask microservice.\n"
                        f"The Swagger spec is:\n{json.dumps(swagger, indent=2)}\n"
                        f"The business logic is:\n{business_logic}\n"
                        f"Implement all endpoints, models, business logic, error handling, and tests for this file as described in the Swagger spec. Do not leave any stubs or placeholders. Output only the code for this file."
                    )
                code = try_llm_with_prompt(prompt)
                # 2nd attempt: more aggressive prompt if first fails
                if is_stub_or_empty(code):
                    prompt2 = (
                        f"{system_instructions}\n\nYou are generating the file: {file_path} for a Python Flask microservice.\n"
                        f"The Swagger spec is:\n{json.dumps(swagger, indent=2)}\n"
                        f"The business logic is:\n{business_logic}\n"
                        f"You must generate real, production-ready code for this file. Do not leave any stubs, placeholders, or empty classes/functions. The code must be complete and ready to deploy. Output only the code for this file."
                    )
                    code = try_llm_with_prompt(prompt2)
                # 3rd attempt: even more explicit prompt if still fails
                if is_stub_or_empty(code):
                    prompt3 = (
                        f"{system_instructions}\n\nYou are generating the file: {file_path} for a Python Flask microservice.\n"
                        f"The Swagger spec is:\n{json.dumps(swagger, indent=2)}\n"
                        f"The business logic is:\n{business_logic}\n"
                        f"You must generate real, deployable, non-empty code for this file. If you do not, the build will fail. Do not output any stubs, placeholders, or empty classes/functions. Output only the code for this file."
                    )
                    code = try_llm_with_prompt(prompt3)
                return code
            else:
                prompt = build_file_prompt(file_path, swagger, business_logic, system_instructions)
                return try_llm_with_prompt(prompt)

        def fallback_model_code(file_path, swagger):
            import re
            # Extract model name from file path
            model_name = re.sub(r".py$", "", file_path.split("/")[-1]).title().replace("_", "")
            # Try to find schema in swagger
            schemas = swagger.get("components", {}).get("schemas", {}) if swagger else {}
            fields = schemas.get(model_name, {}).get("properties", {}) if schemas else {}
            field_lines = []
            for fname, fdef in fields.items():
                ftype = fdef.get("type", "str")
                pytype = {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}.get(ftype, "str")
                field_lines.append(f"    {fname}: {pytype}")
            if not field_lines:
                # Always add at least one field so file is never empty
                field_lines = ["    id: int  # placeholder field"]
            return f"""from dataclasses import dataclass\n\n@dataclass\nclass {model_name}:\n" + "\n".join(field_lines) + "\n"""

        def fallback_route_code(file_path, swagger):
            import re
            # Extract base name for blueprint
            base = re.sub(r"_routes.py$", "", file_path.split("/")[-1])
            blueprint_name = base + "_bp"
            route_prefix = f"/{base}"
            # Try to find relevant paths in swagger
            paths = swagger.get("paths", {}) if swagger else {}
            endpoints = [p for p in paths if base in p] if paths else []
            if not endpoints:
                # Always add at least one endpoint so file is never empty
                endpoints = [f"/{base}"]
            route_lines = [f"@{blueprint_name}.route('{ep}', methods=['GET'])\ndef get_{base}():\n    return '{{}}', 200" for ep in endpoints]
            return f"""from flask import Blueprint, request\n\n{blueprint_name} = Blueprint('{base}', __name__)\n\n" + "\n\n".join(route_lines) + "\n"""

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {}
            for file_path in file_paths:
                agent = agent_for_file(file_path)
                future = executor.submit(call_llm_for_file, agent, file_path, swagger, business_logic, system_instructions) if False else executor.submit(call_llm_for_file, file_path, swagger, business_logic, system_instructions)
                future_to_file[future] = (file_path, agent)
            for future in concurrent.futures.as_completed(future_to_file):
                file_path, agent = future_to_file[future]
                code = future.result()
                # Validate code
                if is_stub_or_empty(code):
                    logging.warning(f"File {file_path} generated by {agent} is stub/empty. Retrying once.")
                    code = call_llm_for_file(file_path, swagger, business_logic, system_instructions)
                if is_stub_or_empty(code):
                    # Fallback: generate minimal valid code for backend files
                    if ('/models/' in file_path and file_path.endswith('.py')):
                        logging.error(f"File {file_path} is still stub/empty after retry. Using fallback model template.")
                        code = fallback_model_code(file_path, swagger)
                    elif ('/routes/' in file_path and file_path.endswith('.py')):
                        logging.error(f"File {file_path} is still stub/empty after retry. Using fallback route template.")
                        code = fallback_route_code(file_path, swagger)
                    else:
                        logging.error(f"File {file_path} generated by {agent} is still stub/empty after retry. Marking as error.")
                        code = f"# ERROR: Could not generate {file_path}. Please implement manually."
                files[file_path] = code

        # --- ENFORCEMENT: Ensure all backend files under models/ and routes/ are non-empty and real code ---
        def enforce_backend_file_completeness(files, file_paths, swagger):
            import re
            schemas = swagger.get("components", {}).get("schemas", {}) if swagger else {}
            paths = swagger.get("paths", {}) if swagger else {}
            def snake_to_pascal(s):
                return ''.join(word.capitalize() for word in s.replace('.py', '').split('_'))
            for file_path in file_paths:
                if file_path not in files or is_stub_or_empty(files[file_path]):
                    # For models: generate dataclass with all fields and docstrings from Swagger
                    if ('/models/' in file_path and file_path.endswith('.py')):
                        file_base = file_path.split("/")[-1]
                        model_name = snake_to_pascal(file_base)
                        schema = schemas.get(model_name, {})
                        properties = schema.get("properties", {})
                        required = schema.get("required", [])
                        docstring = schema.get("description", "")
                        field_lines = []
                        for fname, fdef in properties.items():
                            ftype = fdef.get("type", "str")
                            pytype = {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}.get(ftype, "str")
                            desc = fdef.get("description", "")
                            comment = f"  # {desc}" if desc else ""
                            field_lines.append(f"    {fname}: {pytype}{comment}")
                        if not field_lines:
                            field_lines = ["    id: int  # placeholder field"]
                        docstring_block = f'    """{docstring}"""\n' if docstring else ''
                        # Fallback: always produce valid code
                        files[file_path] = (
                            "from dataclasses import dataclass\n\n"
                            "@dataclass\n"
                            f"class {model_name}:\n"
                            f"{docstring_block}"
                            + ("\n".join(field_lines) if field_lines else "    id: int  # placeholder field")
                            + "\n"
                        )
                    # For routes: generate Flask Blueprint with all endpoints and docstrings from Swagger
                    elif ('/routes/' in file_path and file_path.endswith('.py')):
                        base = re.sub(r"_routes.py$", "", file_path.split("/")[-1])
                        blueprint_name = base + "_bp"
                        endpoints = [p for p in paths if base in p] if paths else []
                        route_lines = []
                        for ep in endpoints:
                            methods = paths[ep].keys()
                            for method in methods:
                                op = paths[ep][method]
                                summ = op.get("summary", "")
                                desc = op.get("description", "")
                                docstring = f'"""{summ or desc}"""' if (summ or desc) else ''
                                # Use parameters and responses for more realistic code
                                params = op.get("parameters", [])
                                param_lines = []
                                for param in params:
                                    pname = param.get("name", "param")
                                    ptype = param.get("schema", {}).get("type", "str")
                                    param_lines.append(f"    {pname} = request.args.get('{pname}')  # type: {ptype}")
                                route_lines.append(
                                    f"@{blueprint_name}.route('{ep}', methods=['{method.upper()}'])\ndef {method.lower()}_{base}():\n    {docstring}\n" + ("\n".join(param_lines) + "\n" if param_lines else "") +
                                    f"    # TODO: Implement logic for {ep} [{method.upper()}]\n    return '{{}}', 200"
                                )
                        if not route_lines:
                            route_lines = [f"@{blueprint_name}.route('/{base}', methods=['GET'])\ndef get_{base}():\n    return '{{}}', 200"]
                        files[file_path] = (
                            "from flask import Blueprint, request\n\n"
                            f"{blueprint_name} = Blueprint('{base}', __name__)\n\n"
                            + "\n\n".join(route_lines) + "\n"
                        )
                    else:
                        files[file_path] = f"# ERROR: No code generated for {file_path}. Please check the Swagger and business logic."
            return files

        files = enforce_backend_file_completeness(files, file_paths, swagger)
        # DEBUG: Log the generated code for each backend file
        for file_path in file_paths:
            if ('/models/' in file_path or '/routes/' in file_path) and file_path in files:
                logging.info(f"Generated code for {file_path}:\n{files[file_path][:500]}{'... [truncated]' if len(files[file_path]) > 500 else ''}")

        # Step 4: Generate msproject.xml only if not already present
        if 'msproject.xml' not in files:
            msproject_prompt = (
                f"{system_instructions}\n\nSwagger Document:\n{json.dumps(swagger, indent=2)}\n\n"
                "Generate only the msproject.xml file as Gantt-compatible XML, based on the endpoints and logic. Output only the XML."
            )
            chat_response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": msproject_prompt}
                ],
                temperature=0.1,
                max_tokens=2048
            )
            msproject_xml = chat_response.choices[0].message.content.strip()
            if msproject_xml.startswith('```'):
                msproject_xml = msproject_xml.split('\n', 1)[-1]
                if msproject_xml.endswith('```'):
                    msproject_xml = msproject_xml.rsplit('```', 1)[0]
            msproject_xml = msproject_xml.strip()
            files['msproject.xml'] = msproject_xml
            if 'msproject.xml' not in project_structure:
                # Add to structure if missing
                project_structure['msproject.xml'] = 'file'

        # Step 5: Aggregate and return, with completeness check and warnings
        incomplete_files = [fp for fp, content in files.items() if is_stub_or_empty(content)]
        build_status = 'success' if not incomplete_files else 'warning'
        message = 'Microservice and MS Project plan generated from Swagger and business logic.'
        if incomplete_files:
            message += f' WARNING: The following files are incomplete or stubs and need regeneration: {incomplete_files}'
        return jsonify({
            'project_structure': project_structure,
            'files': files,
            'build_status': build_status,
            'message': message,
            'incomplete_files': incomplete_files
        })
    except Exception as e:
        logging.error(f"Error in msbuilder-generate: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e), 'build_status': 'error', 'message': 'Server error in msbuilder-generate.'}), 500
    
@app.route('/api/service-builder-swagger', methods=['POST'])
def api_service_builder_swagger():
    try:
        data = request.get_json()
        selected_design = data.get('selected_design', '')
        if not selected_design:
            return jsonify({'error': 'No design element provided'}), 400

        # Load system instructions for service builder
        instructions_path = os.path.join(os.path.dirname(__file__), 'agents', 'service_builder_instructions.txt')
        with open(instructions_path, 'r', encoding='utf-8') as f:
            system_instructions = f.read()

        # Compose prompt for GPT-3.5
        prompt = (
            f"{system_instructions}\n\nSelected Design Element:\n{selected_design}\n\n"
            "Generate a fully exhaustive Swagger (OpenAPI 3.0.0) specification for this service, matching the standards and completeness of https://editor.swagger.io/. "
            "Include all business logic, rules, and flows from the functional breakdown, not just endpoints. "
            "Include all required fields: openapi, info, servers, tags, paths (with all methods, parameters, request/response schemas), components (schemas, security), and security. "
            "For each endpoint, provide detailed request and response bodies, parameters, and error responses. "
            "Output only the Swagger JSON object, no markdown or explanation."
        )

        # Call OpenAI GPT-3.5 with temperature 0.2
        chat_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2048
        )
        output = chat_response.choices[0].message.content
        # Try to parse output as JSON
        swagger_obj = None
        try:
            # Remove code block markers if present
            if output.startswith('```'):
                output = output.split('\n', 1)[-1]
                if output.endswith('```'):
                    output = output.rsplit('```', 1)[0]
            swagger_obj = json.loads(output)
        except Exception as e:
            logging.error(f"Error parsing Swagger JSON: {str(e)}\n{output}")
            return jsonify({'error': 'Failed to parse Swagger JSON', 'raw': output}), 500

        # Ensure business logic, models, and metadata are present
        # Business Logic: try to extract from design or add placeholder
        if 'x-business-logic' not in swagger_obj:
            try:
                design_json = json.loads(selected_design)
                swagger_obj['x-business-logic'] = design_json.get('business_logic', 'Not defined')
            except Exception:
                swagger_obj['x-business-logic'] = 'Not defined'
        # Models: ensure schemas exist
        if 'components' not in swagger_obj:
            swagger_obj['components'] = {}
        if 'schemas' not in swagger_obj['components']:
            swagger_obj['components']['schemas'] = {}
        # --- PATCH: Extract models from design input and inject into Swagger ---
        try:
            design_json = json.loads(selected_design)
            if 'models' in design_json and isinstance(design_json['models'], dict):
                swagger_obj['components']['schemas'].update(design_json['models'])
        except Exception:
            pass
        # Always add at least one demo model if no models present (for debugging)
        if not swagger_obj['components']['schemas'] or list(swagger_obj['components']['schemas'].keys()) == ['Error']:
            swagger_obj['components']['schemas']['DemoModel'] = {
                'type': 'object',
                'properties': {
                    'id': {'type': 'integer', 'description': 'Demo id'},
                    'name': {'type': 'string', 'description': 'Demo name'}
                },
                'description': 'Demo model for debugging Swagger output.'
            }
        # Metadata: ensure info exists
        if 'info' not in swagger_obj:
            swagger_obj['info'] = {
                'title': 'Modernized Service',
                'version': '1.0.0',
                'description': 'Auto-generated Swagger spec for MS Project creation.'
            }
        # Servers: ensure at least one server
        if 'servers' not in swagger_obj or not swagger_obj['servers']:
            swagger_obj['servers'] = [{'url': 'https://bain.ai.workbench.com'}]

        return jsonify({'swagger': json.dumps(swagger_obj)})
    except Exception as e:
        logging.error(f"Error in service-builder-swagger: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
# API endpoint to run Design Decomposer Agent
@app.route('/api/design-decompose', methods=['POST'])
def api_design_decompose():
    try:
        data = request.get_json()
        functions = data.get('functions', [])
        if not functions or not isinstance(functions, list):
            return jsonify({'error': 'No functions provided'}), 400

        # Load system instructions for design decomposer
        instructions_path = os.path.join(os.path.dirname(__file__), 'agents', 'design_decomposer_instructions.txt')
        with open(instructions_path, 'r', encoding='utf-8') as f:
            system_instructions = f.read()

        # Compose prompt for GPT-3.5
        prompt = (
            f"{system_instructions}\n\nFunctions:\n"
            + '\n'.join([f"- {fn.get('name', '')}: {fn.get('description', '')}" for fn in functions])
            + "\n\nFor each design element, generate the core business logic as exhaustive, step-by-step pseudocode (not real code), capturing all business rules, flows, and logic from both the user story and legacy code. The pseudocode must be detailed and actionable, not a summary. Output as a JSON array, each object with: 'element', 'standard', and 'pseudocode' fields."
        )

        # Call OpenAI GPT-3.5 with temperature 0.2 using new API
        chat_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2048
        )
        output = chat_response.choices[0].message.content
        # Strip markdown code block markers (e.g. ```json ... ```)
        import json, re
        def strip_code_block_markers(text):
            # Remove triple backticks and optional language
            text = re.sub(r'^```[a-zA-Z0-9]*\s*', '', text.strip())
            text = re.sub(r'```$', '', text.strip())
            return text.strip()

        def autocorrect_json(text):
            # Remove trailing commas before closing brackets/braces
            text = re.sub(r',\s*([}\]])', r'\1', text)
            # Remove double commas
            text = re.sub(r',\s*,', ',', text)
            # Optionally fix missing commas between objects (very basic)
            text = re.sub(r'}\s*{', '}, {', text)
            return text

        output_stripped = strip_code_block_markers(output)

        # Try to extract JSON array from output, fallback to safe default
        design = []
        try:
            # Try to find a JSON array in the output
            match = re.search(r'(\[.*?\])', output_stripped, re.DOTALL)
            if match:
                try:
                    design = json.loads(match.group(1))
                except Exception:
                    # Try autocorrect if initial parse fails
                    design = json.loads(autocorrect_json(match.group(1)))
            else:
                try:
                    parsed = json.loads(output_stripped)
                except Exception:
                    parsed = json.loads(autocorrect_json(output_stripped))
                if isinstance(parsed, list):
                    design = parsed
                elif isinstance(parsed, dict):
                    design = [parsed]
        except Exception as ex:
            logging.warning(f"Could not parse design decomposer LLM output as JSON. Output was: {output_stripped.strip()[:200]}... Error: {ex}")
            # Fallback: try to extract individual objects from the output
            design = []
            # Improved fallback: extract nested JSON objects using a stack
            def extract_json_objects(text):
                stack = []
                objects = []
                start = None
                for i, c in enumerate(text):
                    if c == '{':
                        if not stack:
                            start = i
                        stack.append(c)
                    elif c == '}':
                        if stack:
                            stack.pop()
                            if not stack and start is not None:
                                obj = text[start:i+1]
                                objects.append(obj)
                                start = None
                return objects
            obj_blocks = extract_json_objects(output_stripped)
            for obj_str in obj_blocks:
                try:
                    obj = json.loads(autocorrect_json(obj_str))
                    if isinstance(obj, dict) and obj.get('element') and obj.get('standard'):
                        design.append(obj)
                except Exception:
                    continue

        # Enforce standards and required fields
        allowed_standards = {"openapi": "OpenAPI", "bian": "BIAN", "iso": "ISO"}
        def normalize_standard(val):
            if not val:
                return "OpenAPI"
            val_lower = str(val).strip().lower()
            for key, std in allowed_standards.items():
                if key in val_lower:
                    return std
            return "OpenAPI"

        def pseudocode_to_string(pseudo):
            if isinstance(pseudo, str):
                return pseudo.strip()
            if isinstance(pseudo, list):
                # If it's a list of strings, join them
                if all(isinstance(step, str) for step in pseudo):
                    return '\n'.join(step.strip() for step in pseudo)
                # If it's a list of step objects, format as numbered steps
                steps = []
                for idx, step in enumerate(pseudo, 1):
                    if isinstance(step, dict):
                        desc = step.get('description') or step.get('step') or ''
                        actions = step.get('actions')
                        if actions and isinstance(actions, list):
                            actions_str = '\n    - ' + '\n    - '.join(str(a) for a in actions)
                        else:
                            actions_str = ''
                        steps.append(f"{idx}. {desc}{actions_str}")
                    else:
                        steps.append(f"{idx}. {str(step)}")
                return '\n'.join(steps)
            return str(pseudo)

        normalized = []
        for item in design:
            if not isinstance(item, dict):
                continue
            element = item.get('element', '').strip()
            pseudocode = item.get('pseudocode', '')
            pseudocode_str = pseudocode_to_string(pseudocode)
            standard = normalize_standard(item.get('standard', ''))
            if not element:
                continue
            if not pseudocode_str:
                pseudocode_str = '// No business logic available'
            normalized.append({
                'element': element,
                'standard': standard,
                'pseudocode': pseudocode_str
            })
        return jsonify({'design': normalized, 'raw': output})
    except Exception as e:
        logging.error(f"Error in design-decompose: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ...existing code...
@app.route('/api/test-endpoints', methods=['GET'])
def api_test_endpoints():
    """
    Returns a list of available API endpoints and sample requests for frontend auto-population.
    """
    endpoints = [
        {
            'url': 'http://localhost:5050/api/generate-complete-microservice',
            'method': 'POST',
            'sample_body': json.dumps({
                'project_structure': {},
                'context': {}
            }, indent=2)
        },
        {
            'url': 'http://localhost:5050/api/msbuilder-generate',
            'method': 'POST',
            'sample_body': json.dumps({
                'swagger': {},
                'business_logic': ''
            }, indent=2)
        },
        {
            'url': 'http://localhost:5050/api/swagger.json',
            'method': 'GET',
            'sample_body': ''
        }
    ]
    return jsonify({'endpoints': endpoints})
@app.route('/api/design-element-business-logic', methods=['POST'])
def api_design_element_business_logic():
    try:
        data = request.get_json()
        element = data.get('element', '')
        if not element:
            return jsonify({'error': 'No design element provided'}), 400
        # Load design decomposer instructions
        instructions_path = os.path.join(os.path.dirname(__file__), 'agents', 'design_decomposer_instructions.txt')
        with open(instructions_path, 'r', encoding='utf-8') as f:
            system_instructions = f.read()
        # Compose prompt for GPT-3.5
        prompt = (
            f"{system_instructions}\n\nDesign Element:\n{element}\n\nGenerate a sample perspective showcasing the implementation of core business logic for this design element. The pseudocode must be exhaustive, step-by-step, and capture all business rules, flows, and logic from both the user story and legacy code. The pseudocode must be detailed and actionable, not a summary. Output as a JSON object with: 'element', 'standard', and 'pseudocode'."
        )
        chat_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        output = chat_response.choices[0].message.content
        import json, re
        def strip_code_block_markers(text):
            text = re.sub(r'^```[a-zA-Z0-9]*\s*', '', text.strip())
            text = re.sub(r'```$', '', text.strip())
            return text.strip()
        output_stripped = strip_code_block_markers(output)
        try:
            obj = json.loads(output_stripped)
        except Exception:
            obj = {'element': element, 'standard': 'OpenAPI', 'pseudocode': output_stripped}
        return jsonify(obj)
    except Exception as e:
        logging.error(f"Error in design-element-business-logic: {str(e)}")
        return jsonify({'error': str(e)}), 500

# API endpoint to synthesize functions from user story and legacy English
@app.route('/api/synthesize-functions', methods=['POST'])
def api_synthesize_functions():
    try:
        data = request.get_json()
        user_story = data.get('user_story', '')
        legacy_english = data.get('legacy_english', '')
        options = data.get('options', {}) or {}
        if not user_story and not legacy_english:
            return jsonify({'error': 'No input provided'}), 400

        # Load system instructions
        instructions_path = os.path.join(os.path.dirname(__file__), 'agents', 'function_synthesizer_instructions.txt')
        with open(instructions_path, 'r', encoding='utf-8') as f:
            system_instructions = f.read()

        # Compose prompt
        prompt = (
            f"{system_instructions}\n\nUser Story:\n{user_story}\n\nLegacy English Description:\n{legacy_english}\n\nSynthesize functions as instructed."
        )

        # Call OpenAI with a larger-capacity model for fuller outputs
        chat_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4096
        )
        output = chat_response.choices[0].message.content

        # Attempt to parse JSON array; if not JSON, return raw text under 'raw'
        import json, re
        raw_text = output.strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw_text, flags=re.IGNORECASE)
        cleaned = re.sub(r'```\s*$', '', cleaned).strip()
        functions = []
        parsed_ok = False
        try:
            # Find first JSON array
            match = re.search(r'\[(.|\n|\r)*\]', cleaned)
            candidate = match.group(0) if match else cleaned
            functions = json.loads(candidate)
            if isinstance(functions, dict):
                functions = [functions]
            if isinstance(functions, list):
                parsed_ok = True
            else:
                functions = []
        except Exception:
            functions = []

        if parsed_ok:
            # Normalize shapes
            norm = []
            for f in functions:
                if not isinstance(f, dict):
                    continue
                name = str(f.get('name') or '').strip()
                desc = str(f.get('description') or f.get('logic') or '').strip()
                source = (f.get('source') or '').strip().lower()
                if source not in ('user_story', 'legacy_code', 'both'):
                    # Defer deciding until we compute similarities; default to unknown
                    source = ''
                if not name and desc:
                    # Heuristic name from desc first 3-4 words
                    words = re.findall(r'[A-Za-z0-9]+', desc)[:4]
                    if words:
                        name = words[0].lower() + ''.join(w.title() for w in words[1:])
                if name:
                    norm.append({'name': name, 'description': desc, 'source': source})

            # Build word sets for overlap analysis
            STOP = {
                'the','and','for','with','from','that','this','have','has','had','into','out','over','under','then','else','when','into','onto',
                'are','was','were','will','can','could','would','should','may','might','must','shall','per','via','each','any','all','such','than',
                'you','your','our','their','its',"it's",'not','but','also','etc','etc.','about','across','after','before','between','during','against',
                'as','at','by','in','of','on','to','up','off','or','if','is','be','do','did','does'
            }

            def token_set(text: str):
                tokens = [w.lower() for w in re.findall(r'[A-Za-z]{3,}', text or '')]
                return set(t for t in tokens if t not in STOP)

            def jaccard(a: set, b: set) -> float:
                if not a or not b:
                    return 0.0
                inter = len(a & b)
                if inter == 0:
                    return 0.0
                return inter / (len(a) + len(b) - inter)

            story_tokens = token_set(user_story)
            legacy_tokens = token_set(legacy_english)

            # Decide or upgrade to 'both' only when overlap is meaningful on BOTH sides
            # Allow thresholds to be overridden via options
            def clamp(v, lo, hi):
                try:
                    v2 = float(v)
                except Exception:
                    return lo
                return max(lo, min(hi, v2))

            MIN_SHARED_EACH = int(options.get('min_shared_each', 2))
            MIN_SHARED_EACH = max(1, min(10, MIN_SHARED_EACH))
            MIN_SIM_EACH = clamp(options.get('min_sim_each', 0.18), 0.0, 0.9)
            MIN_SIM_AVG = clamp(options.get('min_sim_avg', 0.24), 0.0, 0.95)

            for f in norm:
                f_tokens = token_set(f['name'] + ' ' + f['description'])
                s_inter = len(f_tokens & story_tokens)
                l_inter = len(f_tokens & legacy_tokens)
                s_sim = jaccard(f_tokens, story_tokens)
                l_sim = jaccard(f_tokens, legacy_tokens)
                qualifies_both = (
                    (s_inter >= MIN_SHARED_EACH and l_inter >= MIN_SHARED_EACH) or
                    (s_sim >= MIN_SIM_EACH and l_sim >= MIN_SIM_EACH and (s_sim + l_sim)/2.0 >= MIN_SIM_AVG)
                )
                if f['source'] == 'both':
                    # Downgrade misclassified 'both' if it doesn't meet thresholds
                    if not qualifies_both:
                        f['source'] = 'user_story' if s_sim >= l_sim else 'legacy_code'
                elif f['source'] in ('user_story', 'legacy_code'):
                    if qualifies_both:
                        f['source'] = 'both'
                else:
                    # Unknown source: decide based on higher similarity; only mark 'both' if qualifies
                    if qualifies_both:
                        f['source'] = 'both'
                    else:
                        f['source'] = 'user_story' if s_sim > l_sim else 'legacy_code'

            # Merge near-duplicate function entries; resolve source using practical thresholds
            merged = []
            used = set()

            def normalize_name(n: str) -> str:
                # normalize camel/snake/kebab and strip non-alnum
                n2 = re.sub(r'[^A-Za-z0-9]+', ' ', (n or '').lower()).strip()
                # collapse whitespace
                n2 = re.sub(r'\s+', ' ', n2)
                return n2

            for i, f in enumerate(norm):
                if i in used:
                    continue
                # Search for similar names
                cluster = [f]
                for j, g in enumerate(norm[i+1:], start=i+1):
                    if j in used:
                        continue
                    # Only merge if names are effectively the same after normalization
                    name_sim = normalize_name(g['name']) == normalize_name(f['name'])
                    if name_sim:
                        # Combine
                        cluster.append(g)
                        used.add(j)
                # Resolve cluster source with thresholds
                combined_desc = ' '.join([c['description'] for c in cluster])
                combined_tokens = token_set(f['name'] + ' ' + combined_desc)
                s_sim_c = jaccard(combined_tokens, story_tokens)
                l_sim_c = jaccard(combined_tokens, legacy_tokens)
                s_inter_c = len(combined_tokens & story_tokens)
                l_inter_c = len(combined_tokens & legacy_tokens)
                qualifies_both_c = (
                    (s_inter_c >= MIN_SHARED_EACH and l_inter_c >= MIN_SHARED_EACH) or
                    (s_sim_c >= MIN_SIM_EACH and l_sim_c >= MIN_SIM_EACH and (s_sim_c + l_sim_c)/2.0 >= MIN_SIM_AVG)
                )
                if qualifies_both_c:
                    combined_source = 'both'
                else:
                    # If unanimous, keep unanimous; else pick stronger side
                    if all(c['source'] == 'user_story' for c in cluster):
                        combined_source = 'user_story'
                    elif all(c['source'] == 'legacy_code' for c in cluster):
                        combined_source = 'legacy_code'
                    else:
                        combined_source = 'user_story' if s_sim_c >= l_sim_c else 'legacy_code'
                # Concatenate unique description sentences
                desc_parts = []
                seen_sent = set()
                for c in cluster:
                    for sent in re.split(r'(?<=[.!?])\s+', c['description']):
                        s = sent.strip()
                        if s and s.lower() not in seen_sent:
                            seen_sent.add(s.lower())
                            desc_parts.append(s)
                merged.append({
                    'name': f['name'],
                    'description': ' '.join(desc_parts),
                    'source': combined_source
                })
            functions_out = merged

            # --- COVERAGE ENFORCEMENT: Ensure all business-relevant items from both inputs are represented ---
            VERBS = set([
                'validate','check','ensure','verify','calculate','compute','derive','determine','update','persist','save','post','apply',
                'reconcile','transfer','migrate','map','transform','normalize','sanitize','encrypt','decrypt','authorize','authenticate',
                'emit','publish','subscribe','send','receive','read','write','load','extract','parse','format','generate','create','delete',
                'archive','report','notify','alert','log','retry','roll','back','rollback','commit','sort','filter','aggregate','round'
            ])
            STOP2 = STOP | {'shall','should','must','may','might','could','would','therefore','hence','thus','etc','etc.'}

            def split_to_items(text: str):
                if not text:
                    return []
                # Prefer bullet/numbered lines
                raw_lines = [ln.strip() for ln in text.splitlines()]
                items = []
                for ln in raw_lines:
                    if re.match(r'^\s*([\-*]|\d+[.)])\s+', ln):
                        items.append(re.sub(r'^\s*([\-*]|\d+[.)])\s+', '', ln).strip())
                # If too few, fall back to sentence split
                if len(items) < 3:
                    sents = re.split(r'(?<=[.!?])\s+', text)
                    for s in sents:
                        s2 = s.strip()
                        if len(s2) < 8:
                            continue
                        toks = [w.lower() for w in re.findall(r'[A-Za-z0-9]{3,}', s2)]
                        if any(v in toks for v in VERBS):
                            items.append(s2)
                # Dedup while preserving order
                seen = set()
                out = []
                for it in items:
                    key = it.lower()
                    if key not in seen:
                        seen.add(key)
                        out.append(it)
                return out[:200]  # cap to avoid explosion

            def name_from_text(text: str, fallback_prefix: str):
                words = [w.lower() for w in re.findall(r'[A-Za-z0-9]{3,}', text or '')]
                # Prefer first verb + following 2-3 nouns/tokens
                base = []
                verb_idx = -1
                for i, w in enumerate(words):
                    if w in VERBS:
                        verb_idx = i
                        break
                if verb_idx >= 0:
                    base = words[verb_idx:verb_idx+5]
                else:
                    base = words[:5]
                base = [w for w in base if w not in STOP2]
                if not base:
                    base = [fallback_prefix, 'function']
                camel = base[0]
                for w in base[1:]:
                    camel += w.title()
                return camel[:60]

            # Build fast lookup for existing coverage
            def func_tokens(f):
                return token_set(f.get('name','') + ' ' + f.get('description',''))

            existing = list(functions_out)
            existing_tokens = [func_tokens(f) for f in existing]

            # Allow coverage thresholds to be tuned
            COVERAGE_JACCARD = clamp(options.get('coverage_jaccard', 0.22), 0.05, 0.9)
            COVERAGE_INTERSECT_MIN = int(options.get('coverage_intersect_min', 2))
            COVERAGE_INTERSECT_MIN = max(1, min(10, COVERAGE_INTERSECT_MIN))

            def is_covered(text_item: str) -> bool:
                t = token_set(text_item)
                if not t:
                    return True
                for ft in existing_tokens:
                    if jaccard(t, ft) >= COVERAGE_JACCARD or len(t & ft) >= COVERAGE_INTERSECT_MIN:
                        return True
                return False

            # Collect uncovered items from each side and append as delta functions
            added = []
            story_items = split_to_items(user_story)
            legacy_items = split_to_items(legacy_english)

            # Compute initial coverage counts (before adding any delta)
            def count_covered(items_list):
                covered = 0
                for it in items_list:
                    if is_covered(it):
                        covered += 1
                return covered

            story_covered_before = count_covered(story_items)
            legacy_covered_before = count_covered(legacy_items)

            names_in_use = set(f['name'].lower() for f in existing)

            def uniquify(name: str) -> str:
                base = name
                k = 2
                candidate = name
                while candidate.lower() in names_in_use:
                    candidate = f"{base}{k}"
                    k += 1
                names_in_use.add(candidate.lower())
                return candidate

            # Add uncovered items for full coverage; bounded by the split_to_items cap (<=200 per side)
            max_add_per_side = 200
            story_added = 0
            for it in story_items:
                if story_added >= max_add_per_side:
                    break
                if not is_covered(it):
                    nm = uniquify(name_from_text(it, 'story'))
                    added.append({'name': nm, 'description': f"Covers user story item: {it}", 'source': 'user_story'})
                    existing_tokens.append(token_set(nm + ' ' + it))
                    story_added += 1

            legacy_added = 0
            for it in legacy_items:
                if legacy_added >= max_add_per_side:
                    break
                if not is_covered(it):
                    nm = uniquify(name_from_text(it, 'legacy'))
                    added.append({'name': nm, 'description': f"Covers legacy code item: {it}", 'source': 'legacy_code'})
                    existing_tokens.append(token_set(nm + ' ' + it))
                    legacy_added += 1

            if added:
                logging.info(f"Function Synthesizer coverage: added {len(added)} delta function(s) to ensure full representation.")
                functions_out.extend(added)

            # Recompute coverage after additions to verify full representation
            def count_covered_with_current(items_list):
                cov = 0
                for it in items_list:
                    t = token_set(it)
                    # Covered if any function overlaps sufficiently with this item's tokens
                    ok = False
                    for f in functions_out:
                        ft = func_tokens(f)
                        if jaccard(t, ft) >= COVERAGE_JACCARD or len(t & ft) >= COVERAGE_INTERSECT_MIN:
                            ok = True
                            break
                    if ok:
                        cov += 1
                return cov

            story_covered_after = count_covered_with_current(story_items)
            legacy_covered_after = count_covered_with_current(legacy_items)

            # Ensure preference: common ('both') functions first
            functions_out.sort(key=lambda x: (0 if x['source'] == 'both' else 1, x['name']))
            coverage_summary = {
                'story': {
                    'total_items': len(story_items),
                    'covered_before': story_covered_before,
                    'added': story_added,
                    'covered_after': story_covered_after
                },
                'legacy': {
                    'total_items': len(legacy_items),
                    'covered_before': legacy_covered_before,
                    'added': legacy_added,
                    'covered_after': legacy_covered_after
                }
            }
            thresholds_info = {
                'min_shared_each': MIN_SHARED_EACH,
                'min_sim_each': MIN_SIM_EACH,
                'min_sim_avg': MIN_SIM_AVG,
                'coverage_jaccard': COVERAGE_JACCARD,
                'coverage_intersect_min': COVERAGE_INTERSECT_MIN
            }
            return jsonify({'functions': functions_out, 'coverage': coverage_summary, 'thresholds': thresholds_info, 'raw': raw_text})
        else:
            return jsonify({'raw': raw_text, 'note': 'Could not parse JSON – returning raw model output'})
    except Exception as e:
        logging.error(f"Error in synthesize-functions: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Route to serve microservice_project_view.html
@app.route('/microservice-project-view')
def microservice_project_view():
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
    return send_from_directory(frontend_dir, 'microservice_project_view.html')

# Route map for workflow
ROUTES = [
    ('/', 'legacy_code_parser_agent.html'),
    ('/user-story-decomposer', 'user_story_decomposer_agent.html'),
    ('/function-synthesizer', 'function_synthesizer_agent.html'),
    ('/function-synthesizer-chat', 'function_synthesizer_agent_chat.html'),
    ('/design-decomposer', 'design_decomposer_agent.html'),
    ('/review-prompt', 'review_prompt_agent.html'),
    ('/service-builder', 'service_builder_agent.html'),
    ('/modernization-validator', 'modernization_validator_agent.html'),
]

# Dynamically create routes for each page
for route, html_file in ROUTES:
    def make_route(html_file):
        def route_func():
            return send_from_directory(app.template_folder, html_file)
        return route_func
    # Use a unique endpoint name to avoid conflicts with API endpoints
    endpoint_name = f"page_{html_file.replace('.html','').replace('-','_')}"
    app.add_url_rule(route, endpoint=endpoint_name, view_func=make_route(html_file))


# Optional: redirect /start to first page
@app.route('/start')
def start():
    return redirect(url_for('legacy_code_parser_agent.html'))

# ---------------- Chat Refinement Endpoints (added to avoid 404 HTML responses) ----------------
CHAT_HISTORY = []  # simple in-memory store; resets on restart

@app.route('/chat', methods=['POST'])
def chat_refine():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
        data = request.get_json(silent=True) or {}
        user_msg = (data.get('message') or '').strip()
        current_html = (data.get('current_html') or '').strip()
        if not user_msg:
            return jsonify({'error': 'Empty message'}), 400

        # --- Simple rule-based refinement engine ---
        updated_html = current_html or ''
        import re, html
        # Remove previous highlights to avoid nesting
        if updated_html:
            updated_html = re.sub(r'</?mark[^>]*>', '', updated_html, flags=re.IGNORECASE)
        applied_rules = []

        # 1. Replace pattern: replace "X" with "Y"
        repl_match = re.search(r'replace\s+"([^"]{1,200})"\s+with\s+"([^"]{1,200})"', user_msg, re.IGNORECASE)
        if repl_match and updated_html:
            old, new = repl_match.group(1), repl_match.group(2)
            pattern = re.compile(re.escape(old), re.IGNORECASE)
            def _do_replace(m):
                return f"<mark data-change='replace' title='Replaced from &quot;{html.escape(old)}&quot;'>{html.escape(new)}</mark>"
            new_html_candidate = pattern.sub(_do_replace, updated_html)
            if new_html_candidate != updated_html:
                updated_html = new_html_candidate
                applied_rules.append(f'replaced "{old}" with "{new}"')

        # 2. Remove bullet / line containing quoted phrase: remove "PHRASE"
        rem_match = re.search(r'(remove|delete)\s+"([^"]{1,200})"', user_msg, re.IGNORECASE)
        if rem_match and updated_html:
            phrase = rem_match.group(2)
            # Remove any <li> that contains phrase (case-insensitive)
            updated_html_before = updated_html
            updated_html = re.sub(r'<li[^>]*>[^<]*?' + re.escape(phrase) + r'[^<]*?</li>', '', updated_html, flags=re.IGNORECASE)
            if updated_html != updated_html_before:
                applied_rules.append(f'removed bullet containing "{phrase}"')

        # 3. Add bullet: add bullet: Some text OR add line: Some text / append: Some text
        add_match = re.search(r'(add bullet|add line|append)\s*:?\s+"?([^"].+?)"?$', user_msg, re.IGNORECASE)
        if add_match:
            bullet_text = add_match.group(2).strip()
            if bullet_text:
                # If a UL exists, append inside first; else create new UL at end
                if '<ul' in updated_html.lower():
                    # Insert before closing </ul> of first list
                    safe_bt = html.escape(bullet_text)
                    updated_html = re.sub(r'</ul>', f"<li><mark data-change='add' title='Added bullet'>{safe_bt}</mark></li></ul>", updated_html, count=1, flags=re.IGNORECASE)
                else:
                    safe_bt = html.escape(bullet_text)
                    updated_html += ('' if updated_html.endswith('\n') else '\n') + f"<ul><li><mark data-change='add' title='Added bullet'>{safe_bt}</mark></li></ul>"
                applied_rules.append(f'added bullet "{bullet_text}"')

        # 4. If no rules applied, just append a note at end if current_html exists
        if not applied_rules and updated_html:
            # Append a comment marker (invisible to user unless they view source)
            updated_html += f"\n<!-- no structural rule matched for instruction: {user_msg[:120]} -->"

        # Fallback: if no current_html provided, echo back instruction inside minimal wrapper
        if not updated_html:
            updated_html = f"<div><p>{user_msg}</p></div>"

        timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        assistant_summary = (
            ('Applied: ' + '; '.join(applied_rules)) if applied_rules else 'No structured rule matched; appended note.'
        ) + f" ({timestamp})"

        CHAT_HISTORY.append({'role': 'user', 'content': user_msg})
        CHAT_HISTORY.append({'role': 'assistant', 'content': assistant_summary})
        del CHAT_HISTORY[:-40]

        return jsonify({
            'response': assistant_summary,
            'updated_html': updated_html,
            'applied_rules': applied_rules,
            'history': CHAT_HISTORY
        })
    except Exception as e:
        logging.error(f"/chat error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Server error'}), 500

@app.route('/apply-business-breakdown', methods=['POST'])
def apply_business_breakdown():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
        data = request.get_json(silent=True) or {}
        new_html = (data.get('html') or '').strip()
        if not new_html:
            return jsonify({'error': 'No html provided'}), 400
        # Placeholder: could persist or version this later
        return jsonify({'status': 'ok', 'applied': True})
    except Exception as e:
        logging.error(f"/apply-business-breakdown error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Server error'}), 500

@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith('/chat') or request.path.startswith('/apply-business-breakdown'):
        return jsonify({'error': 'Endpoint not found'}), 404
    return e

@app.errorhandler(405)
def handle_405(e):
    if request.path.startswith('/chat') or request.path.startswith('/apply-business-breakdown'):
        return jsonify({'error': 'Method not allowed'}), 405
    return e


# Assistant IDs
assistant_id_agent_1 = "asst_P2HdYTBuZtBZqHGZizbsii1P"
assistant_id_agent_3 = "asst_bHzpgDT7IB6Bb80GpDmhOxcW"  # Replace with Agent 3's ID

def call_agent(assistant_id, message):
    logging.info(f"Calling assistant with ID: {assistant_id} and message: {message}")
    start_time = time.time()
    try:
        # Step 1: Create a thread (conversation container)
        thread = openai.beta.threads.create()

        # Step 2: Add user message
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=message
        )

        # Step 3: Run the assistant
        run = openai.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        # Step 4: Poll for completion
        while True:
            run_status = openai.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_status.status == "completed":
                break
            elif run_status.status in ["failed", "cancelled", "expired"]:
                raise Exception(f"Run status: {run_status.status}")
            time.sleep(1)

        # Step 5: Get assistant's reply
        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        for msg in reversed(messages.data):
            if msg.role == "assistant":
                end_time = time.time()
                elapsed_time = end_time - start_time
                logging.info(f"Assistant {assistant_id} completed in {elapsed_time:.2f} seconds")
                return msg.content[0].text.value.strip()

    except Exception as e:
        logging.error(f"Error in call_agent for assistant {assistant_id}: {str(e)}")
        raise

@app.route('/upload', methods=['POST'])
def upload_file():
    import traceback
    try:
        uploads_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        instructions_path = os.path.join(os.path.dirname(__file__), 'agents', 'legacy_code_parser_instructions.txt')
        with open(instructions_path, 'r', encoding='utf-8') as f:
            system_instructions = f.read()

        # Check for multiple files (folder upload)
        files = request.files.getlist('files')
        MAX_FILES = 10  # Limit number of files processed
        MAX_FILE_SIZE = 100 * 1024  # 100KB per file
        MAX_CONTENT_CHARS = 4000  # Truncate file content to 4000 chars
        all_outputs = []
        if files and len(files) > 0:
            processed_count = 0
            for file in files:
                if processed_count >= MAX_FILES:
                    logging.warning(f"Skipping file {file.filename}: max file count reached.")
                    break
                if file.filename == '':
                    continue
                # Normalize path and create subdirectories if needed
                norm_filename = os.path.normpath(file.filename)
                file_path = os.path.join(uploads_dir, norm_filename)
                file_dir = os.path.dirname(file_path)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir, exist_ok=True)
                file.save(file_path)
                file_size = os.path.getsize(file_path)
                if file_size > MAX_FILE_SIZE:
                    logging.warning(f"Skipping file {file.filename}: exceeds max size {MAX_FILE_SIZE} bytes.")
                    all_outputs.append(f"### {file.filename}\nSkipped: File too large (>100KB)")
                    continue
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    file_content = f.read(MAX_CONTENT_CHARS)
                if len(file_content) == MAX_CONTENT_CHARS:
                    file_content += '\n...TRUNCATED...'
                prompt = f"{system_instructions}\n\nLegacy Code:\n{file_content}\n\nTranslate and output as instructed. Ensure output uses markdown headings and subheadings for business requirements."
                chat_response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.41,
                    max_tokens=1024
                )
                output = chat_response.choices[0].message.content
                all_outputs.append(f"### {file.filename}\n{output}")
                logging.info(f"Processed file: {file.filename} (folder upload)")
                processed_count += 1
        if all_outputs:
            combined_output = '\n\n'.join(all_outputs)
            return jsonify({'response_agent_1': combined_output})

        # Otherwise, check for single file
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                logging.error('No selected file')
                return jsonify({'error': 'No selected file'}), 400
            file_path = os.path.join(uploads_dir, file.filename)
            file.save(file_path)
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                file_content = f.read()
            prompt = f"{system_instructions}\n\nLegacy Code:\n{file_content}\n\nTranslate and output as instructed. Ensure output uses markdown headings and subheadings for business requirements."
            chat_response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.41,
                max_tokens=1024
            )
            output = chat_response.choices[0].message.content
            logging.info(f"Processed file: {file.filename} (single file upload)")
            return jsonify({'response_agent_1': output})

        logging.error('No file part in request')
        return jsonify({'error': 'No file part'}), 400
    except Exception as e:
        logging.error(f"Error in /upload: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500


# Only run the server after all routes are defined
from jira_user_stories import fetch_user_stories, get_jira_connection

# --- Helpers for JIRA metadata enrichment ---
def derive_labels_components(text: str):
    """
    Very lightweight keyword-based derivation of labels and components.
    - Returns (labels:list[str], components:list[str])
    - All values are lowercase, deduped, and trimmed.
    """
    if not text:
        return [], []
    t = (text or '').lower()
    labels = set()
    components = set()

    # Labels
    if any(k in t for k in ['legacy', 'cobol', 'mainframe']):
        labels.add('legacy')
    if any(k in t for k in ['modern', 'refactor', 'rewrite', 'replatform', 'migration', 'migrate']):
        labels.add('modernization')
        labels.add('migration')
    if any(k in t for k in ['api', 'rest', 'openapi', 'swagger', 'endpoint']):
        labels.add('api')
    if any(k in t for k in ['kafka', 'rabbit', 'queue', 'pub/sub', 'pubsub', 'messag']):
        labels.add('messaging')
    if any(k in t for k in ['auth', 'sso', 'oauth', 'jwt', 'saml', 'mfa', 'authorization']):
        labels.add('security')
        labels.add('authentication')
    if any(k in t for k in ['gdpr', 'pii', 'hipaa', 'sox', 'pci', 'compliance']):
        labels.add('compliance')
    if any(k in t for k in ['performance', 'latency', 'throughput', 'scalab', 'optimiz']):
        labels.add('performance')
    if any(k in t for k in ['k8s', 'kubernetes', 'container', 'docker']):
        labels.add('kubernetes')
        labels.add('containers')
    if any(k in t for k in ['cloud', 'aws', 'azure', 'gcp']):
        labels.add('cloud')

    # Components (heuristic; will only apply if user has permission / component exists)
    if any(k in t for k in ['payment', 'payments', 'billing', 'invoice']):
        components.add('payments')
        if 'billing' in t:
            components.add('billing')
    if any(k in t for k in ['onboarding', 'signup', 'kyc']):
        components.add('onboarding')
    if any(k in t for k in ['auth', 'sso', 'oauth', 'jwt', 'saml', 'mfa']):
        components.add('auth')
    if any(k in t for k in ['kafka', 'rabbit', 'queue', 'pub/sub', 'pubsub']):
        components.add('messaging')
    if any(k in t for k in ['database', 'oracle', 'postgres', 'mysql', 'sqlserver', 'mongodb', 'sql']):
        components.add('database')
    if any(k in t for k in ['report', 'analytics', 'dashboard']):
        components.add('analytics')
    if any(k in t for k in ['order', 'cart', 'checkout']):
        components.add('orders')
    if any(k in t for k in ['cache', 'redis', 'memcached']):
        components.add('cache')

    return sorted(labels), sorted(components)

# API endpoint to get user stories from JIRA
@app.route('/api/jira-user-stories')
def api_jira_user_stories():
    try:
        project_key = request.args.get('project_key', 'SCRUM')
        # Optional paging params
        try:
            startAt = int(request.args.get('startAt', '0'))
        except Exception:
            startAt = 0
        try:
            maxResults = int(request.args.get('maxResults', '20'))
        except Exception:
            maxResults = 20
        result = fetch_user_stories(project_key, startAt=startAt, maxResults=maxResults)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error fetching user stories from JIRA: {str(e)}")
        return jsonify({'error': str(e)}), 500

# API endpoint to get Epics from JIRA
@app.route('/api/jira-epics')
def api_jira_epics():
    try:
        project_key = request.args.get('project_key', os.getenv('JIRA_PROJECT_KEY', 'SCRUM'))
        max_results = int(request.args.get('maxResults', '50'))
        jira = get_jira_connection()
        jql = f"project={project_key} AND issuetype=Epic ORDER BY updated DESC"
        issues = jira.search_issues(jql, maxResults=max_results)
        epics = []
        for issue in issues:
            epics.append({
                'key': issue.key,
                'summary': getattr(issue.fields, 'summary', '')
            })
        return jsonify({'epics': epics})
    except Exception as e:
        logging.error(f"Error fetching epics from JIRA: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# API endpoint to create a new JIRA story from selected legacy requirements
@app.route('/api/create-jira-story', methods=['POST'])
def api_create_jira_story():
    try:
        data = request.get_json(force=True)
        summary = (data.get('summary') or '').strip()
        description = (data.get('description') or '').strip()
        issuetype = (data.get('issuetype') or 'Story').strip() or 'Story'
        labels = data.get('labels') or []
        components = data.get('components') or []
        epic_key = (data.get('epic_key') or '').strip()
        auto_derive = bool(data.get('auto_derive', True))
        if not summary:
            return jsonify({'error': 'summary is required'}), 400
        # Auto-derive labels/components from content and merge with provided
        if auto_derive:
            auto_labels, auto_components = derive_labels_components(f"{summary}\n{description}")
            labels = list({*(l.strip().lower() for l in labels if l), *auto_labels})[:20]
            components = list({*(c.strip().lower() for c in components if c), *auto_components})[:20]
        else:
            labels = [l.strip().lower() for l in labels if l][:20]
            components = [c.strip().lower() for c in components if c][:20]
        jira = get_jira_connection()
        project_key = os.getenv('JIRA_PROJECT_KEY', 'SCRUM')
        # Preferred names (env can override)
        preferred_story_types = [
            (os.getenv('JIRA_STORY_ISSUETYPE_NAME') or issuetype or 'Story').strip() or 'Story',
            'User Story',
            'Story'
        ]
        # Build base fields without issuetype first
        base_fields = {
            'project': {'key': project_key},
            'summary': summary[:255],
            'description': description,
        }
        issue_dict = dict(base_fields)
        issue_dict['issuetype'] = {'name': preferred_story_types[0]}
        if isinstance(labels, list) and labels:
            issue_dict['labels'] = labels
        if isinstance(components, list) and components:
            # JIRA expects components as objects with names
            issue_dict['components'] = [{'name': c} for c in components if c]
        # Try create with fallback issue types if necessary
        last_err = None
        for idx, it_name in enumerate(preferred_story_types):
            issue_dict['issuetype'] = {'name': it_name}
            try:
                try:
                    issue = jira.create_issue(fields=issue_dict)
                    last_err = None
                    break
                except Exception as je:
                    err_text = str(je).lower()
                    # If components cannot be set (field not on screen/unknown), retry without components once
                    if 'components' in err_text and ('cannot be set' in err_text or 'unknown' in err_text):
                        logging.warning('JIRA create failed due to components field; retrying without components...')
                        issue_dict.pop('components', None)
                        issue = jira.create_issue(fields=issue_dict)
                        last_err = None
                        break
                    # If issue type invalid, try next preference
                    if 'issuetype' in err_text and 'valid' in err_text:
                        logging.warning(f"Invalid issuetype '{it_name}' for project {project_key}; trying next fallback...")
                        last_err = je
                        continue
                    # Unknown error, bubble up
                    last_err = je
                    break
            except Exception as je2:
                last_err = je2
                break
        if last_err:
            raise last_err
        # Try to associate to Epic if provided
        if epic_key:
            try:
                # Find the Epic Link field dynamically
                epic_field_id = None
                for fld in jira.fields():
                    try:
                        if str(fld.get('name', '')).strip().lower() == 'epic link':
                            epic_field_id = fld.get('id')
                            break
                    except Exception:
                        continue
                if epic_field_id:
                    issue.update(fields={epic_field_id: epic_key})
                else:
                    # Fallback to Agile API helper if available
                    epic_issue = jira.issue(epic_key)
                    try:
                        jira.add_issues_to_epic(epic_issue.id, [issue.key])
                    except Exception:
                        # Last resort: ignore Epic association if environment doesn't support it
                        logging.warning('Epic association failed: no Epic Link field and Agile API fallback not available.')
            except Exception as epic_err:
                logging.warning(f"Failed to link issue {issue.key} to epic {epic_key}: {epic_err}")
        base = os.getenv('JIRA_BASE_URL', os.getenv('JIRA_SERVER', 'https://lalluluke.atlassian.net'))
        url = f"{base}/browse/{issue.key}".replace('//browse','/browse')
        return jsonify({'key': issue.key, 'url': url})
    except Exception as e:
        logging.error(f"Error creating JIRA story: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# API endpoint to create a new JIRA sub-task under a parent story
@app.route('/api/create-jira-subtask', methods=['POST'])
def api_create_jira_subtask():
    try:
        data = request.get_json(force=True)
        parent_key = (data.get('parent_key') or '').strip()
        summary = (data.get('summary') or '').strip()
        description = (data.get('description') or '').strip()
        labels = data.get('labels') or []
        components = data.get('components') or []
        auto_derive = bool(data.get('auto_derive', True))
        if not parent_key:
            return jsonify({'error': 'parent_key is required'}), 400
        if not summary:
            return jsonify({'error': 'summary is required'}), 400

        if auto_derive:
            auto_labels, auto_components = derive_labels_components(f"{summary}\n{description}")
            labels = list({*(l.strip().lower() for l in labels if l), *auto_labels})[:20]
            components = list({*(c.strip().lower() for c in components if c), *auto_components})[:20]
        else:
            labels = [l.strip().lower() for l in labels if l][:20]
            components = [c.strip().lower() for c in components if c][:20]

        jira = get_jira_connection()
        project_key = os.getenv('JIRA_PROJECT_KEY', 'SCRUM')
        preferred_subtask_types = [
            (os.getenv('JIRA_SUBTASK_ISSUETYPE_NAME') or 'Sub-task'),
            'Subtask',
            'Sub-Task'
        ]
        base_fields = {
            'project': {'key': project_key},
            'summary': summary[:255],
            'description': description,
        }
        # We'll set parent only for sub-task creation
        subtask_issue = dict(base_fields)
        subtask_issue['parent'] = {'key': parent_key}
        subtask_issue['issuetype'] = {'name': preferred_subtask_types[0]}
        if isinstance(labels, list) and labels:
            subtask_issue['labels'] = labels
        if isinstance(components, list) and components:
            subtask_issue['components'] = [{'name': c} for c in components if c]

        issue = None
        last_err = None
        # Try sub-task issue types first
        for it_name in preferred_subtask_types:
            subtask_issue['issuetype'] = {'name': it_name}
            try:
                tmp = dict(subtask_issue)
                if isinstance(components, list) and components:
                    tmp['components'] = [{'name': c} for c in components if c]
                if isinstance(labels, list) and labels:
                    tmp['labels'] = labels
                try:
                    issue = jira.create_issue(fields=tmp)
                    last_err = None
                    break
                except Exception as je:
                    err_text = str(je).lower()
                    if 'components' in err_text and ('cannot be set' in err_text or 'unknown' in err_text):
                        logging.warning('JIRA sub-task create failed due to components field; retrying without components...')
                        tmp.pop('components', None)
                        issue = jira.create_issue(fields=tmp)
                        last_err = None
                        break
                    if 'issuetype' in err_text and 'valid' in err_text:
                        logging.warning(f"Invalid sub-task issuetype '{it_name}' for project {project_key}; trying next fallback...")
                        last_err = je
                        continue
                    last_err = je
                    break
            except Exception as je2:
                last_err = je2
                break
        # Fallback: create a Task and link to parent if sub-tasks are unavailable
        if not issue and last_err:
            logging.warning('Falling back to create Task and link to parent, since Sub-task creation failed.')
            task_fields = dict(base_fields)
            # Try common task names
            for task_name in [(os.getenv('JIRA_TASK_ISSUETYPE_NAME') or 'Task'), 'Task']:
                task_fields['issuetype'] = {'name': task_name}
                try:
                    if isinstance(components, list) and components:
                        task_fields['components'] = [{'name': c} for c in components if c]
                    if isinstance(labels, list) and labels:
                        task_fields['labels'] = labels
                    try:
                        child_issue = jira.create_issue(fields=task_fields)
                    except Exception as te1:
                        # Retry without components if the field is not available
                        err_text = str(te1).lower()
                        if 'components' in err_text and ('cannot be set' in err_text or 'unknown' in err_text):
                            logging.warning('JIRA Task create failed due to components field; retrying without components...')
                            task_fields.pop('components', None)
                            child_issue = jira.create_issue(fields=task_fields)
                        else:
                            raise
                    # Link child to parent story
                    try:
                        jira.create_issue_link(type='Relates', inwardIssue=child_issue.key, outwardIssue=parent_key)
                    except Exception as link_err:
                        logging.warning(f'Failed to create link between {child_issue.key} and {parent_key}: {link_err}')
                    issue = child_issue
                    last_err = None
                    break
                except Exception as te:
                    last_err = te
                    # Try next task name or exit
                    continue
        if last_err:
            raise last_err

        base = os.getenv('JIRA_BASE_URL', os.getenv('JIRA_SERVER', 'https://lalluluke.atlassian.net'))
        url = f"{base}/browse/{issue.key}".replace('//browse','/browse')
        return jsonify({'key': issue.key, 'url': url})
    except Exception as e:
        logging.error(f"Error creating JIRA sub-task: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# API endpoint to update a JIRA user story by appending delta legacy functions
@app.route('/api/update-jira-user-story', methods=['POST'])
def api_update_jira_user_story():
    try:
        data = request.get_json(force=True)
        story_key = data.get('story_key')
        delta_functions = data.get('delta_functions', [])  # list of objects or strings
        if not story_key:
            return jsonify({'error': 'story_key is required'}), 400
        if not isinstance(delta_functions, list) or len(delta_functions) == 0:
            return jsonify({'error': 'delta_functions must be a non-empty list'}), 400

        jira = get_jira_connection()
        issue = jira.issue(story_key)
        existing_description = getattr(issue.fields, 'description', '') or ''

        # Normalize delta functions into markdown bullet list
        bullets = []
        for fn in delta_functions:
            if isinstance(fn, dict):
                name = fn.get('name') or fn.get('function') or 'Unnamed Function'
                desc = fn.get('description') or ''
                bullets.append(f"- {name}: {desc}")
            else:
                bullets.append(f"- {fn}")
        delta_section = "\n\n---\n**Legacy Delta Functions Added:**\n" + "\n".join(bullets) + "\n"

        # Avoid duplicating the same delta section by simple hash / marker
        marker = f"<!-- DELTA-{hash(tuple(bullets))} -->"
        if marker in existing_description:
            return jsonify({'status': 'no_change', 'message': 'Delta already applied', 'story_key': story_key})

        new_description = existing_description + "\n" + marker + delta_section
        issue.update(fields={'description': new_description})
        return jsonify({'status': 'updated', 'story_key': story_key, 'added_functions': len(bullets)})
    except Exception as e:
        logging.error(f"Error updating JIRA story: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# API endpoint to run User Story Decomposer Agent
@app.route('/api/decompose-user-story', methods=['POST'])
def api_decompose_user_story():
    try:
        data = request.get_json()
        user_story = data.get('user_story', '')
        if not user_story:
            return jsonify({'error': 'No user story provided'}), 400

        # Load system instructions
        instructions_path = os.path.join(os.path.dirname(__file__), 'agents', 'user_story_decomposer_instructions.txt')
        with open(instructions_path, 'r', encoding='utf-8') as f:
            system_instructions = f.read()

        # Compose prompt for GPT-3.5
        prompt = f"{system_instructions}\n\nUser Story:\n{user_story}\n\nDecompose and output as instructed."

        # Call OpenAI GPT-3.5 with temperature 0.2 using new API
        chat_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_story}
            ],
            temperature=0.2,
            max_tokens=512
        )
        output = chat_response.choices[0].message.content
        return jsonify({'result': output})
    except Exception as e:
        logging.error(f"Error in decompose-user-story: {str(e)}")
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    import os
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5051))
    app.run(host=host, debug=True, port=port)
