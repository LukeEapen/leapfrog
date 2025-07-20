import os
import openai
import time
import logging
import traceback
import json
from dotenv import load_dotenv
from flask import Flask, send_from_directory, redirect, url_for, request, jsonify

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



# --- Refactored MSBuilder Orchestration ---
import concurrent.futures

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
    prompt = build_file_prompt(file_path, swagger, business_logic, system_instructions)
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
        # Remove code block markers if present
        if code.startswith('```'):
            code = code.split('\n', 1)[-1]
            if code.endswith('```'):
                code = code.rsplit('```', 1)[0]
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

        # Step 2: For each file, generate code in parallel
        file_paths = walk_project_structure(project_structure)
        files = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(call_llm_for_file, file_path, swagger, business_logic, system_instructions): file_path
                for file_path in file_paths
            }
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                code = future.result()
                # Validate code
                if is_stub_or_empty(code):
                    # Retry once if stub/empty
                    code = call_llm_for_file(file_path, swagger, business_logic, system_instructions)
                files[file_path] = code

        # Step 3: Validate all files are present and non-empty
        for file_path in file_paths:
            if file_path not in files or is_stub_or_empty(files[file_path]):
                return jsonify({
                    'error': f'File {file_path} is missing or incomplete after generation.',
                    'build_status': 'error',
                    'message': f'File {file_path} is missing or incomplete. Please retry.'
                }), 500

        # Step 4: Generate msproject.xml (single call)
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
        if 'msproject.xml' not in file_paths:
            # Add to structure if missing
            project_structure['msproject.xml'] = 'file'

        # Step 5: Aggregate and return
        return jsonify({
            'project_structure': project_structure,
            'files': files,
            'build_status': 'success',
            'message': 'Microservice and MS Project plan generated from Swagger and business logic. Review for completeness.'
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
        return jsonify({'swagger': output})
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

        # Try to extract JSON array from output, fallback to safe default
        import json, re
        design = []
        try:
            # Try to find a JSON array in the output
            match = re.search(r'(\[.*?\])', output, re.DOTALL)
            if match:
                design = json.loads(match.group(1))
            else:
                parsed = json.loads(output)
                if isinstance(parsed, list):
                    design = parsed
                elif isinstance(parsed, dict):
                    design = [parsed]
        except Exception as ex:
            logging.warning(f"Could not parse design decomposer LLM output as JSON. Output was: {output.strip()[:200]}... Error: {ex}")
            design = []

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
                # If it's a list of step objects, try to format as numbered steps
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

# API endpoint to synthesize functions from user story and legacy English
@app.route('/api/synthesize-functions', methods=['POST'])
def api_synthesize_functions():
    try:
        data = request.get_json()
        user_story = data.get('user_story', '')
        legacy_english = data.get('legacy_english', '')
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

        # Call OpenAI GPT-3.5 with temperature 0.2 using new API
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
        return jsonify({'result': output})
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
from jira_user_stories import fetch_user_stories

# API endpoint to get user stories from JIRA
@app.route('/api/jira-user-stories')
def api_jira_user_stories():
    try:
        project_key = request.args.get('project_key', 'SCRUM')
        stories = fetch_user_stories(project_key)
        return jsonify({'stories': stories})
    except Exception as e:
        logging.error(f"Error fetching user stories from JIRA: {str(e)}")
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
    app.run(debug=True, port=5050)
