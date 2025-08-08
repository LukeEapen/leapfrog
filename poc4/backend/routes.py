# --- Static Schema Preview Route ---
import os
from flask import send_file, abort
from flask import Blueprint, render_template, request, redirect, url_for, session
from .agents import (
    schema_mapping_agent,
    transformation_rule_agent,
    validation_agent,
    migration_execution_agent,
    reconciliation_agent,
    chatbot_agent
)

poc4_bp = Blueprint('poc4', __name__, url_prefix='/poc4')

@poc4_bp.route('/page1', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        print('Page 1 POST: Next button clicked. Processing schema selection.')
        # Get selected schema filenames from form
        source_schema = request.form.get('source_schema')
        target_schema = request.form.get('target_schema')
        print(f'Selected source_schema: {source_schema}, target_schema: {target_schema}')
        session['source_schema'] = source_schema or 'source_schema.json'
        session['target_schema'] = target_schema or 'target_schema.json'
        return redirect(url_for('poc4.page2'))
    return render_template('poc4/page1_upload.html')

@poc4_bp.route('/page2', methods=['GET', 'POST'])
def page2():
    import json
    from .agents import schema_mapping_agent
    # Get schema filenames from session
    source_schema_file = session.get('source_schema', 'source_schema.json')
    target_schema_file = session.get('target_schema', 'target_schema.json')
    src_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas', source_schema_file)
    tgt_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas', target_schema_file)
    with open(src_path) as f:
        source_schema = json.load(f)
    with open(tgt_path) as f:
        target_schema = json.load(f)
    # If mapping is already in session, use it; else, auto-map
    mapping = session.get('mapping')
    if not mapping:
        mapping = schema_mapping_agent.map_schema(source_schema, target_schema)
    # POST: Save mapping edits and go to next step
    if request.method == 'POST':
        # Save mapping corrections from form
        new_mapping = []
        for i, m in enumerate(mapping):
            target_field = request.form.get(f'target_{i}', '')
            target_type = request.form.get(f'target_type_{i}', '')
            auto_mapped = bool(target_field)
            new_mapping.append({
                'source': m['source'],
                'source_type': m['source_type'],
                'target': target_field,
                'target_type': target_type,
                'auto_mapped': auto_mapped
            })
        session['mapping'] = new_mapping
        # Call TransformationRuleAgent with mapping
        from .agents import transformation_rule_agent
        rules = transformation_rule_agent.suggest_rules(new_mapping)
        session['rules'] = rules
        return redirect(url_for('poc4.page3'))
    # GET: Render mapping visually
    return render_template('poc4/page2_mapping.html', mapping=mapping, target_fields=[f['name'] for t in target_schema.get('tables', []) for f in t.get('fields', [])], target_types={f['name']: f['type'] for t in target_schema.get('tables', []) for f in t.get('fields', [])})

@poc4_bp.route('/page3', methods=['GET', 'POST'])
def page3():
    if request.method == 'POST':
        # Save transformation rules
        # Call validation_agent to simulate migration
        return redirect(url_for('poc4.page4'))
    # GET: Show mapping, rules, and example
    mapping = session.get('mapping', [])
    rules = session.get('rules', [])
    example = "Example transformation: Convert int to uuid for product_id."
    return render_template('poc4/page3_rules.html', mapping=mapping, rules=rules, example=example)

@poc4_bp.route('/page4', methods=['GET', 'POST'])
def page4():
    if request.method == 'POST':
        print('Page 4 POST: Next button clicked. Running sqlite_data_transfer_demo.py as subprocess with mapping and rules.')
        import subprocess, tempfile, json
        try:
            # Write mapping and rules to a temp file
            mapping = session.get('mapping', [])
            rules = session.get('rules', [])
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as tf:
                json.dump({'mapping': mapping, 'rules': rules}, tf)
                tf.flush()
                mapfile = tf.name
            # Run sqlite_data_transfer_demo.py and pass mapping file
            result = subprocess.run([
                'python', 'sqlite_data_transfer_demo.py',
                '--source', 'poc4/frontend/static/schemas/source_schema.json',
                '--target', 'poc4/frontend/static/schemas/target_schema.json',
                '--mapfile', mapfile
            ], capture_output=True, text=True)
            output = result.stdout + '\n' + result.stderr
            session['migration_result'] = output
            print(f'Migration execution output:\n{output}')
        except Exception as e:
            print(f'Error running sqlite_data_transfer_demo.py: {e}')
            return render_template('poc4/page4_validation.html', error=str(e))
        return redirect(url_for('poc4.page5'))
    # Show validation results from validation_agent
    # List available schemas for dropdowns
    schemas_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas')
    source_schemas = [f for f in os.listdir(schemas_dir) if f.endswith('.json')]
    target_schemas = [f for f in os.listdir(schemas_dir) if f.endswith('.json')]
    selected_source_db = session.get('source_schema', 'source_schema.json')
    selected_target_db = session.get('target_schema', 'target_schema.json')
    return render_template(
        'poc4/page4_validation.html',
        source_schemas=source_schemas,
        target_schemas=target_schemas,
        selected_source_db=selected_source_db,
        selected_target_db=selected_target_db
    )

@poc4_bp.route('/page5', methods=['GET', 'POST'])
def page5():
    if request.method == 'POST':
        print('Page 5 POST: Next button clicked. Approving/exporting results.')
        try:
            # Approve/export results (simulate or call real agent)
            result = reconciliation_agent.approve(session.get('migration_result', {}))
            session['reconciliation_result'] = result
            print(f'Reconciliation approve result: {result}')
        except Exception as e:
            print(f'Error in reconciliation_agent.approve: {e}')
            return render_template('poc4/page5_reconciliation.html', error=str(e))
    # Show reconciliation report from reconciliation_agent
    return render_template('poc4/page5_reconciliation.html')

@poc4_bp.route('/static_schema/schemas/<filename>')
def static_schema(filename):
    # Only allow .json files in the schemas directory
    if not filename.endswith('.json'):
        abort(404)
    schema_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas', filename)
    if not os.path.exists(schema_path):
        abort(404)
    return send_file(schema_path, mimetype='application/json')

@poc4_bp.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    response = chatbot_agent.handle_message(user_message, session)
    return {'response': response}

