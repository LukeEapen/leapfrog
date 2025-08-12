# --- Static Schema Preview Route ---
import os
import sqlite3
import shutil  # added for backup/rollback
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
        # Capture schema selections including legacy
        source_schema = request.form.get('source_schema') or 'source_schema.json'
        legacy_schema = request.form.get('legacy_schema') or 'legacy_schema.json'
        target_schema = request.form.get('target_schema') or 'target_schema.json'
        session['source_schema'] = source_schema
        session['legacy_schema'] = legacy_schema
        session['target_schema'] = target_schema
        # Clear any previous selections
        for k in ['selected_source_fields', 'selected_legacy_fields', 'selected_target_fields', 'mapping']:
            session.pop(k, None)
        return redirect(url_for('poc4.page1_fields'))
    return render_template('poc4/page1_upload.html')

@poc4_bp.route('/page1_fields', methods=['GET', 'POST'])
def page1_fields():
    """Intermediate field selection page showing 3-way schema view with checkboxes."""
    import json
    base_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas')
    def load_schema(fname):
        try:
            with open(os.path.join(base_path, fname), 'r') as f:
                return json.load(f)
        except Exception:
            return {"tables": []}
    source_schema_file = session.get('source_schema', 'source_schema.json')
    legacy_schema_file = session.get('legacy_schema', 'legacy_schema.json')
    target_schema_file = session.get('target_schema', 'target_schema.json')
    source_schema = load_schema(source_schema_file)
    legacy_schema = load_schema(legacy_schema_file)
    target_schema = load_schema(target_schema_file)
    if request.method == 'POST':
        # Collect selected fields
        selected_source = request.form.getlist('source_field')
        selected_legacy = request.form.getlist('legacy_field')
        selected_target = request.form.getlist('target_field')
        session['selected_source_fields'] = selected_source
        session['selected_legacy_fields'] = selected_legacy
        session['selected_target_fields'] = selected_target
        session.modified = True
        return redirect(url_for('poc4.page2'))
    # Provide flattened field lists
    def flatten(schema, prefix_key):
        rows = []
        for t in schema.get('tables', []):
            for f in t.get('fields', []):
                rows.append({
                    'table': t.get('name'),
                    'name': f.get('name'),
                    'type': f.get('type'),
                    'id': f"{prefix_key}::{t.get('name')}::{f.get('name')}"
                })
        return rows
    return render_template(
        'poc4/page1_fields_select.html',
        source_fields=flatten(source_schema, 'SRC'),
        legacy_fields=flatten(legacy_schema, 'LEG'),
        target_fields=flatten(target_schema, 'TGT'),
        source_schema_file=source_schema_file,
        legacy_schema_file=legacy_schema_file,
        target_schema_file=target_schema_file
    )

@poc4_bp.route('/page2', methods=['GET', 'POST'])
def page2():
    import json
    from .agents import schema_mapping_agent
    # Load schemas
    source_schema_file = session.get('source_schema', 'source_schema.json')
    legacy_schema_file = session.get('legacy_schema', 'legacy_schema.json')
    target_schema_file = session.get('target_schema', 'target_schema.json')
    base_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas')
    def load_schema(fname):
        with open(os.path.join(base_path, fname)) as f:
            return json.load(f)
    source_schema = load_schema(source_schema_file)
    legacy_schema = load_schema(legacy_schema_file)
    target_schema = load_schema(target_schema_file)
    # Helper: field -> table map
    def build_field_table_map(schema):
        m = {}
        for t in schema.get('tables', []):
            for f in t.get('fields', []):
                m[f.get('name')] = t.get('name')
        return m
    source_field_table_map = build_field_table_map(source_schema)
    legacy_field_table_map = build_field_table_map(legacy_schema)
    target_field_table_map = build_field_table_map(target_schema)
    # Filter fields if user selected critical subset
    selected_source = set(session.get('selected_source_fields', []))
    selected_legacy = set(session.get('selected_legacy_fields', []))
    # Build filtered schemas
    def filter_schema(schema, selected):
        if not selected:
            return schema
        filtered_tables = []
        for t in schema.get('tables', []):
            new_fields = [f for f in t.get('fields', []) if f.get('name') in selected]
            if new_fields:
                filtered_tables.append({'name': t.get('name'), 'fields': new_fields})
        return {'tables': filtered_tables}
    filtered_source = filter_schema(source_schema, selected_source)
    filtered_legacy = filter_schema(legacy_schema, selected_legacy)
    # Generate mapping for each origin separately (only once if not in session)
    mapping = session.get('mapping')
    if not mapping:
        source_mapping = schema_mapping_agent.map_schema(filtered_source, target_schema)
        legacy_mapping = schema_mapping_agent.map_schema(filtered_legacy, target_schema)
        # Annotate origin and table names
        for m in source_mapping:
            m['origin'] = 'source'
            m['source_table'] = source_field_table_map.get(m.get('source'))
            if m.get('target'):
                m['target_table'] = target_field_table_map.get(m.get('target'))
            # Auto justification if missing
            if m.get('target') and not m.get('justification'):
                sim = m.get('similarity')
                if isinstance(sim, (int, float)):
                    m['justification'] = f"Auto-mapped by name similarity {round(sim*100,1)}% between {m.get('source')} and {m.get('target')}"
                else:
                    m['justification'] = f"Auto-mapped {m.get('source')} to {m.get('target')} based on heuristic"
        for m in legacy_mapping:
            m['origin'] = 'legacy'
            m['source_table'] = legacy_field_table_map.get(m.get('source'))
            if m.get('target'):
                m['target_table'] = target_field_table_map.get(m.get('target'))
            if m.get('target') and not m.get('justification'):
                sim = m.get('similarity')
                if isinstance(sim, (int, float)):
                    m['justification'] = f"Auto-mapped by name similarity {round(sim*100,1)}% between {m.get('source')} and {m.get('target')}"
                else:
                    m['justification'] = f"Auto-mapped {m.get('source')} to {m.get('target')} based on heuristic"
        mapping = source_mapping + legacy_mapping
    else:
        # Backfill justification if previously stored mapping lacks it
        for m in mapping:
            if m.get('target') and not m.get('justification'):
                sim = m.get('similarity')
                if isinstance(sim, (int, float)):
                    m['justification'] = f"Auto-mapped by name similarity {round(sim*100,1)}% between {m.get('source')} and {m.get('target')}"
                else:
                    m['justification'] = f"Auto-mapped {m.get('source')} to {m.get('target')} based on heuristic"
    # POST updates mapping
    if request.method == 'POST':
        new_mapping = []
        for i, m in enumerate(mapping):
            target_field = request.form.get(f'target_{i}', '')
            target_type = request.form.get(f'target_type_{i}', '')
            new_mapping.append({
                'source': m['source'],
                'source_type': m.get('source_type'),
                'source_table': m.get('source_table'),
                'target': target_field,
                'target_type': target_type,
                'target_table': target_field and target_field_table_map.get(target_field),
                'auto_mapped': bool(target_field),
                'origin': m.get('origin', 'source'),
                'similarity': m.get('similarity'),
                'justification': m.get('justification'),
                'description': m.get('description')
            })
        session['mapping'] = new_mapping
        from .agents import transformation_rule_agent
        rules = transformation_rule_agent.suggest_rules(new_mapping)
        # Enrich rules with table-qualified field name if possible
        for r in rules:
            fld = r.get('field')
            # Find mapping entry with this target
            mtch = next((mm for mm in new_mapping if mm.get('target') == fld), None)
            if mtch and mtch.get('target_table'):
                r['field_full'] = f"{mtch.get('target_table')}.{fld}"
            else:
                r['field_full'] = fld
        session['rules'] = rules
        return redirect(url_for('poc4.page3'))
    # Build target helper structures
    target_fields_list = [f['name'] for t in target_schema.get('tables', []) for f in t.get('fields', [])]
    target_types = {f['name']: f['type'] for t in target_schema.get('tables', []) for f in t.get('fields', [])}
    return render_template('poc4/page2_mapping.html', mapping=mapping, target_fields=target_fields_list, target_types=target_types)

@poc4_bp.route('/page3', methods=['GET', 'POST'])
def page3():
    if request.method == 'POST':
        # Save transformation rules (future: persist edits)
        return redirect(url_for('poc4.page4'))
    mapping = session.get('mapping', [])
    rules = session.get('rules', [])
    # Build helper structures for enrichment
    target_table_lookup = {}
    sources_by_target = {}
    for m in mapping:
        tgt = m.get('target')
        if not tgt:
            continue
        # target table
        if m.get('target_table'):
            target_table_lookup[tgt] = m.get('target_table')
        sources_by_target.setdefault(tgt, []).append(m)
    # Enrich each rule with field_full (table.field) and sources_display
    for r in rules:
        fld = r.get('field')
        if fld:
            tbl = target_table_lookup.get(fld)
            if tbl:
                r.setdefault('field_full', f"{tbl}.{fld}")
            else:
                r.setdefault('field_full', fld)
            src_entries = sources_by_target.get(fld, [])
            if src_entries:
                r['sources_display'] = ', '.join(
                    (f"{se.get('source_table')}.{se.get('source')}" if se.get('source_table') else se.get('source'))
                    for se in src_entries
                )
            else:
                r['sources_display'] = ''
        else:
            r.setdefault('field_full', '')
            r['sources_display'] = ''
    example = "Example transformation: Convert int to uuid for product_id."
    return render_template('poc4/page3_rules.html', mapping=mapping, rules=rules, example=example)

@poc4_bp.route('/page4', methods=['GET', 'POST'])
def page4():
    """Validation & Migration Execution.
    Two-step flow:
      1. Validate (preview summary of planned migration)
      2. Execute (perform migration then redirect to Page 5)
    Also supports running a SELECT query against target DB and provides reset.
    """
    schemas_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas')
    src_schema_file = session.get('source_schema', 'source_schema.json')
    leg_schema_file = session.get('legacy_schema', 'legacy_schema.json')
    tgt_schema_file = session.get('target_schema', 'target_schema.json')
    src_schema_path = os.path.join(schemas_dir, src_schema_file)
    leg_schema_path = os.path.join(schemas_dir, leg_schema_file)
    tgt_schema_path = os.path.join(schemas_dir, tgt_schema_file)
    src_db_path = os.path.join(schemas_dir, 'source.db')
    leg_db_path = os.path.join(schemas_dir, 'legacy.db')
    tgt_db_path = os.path.join(schemas_dir, 'target.db')

    # Ensure DBs exist (create/populate if missing)
    try:
        from sqlite_data_transfer_demo import create_and_populate_db, SAMPLE_DATA
        for p, sp in [(src_db_path, src_schema_path), (leg_db_path, leg_schema_path), (tgt_db_path, tgt_schema_path)]:
            if not os.path.exists(p):
                create_and_populate_db(sp, p, SAMPLE_DATA)
    except Exception:
        pass

    query_result = None
    query_error = None
    migration_preview = session.get('migration_preview')  # previously generated validation summary

    # Query handling
    if request.method == 'POST' and request.form.get('query') is not None:
        user_query = request.form.get('query', '').strip()
        if not user_query.lower().startswith('select'):
            query_error = 'Only SELECT queries are allowed.'
        else:
            try:
                with sqlite3.connect(tgt_db_path) as conn_q:
                    cur = conn_q.execute(user_query)
                    cols = [d[0] for d in cur.description] if cur.description else []
                    rows = cur.fetchall()
                query_result = {'columns': cols, 'rows': [dict(zip(cols, r)) for r in rows]}
            except Exception as e:
                query_error = str(e)
        source_schemas = [f for f in os.listdir(schemas_dir) if f.endswith('.json')]
        target_schemas = [f for f in os.listdir(schemas_dir) if f.endswith('.json')]
        return render_template('poc4/page4_validation.html',
                               source_schemas=source_schemas,
                               target_schemas=target_schemas,
                               selected_source_db=src_schema_file,
                               selected_target_db=tgt_schema_file,
                               query_result=query_result,
                               query_error=query_error,
                               migration_preview=migration_preview)

    # Migration / reset actions
    if request.method == 'POST':
        action = request.form.get('action') or 'validate'
        # Reset databases
        if action == 'reset':
            try:
                from sqlite_data_transfer_demo import reset_all_dbs
                reset_all_dbs(src_schema_path, leg_schema_path, tgt_schema_path)
                session.pop('migration_preview', None)
                session.pop('migration_result', None)
                session.pop('target_backup', None)
                session.modified = True
                migration_preview = None
            except Exception as e:
                query_error = f'Reset failed: {e}'
            source_schemas = [f for f in os.listdir(schemas_dir) if f.endswith('.json')]
            target_schemas = [f for f in os.listdir(schemas_dir) if f.endswith('.json')]
            return render_template('poc4/page4_validation.html',
                                   source_schemas=source_schemas,
                                   target_schemas=target_schemas,
                                   selected_source_db=src_schema_file,
                                   selected_target_db=tgt_schema_file,
                                   migration_preview=migration_preview,
                                   query_error=query_error)

        mapping = session.get('mapping', []) or []
        rules = session.get('rules', []) or []
        if not mapping:
            source_schemas = [f for f in os.listdir(schemas_dir) if f.endswith('.json')]
            target_schemas = [f for f in os.listdir(schemas_dir) if f.endswith('.json')]
            return render_template('poc4/page4_validation.html',
                                   source_schemas=source_schemas,
                                   target_schemas=target_schemas,
                                   selected_source_db=src_schema_file,
                                   selected_target_db=tgt_schema_file,
                                   query_error='No mapping available. Please complete mapping on Page 2.')

        # Build preview summary if validating
        if action == 'validate':
            groups = {}
            for m in mapping:
                s_tbl = m.get('source_table')
                t_tbl = m.get('target_table')
                origin = m.get('origin', 'source')
                s_col = m.get('source')
                t_col = m.get('target')
                if not (s_tbl and t_tbl and s_col and t_col):
                    continue
                groups.setdefault((origin, s_tbl, t_tbl), []).append((s_col, t_col))
            summary_items = []
            total_fields = 0
            total_rows_est = 0
            with sqlite3.connect(src_db_path) as s_conn, sqlite3.connect(leg_db_path) as l_conn:
                for (origin, s_tbl, t_tbl), pairs in groups.items():
                    # unique pairs by target col to avoid duplicates
                    seen = set()
                    uniq = []
                    for s_col, t_col in pairs:
                        if (s_col, t_col) in seen:
                            continue
                        seen.add((s_col, t_col))
                        uniq.append({'source_col': s_col, 'target_col': t_col})
                    total_fields += len(uniq)
                    # Row count estimate
                    row_count = 0
                    try:
                        conn_tmp = s_conn if origin == 'source' else l_conn
                        cur_tmp = conn_tmp.execute(f'SELECT COUNT(1) FROM "{s_tbl}"')
                        row_count = cur_tmp.fetchone()[0]
                        total_rows_est += row_count
                    except Exception:
                        pass
                    summary_items.append({
                        'origin': origin,
                        'source_table': s_tbl,
                        'target_table': t_tbl,
                        'field_count': len(uniq),
                        'fields': uniq,
                        'row_count': row_count
                    })
            migration_preview = {
                'total_table_pairs': len(summary_items),
                'total_field_mappings': total_fields,
                'estimated_total_rows': total_rows_est,
                'items': summary_items
            }
            session['migration_preview'] = migration_preview
            session.modified = True
            source_schemas = [f for f in os.listdir(schemas_dir) if f.endswith('.json')]
            target_schemas = [f for f in os.listdir(schemas_dir) if f.endswith('.json')]
            return render_template('poc4/page4_validation.html',
                                   source_schemas=source_schemas,
                                   target_schemas=target_schemas,
                                   selected_source_db=src_schema_file,
                                   selected_target_db=tgt_schema_file,
                                   migration_preview=migration_preview)

        # Execute migration when approved
        if action == 'execute':
            # Backup target for rollback
            try:
                backup_path = tgt_db_path + '.bak'
                shutil.copyfile(tgt_db_path, backup_path)
                session['target_backup'] = backup_path
            except Exception:
                pass
            tgt_conn = sqlite3.connect(tgt_db_path)
            src_conn = sqlite3.connect(src_db_path)
            leg_conn = sqlite3.connect(leg_db_path)
            total_ins = {'source': 0, 'legacy': 0}
            details = []
            # Build simple transformation map (target_col -> (source_col, cast_sql or None))
            transform_info = {}
            for m in mapping:
                s_col = m.get('source'); t_col = m.get('target')
                if not (s_col and t_col):
                    continue
                src_type = (m.get('source_type') or '').lower()
                tgt_type = (m.get('target_type') or '').lower()
                cast_sql = None
                if src_type and tgt_type and src_type.split('(')[0] != tgt_type.split('(')[0]:
                    # Simplistic cast mapping to TEXT/INTEGER/REAL
                    if 'int' in tgt_type:
                        cast_sql = f'CAST("{s_col}" AS INTEGER)'
                    elif any(x in tgt_type for x in ['real', 'dec', 'float']):
                        cast_sql = f'CAST("{s_col}" AS REAL)'
                    else:
                        cast_sql = f'CAST("{s_col}" AS TEXT)'
                transform_info[(m.get('origin','source'), m.get('source_table'), m.get('target_table'), s_col, t_col)] = cast_sql

            def migrate_from(origin: str, o_conn: sqlite3.Connection):
                groups = {}
                for m in mapping:
                    if m.get('origin', 'source') != origin:
                        continue
                    s_tbl = m.get('source_table')
                    t_tbl = m.get('target_table')
                    s_col = m.get('source')
                    t_col = m.get('target')
                    if not (s_tbl and t_tbl and s_col and t_col):
                        continue
                    groups.setdefault((s_tbl, t_tbl), []).append((s_col, t_col))
                inserted = 0
                for (s_tbl, t_tbl), pairs in groups.items():
                    used_targets = set()
                    select_exprs = []
                    tgt_cols = []
                    for s_col, t_col in pairs:
                        if t_col in used_targets:
                            continue
                        used_targets.add(t_col)
                        cast_sql = transform_info.get((origin, s_tbl, t_tbl, s_col, t_col))
                        if cast_sql:
                            select_exprs.append(f'{cast_sql} AS "{t_col}"')
                        else:
                            select_exprs.append(f'"{s_col}" AS "{t_col}"')
                        tgt_cols.append(t_col)
                    if not select_exprs:
                        continue
                    select_sql = ', '.join(select_exprs)
                    try:
                        src_rows = list(o_conn.execute(f'SELECT {select_sql} FROM "{s_tbl}"'))
                    except Exception as e:
                        details.append(f'{origin}: Failed reading {s_tbl}: {e}')
                        continue
                    qtgt_cols = ', '.join([f'"{c}"' for c in tgt_cols])
                    placeholders = ', '.join(['?' for _ in tgt_cols])
                    for row in src_rows:
                        try:
                            tgt_conn.execute(f'INSERT OR IGNORE INTO "{t_tbl}" ({qtgt_cols}) VALUES ({placeholders})', row)
                            inserted += 1
                        except Exception as e:
                            details.append(f'{origin}: Insert into {t_tbl} failed: {e}')
                    tgt_conn.commit()
                    details.append(f'{origin}: {len(src_rows)} rows processed from {s_tbl} -> {t_tbl} ({len(tgt_cols)} fields; transforms applied: {sum(1 for s_col, t_col in pairs if transform_info.get((origin, s_tbl, t_tbl, s_col, t_col)))})')
                return inserted
            try:
                total_ins['source'] = migrate_from('source', src_conn)
                total_ins['legacy'] = migrate_from('legacy', leg_conn)
            finally:
                src_conn.close(); leg_conn.close(); tgt_conn.close()
            # Re-open target for metrics
            table_counts = {}
            try:
                with sqlite3.connect(tgt_db_path) as c:
                    tgt_tables = {m.get('target_table') for m in mapping if m.get('target_table')}
                    for t in tgt_tables:
                        try:
                            cur = c.execute(f'SELECT COUNT(1) FROM "{t}"')
                            table_counts[t] = cur.fetchone()[0]
                        except Exception:
                            table_counts[t] = 'err'
            except Exception:
                pass
            unmatched_sources = len([m for m in mapping if m.get('target') and not m.get('source')])
            unmatched_targets = len([m for m in mapping if m.get('source') and not m.get('target')])
            summary_text = (f"Migration complete. Inserted source rows: {total_ins['source']}, "
                            f"legacy rows: {total_ins['legacy']}.")
            session['migration_result'] = {
                'summary': summary_text,
                'details': details,
                'preview': migration_preview,
                'table_counts': table_counts,
                'unmatched_sources': unmatched_sources,
                'unmatched_targets': unmatched_targets
            }
            session.pop('migration_preview', None)
            session.modified = True
            return redirect(url_for('poc4.page5'))

    # GET render
    source_schemas = [f for f in os.listdir(schemas_dir) if f.endswith('.json')]
    target_schemas = [f for f in os.listdir(schemas_dir) if f.endswith('.json')]
    return render_template('poc4/page4_validation.html',
                           source_schemas=source_schemas,
                           target_schemas=target_schemas,
                           selected_source_db=src_schema_file,
                           selected_target_db=tgt_schema_file,
                           migration_preview=migration_preview)

@poc4_bp.route('/page5', methods=['GET', 'POST'])
def page5():
    if request.method == 'POST':
        action = request.form.get('action') or 'approve'
        if action == 'rollback':
            backup = session.get('target_backup')
            if backup and os.path.exists(backup):
                tgt_db_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas', 'target.db')
                try:
                    shutil.copyfile(backup, tgt_db_path)
                    msg = 'Rollback completed from backup.'
                    session['reconciliation_result'] = {'status': 'rolled_back', 'message': msg}
                    session.modified = True
                except Exception as e:
                    return render_template('poc4/page5_reconciliation.html', error=f'Rollback failed: {e}', migration_result=session.get('migration_result'), reconciliation_result=session.get('reconciliation_result'), target_backup=backup)
            return render_template('poc4/page5_reconciliation.html', migration_result=session.get('migration_result'), reconciliation_result=session.get('reconciliation_result'), target_backup=backup)
        # Approve/export action
        try:
            result = reconciliation_agent.approve(session.get('migration_result', {}))
            session['reconciliation_result'] = result
            session.modified = True
        except Exception as e:
            return render_template('poc4/page5_reconciliation.html', error=str(e), migration_result=session.get('migration_result'), reconciliation_result=session.get('reconciliation_result'), target_backup=session.get('target_backup'))
    return render_template('poc4/page5_reconciliation.html', migration_result=session.get('migration_result'), reconciliation_result=session.get('reconciliation_result'), target_backup=session.get('target_backup'))

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

