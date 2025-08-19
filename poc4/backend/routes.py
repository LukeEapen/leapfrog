# --- Static Schema Preview Route ---
import os
import sys
import sqlite3
import shutil  # added for backup/rollback
from flask import send_file, abort
import uuid
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

# Ensure project root is on sys.path so we can import helper modules at repo root
_BACKEND_DIR = os.path.dirname(__file__)
_ROOT_DIR = os.path.abspath(os.path.join(_BACKEND_DIR, '..', '..'))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# --- Server-side state store to avoid oversized cookies ---
_SERVER_STATE: dict[str, dict] = {}

def _ensure_sid():
    sid = session.get('sid')
    if not sid:
        sid = uuid.uuid4().hex
        session['sid'] = sid
    return sid

def _state():
    sid = _ensure_sid()
    return _SERVER_STATE.setdefault(sid, {})

def get_mapping():
    return _state().get('mapping', [])

def set_mapping(m):
    _state()['mapping'] = m

def get_rules():
    return _state().get('rules', [])

def set_rules(r):
    _state()['rules'] = r

def get_preview():
    return _state().get('migration_preview')

def set_preview(p):
    _state()['migration_preview'] = p

def get_result():
    return _state().get('migration_result')

def set_result(x):
    _state()['migration_result'] = x

def clear_state(*keys):
    st = _state()
    for k in keys:
        st.pop(k, None)

# --- Optional demo helpers import with fallbacks ---
try:
    from sqlite_data_transfer_demo import create_and_populate_db as _demo_create_db, SAMPLE_DATA as _DEMO_SAMPLE_DATA, reset_all_dbs as _demo_reset
except Exception:
    _demo_create_db = None
    _DEMO_SAMPLE_DATA = None
    _demo_reset = None

def _fallback_create_and_populate_db(schema_path: str, db_path: str):
    """Create SQLite DB from a JSON schema with empty tables (minimal fallback)."""
    import json as _json
    # Ensure the target directory exists to avoid 'unable to open database file'
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    except Exception:
        pass
    try:
        with open(schema_path, 'r') as f:
            schema = _json.load(f)
    except Exception:
        schema = {'tables': []}
    conn = sqlite3.connect(db_path)
    for t in schema.get('tables', []):
        cols = []
        pk = None
        seen = set()
        for fld in t.get('fields', []):
            name = fld.get('name')
            if not name or name in seen:
                continue
            seen.add(name)
            typ = (fld.get('type') or 'TEXT').upper()
            # very rough mapping
            if typ.startswith('INT'):
                sqlt = 'INTEGER'
            elif any(x in typ for x in ('REAL', 'DEC', 'FLOAT')):
                sqlt = 'REAL'
            else:
                sqlt = 'TEXT'
            cols.append(f'"{name}" {sqlt}')
            if fld.get('primary_key'):
                pk = name
        if pk:
            cols.append(f'PRIMARY KEY("{pk}")')
        try:
            conn.execute(f'DROP TABLE IF EXISTS "{t.get("name")}"')
            conn.execute(f'CREATE TABLE "{t.get("name")}" ({", ".join(cols)})')
        except Exception:
            pass
    conn.commit(); conn.close()
    return True

def _fallback_reset_all_dbs(src_schema_path: str, leg_schema_path: str, tgt_schema_path: str):
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas')
    # Ensure base directory exists
    try:
        os.makedirs(base_dir, exist_ok=True)
    except Exception:
        pass
    src_db = os.path.join(base_dir, 'source.db')
    leg_db = os.path.join(base_dir, 'legacy.db')
    tgt_db = os.path.join(base_dir, 'target.db')
    for p in (src_db, leg_db, tgt_db):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    _fallback_create_and_populate_db(src_schema_path, src_db)
    _fallback_create_and_populate_db(leg_schema_path, leg_db)
    _fallback_create_and_populate_db(tgt_schema_path, tgt_db)
    return {'source': src_db, 'legacy': leg_db, 'target': tgt_db}

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
        for k in ['selected_source_fields', 'selected_legacy_fields', 'selected_target_fields']:
            session.pop(k, None)
        clear_state('mapping', 'rules', 'migration_preview', 'migration_result', 'target_backup')
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
        try:
            with open(os.path.join(base_path, fname), 'r') as f:
                return json.load(f)
        except Exception:
            return {'tables': []}
    source_schema = load_schema(source_schema_file)
    legacy_schema = load_schema(legacy_schema_file)
    target_schema = load_schema(target_schema_file)
    # Helper: field -> table mapping
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
    mapping = get_mapping()
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
        set_mapping(new_mapping)
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
        set_rules(rules)
        return redirect(url_for('poc4.page3'))
    # Build target helper structures
    target_fields_list = [f['name'] for t in target_schema.get('tables', []) for f in t.get('fields', [])]
    target_types = {f['name']: f['type'] for t in target_schema.get('tables', []) for f in t.get('fields', [])}
    return render_template('poc4/page2_mapping.html', mapping=mapping, target_fields=target_fields_list, target_types=target_types)

@poc4_bp.route('/page3', methods=['GET', 'POST'])
def page3():
    if request.method == 'POST':
        action = request.form.get('action') or 'next'
        if action == 'regenerate':
            # Rebuild rules from current mapping then reload page 3 (with variations/examples)
            mapping_cur = get_mapping()
            rebuilt_rules = []
            rules_existing = get_rules() or []
            # First, fold in any preferred selections submitted in this request
            for i, r in enumerate(rules_existing):
                pref_val = request.form.get(f'preferred_{i}')
                if pref_val is not None and pref_val != '':
                    r['preferred'] = pref_val
            # Helper to compute rule variants and example
            def compute_rule_examples(target, tgt_type, src_entries):
                src_names = [se.get('source') for se in src_entries if se.get('source')]
                tgt_bt = (tgt_type or '').split('(')[0].lower()
                tname = (target or '').lower()
                # Start with defaults
                rule_txt = 'Direct mapping' if src_names else 'Default/static value'
                example = (f"Example: Map {src_names[0]} to {target}" if src_names else f"Example: Set {target} to '' (empty)")
                variants = []
                # Multi-source concat
                if len(src_names) >= 2:
                    rule_txt = f"Concatenate {', '.join(src_names[:-1])} + ' ' + {src_names[-1]}"
                    if any('name' in s.lower() for s in src_names) or 'name' in tname:
                        example = "Example: 'John' + ' ' + 'Doe' => 'John Doe'"
                    else:
                        example = "Example: 'A' + ' ' + 'B' => 'A B'"
                    variants.append('Trim multiple spaces after concatenation')
                # Type conversion
                if src_entries:
                    src_bt = (src_entries[0].get('source_type') or '').split('(')[0].lower()
                    if src_bt and tgt_bt and src_bt != tgt_bt:
                        variants.append(f"Type conversion: CAST({src_names[0]} AS {tgt_bt.upper()})")
                        if not src_names:
                            example = f"Example: Default to empty {tgt_bt} for {target}"
                # Intent-specific
                if 'email' in tname:
                    variants.append('Normalize email: LOWER(TRIM(src))')
                    example = "Example: ' Alice@Example.Com ' => 'alice@example.com'"
                if 'phone' in tname:
                    variants.append('Normalize phone: digits only; format E.164')
                    example = "Example: '+1 (415) 555-1234' => '14155551234'"
                if 'date' in tname or 'time' in tname:
                    variants.append('Parse date format YYYYMMDD -> YYYY-MM-DD')
                    example = "Example: '20250107' => '2025-01-07'"
                if 'name' in tname and len(src_names) == 1:
                    variants.append('Title-case and trim name')
                    example = "Example: '  aLiCe  ' => 'Alice'"
                if 'address' in tname:
                    variants.append('Normalize abbreviations (St, Ave); remove double spaces')
                if any(k in tname for k in ['amount','balance','rate']):
                    variants.append('Round to 2 decimals')
                if src_names:
                    variants.append(f"Default when null: COALESCE({src_names[0]}, '')")
                return rule_txt, example, variants

            seen_keys = set()
            for m in mapping_cur:
                tgt = m.get('target'); t_tbl = m.get('target_table'); origin = m.get('origin','source')
                if not (tgt and t_tbl):
                    continue
                key = (origin, t_tbl, tgt)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                src_entries = [mm for mm in mapping_cur if mm.get('target') == tgt and mm.get('origin','source') == origin]
                sources_display = ', '.join(
                    (
                        f"{se.get('source_table')}.{se.get('source')}" if se.get('source_table') and se.get('source') else (se.get('source') or '')
                    ) for se in src_entries if se.get('source')
                )
                rule_txt, example, variants = compute_rule_examples(tgt, m.get('target_type'), src_entries)
                rebuilt_rules.append({
                    'origin': origin,
                    'field': tgt,
                    'field_name': tgt,
                    'target_table': t_tbl,
                    'field_full': f"{t_tbl}.{tgt}",
                    'sources_display': sources_display,
                    'rule': rule_txt,
                    'example': example,
                    'variants': variants
                })
            # Overlay previously selected preferences when possible
            prev_map = { (r.get('origin','source'), r.get('target_table'), r.get('field')): r for r in rules_existing }
            for r in rebuilt_rules:
                k = (r.get('origin','source'), r.get('target_table'), r.get('field'))
                if k in prev_map:
                    if prev_map[k].get('preferred'):
                        r['preferred'] = prev_map[k]['preferred']
            set_rules(rebuilt_rules)
            return redirect(url_for('poc4.page3'))
        elif action in ('apply_all', 'apply_all_source', 'apply_all_legacy'):
            # Apply preferred (or recommended) variant for every rule (optionally filtered by origin), then stay on Page 3
            cur_rules = get_rules() or []
            # Capture any changed preferences from the form first
            for i, r in enumerate(cur_rules):
                pref_val = request.form.get(f'preferred_{i}')
                if pref_val is not None and pref_val != '':
                    r['preferred'] = pref_val
            # Heuristic recommender
            def choose_recommended(rule_obj: dict):
                # 1) explicit preferred
                if rule_obj.get('preferred'):
                    return rule_obj['preferred']
                variants = rule_obj.get('variants') or []
                if not variants:
                    return rule_obj.get('rule')
                fname = (rule_obj.get('field') or '').lower()
                # 2) intent-based
                intent_priority = [
                    ('email', 'Normalize email'),
                    ('phone', 'Normalize phone'),
                    ('date', 'Parse date'),
                    ('time', 'Parse date'),
                ]
                for key, text in intent_priority:
                    if key in fname:
                        for v in variants:
                            if text.lower() in v.lower():
                                return v
                # 3) type conversion if present
                for v in variants:
                    if 'Type conversion' in v:
                        return v
                # 4) default/null handling
                for v in variants:
                    if 'Default when null' in v or 'COALESCE' in v:
                        return v
                # 5) fallback: first variant
                return variants[0]
            # Filter by origin when requested
            origin_filter = None
            if action == 'apply_all_source':
                origin_filter = 'source'
            elif action == 'apply_all_legacy':
                origin_filter = 'legacy'
            for r in cur_rules:
                if origin_filter and r.get('origin') != origin_filter:
                    continue
                chosen = choose_recommended(r)
                if chosen:
                    r['rule'] = chosen
            set_rules(cur_rules)
            return redirect(url_for('poc4.page3'))
        # default is proceed to next step
        # Persist any edited rule text
        cur_rules = get_rules() or []
        for i, r in enumerate(cur_rules):
            new_txt = request.form.get(f'rule_{i}')
            if new_txt is not None:
                r['rule'] = new_txt
            # Also persist any preferred selections
            pref_val = request.form.get(f'preferred_{i}')
            if pref_val is not None and pref_val != '':
                r['preferred'] = pref_val
        set_rules(cur_rules)
        return redirect(url_for('poc4.page4'))
    mapping = get_mapping()
    rules = get_rules()
    # Fallback: if mapping missing (e.g., cookie too large dropped session), rebuild from schemas
    if not mapping:
        try:
            import json
            schemas_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas')
            src_schema_file = session.get('source_schema', 'source_schema.json')
            leg_schema_file = session.get('legacy_schema', 'legacy_schema.json')
            tgt_schema_file = session.get('target_schema', 'target_schema.json')
            def load_schema(fname):
                p = os.path.join(schemas_dir, fname)
                try:
                    with open(p, 'r') as f:
                        return json.load(f)
                except Exception:
                    return {'tables': []}
            src_schema = load_schema(src_schema_file)
            leg_schema = load_schema(leg_schema_file)
            tgt_schema = load_schema(tgt_schema_file)
            # Apply user-selected field filters if present
            selected_source = set(session.get('selected_source_fields', []))
            selected_legacy = set(session.get('selected_legacy_fields', []))
            def filter_schema(schema, selected):
                if not selected:
                    return schema
                out = {'tables': []}
                for t in schema.get('tables', []):
                    flds = [f for f in t.get('fields', []) if f.get('name') in selected]
                    if flds:
                        out['tables'].append({'name': t.get('name'), 'fields': flds})
                return out
            filtered_source = filter_schema(src_schema, selected_source)
            filtered_legacy = filter_schema(leg_schema, selected_legacy)
            # Build helper table maps for annotations
            def field_to_table(schema):
                m = {}
                for t in schema.get('tables', []):
                    for f in t.get('fields', []):
                        m[f.get('name')] = t.get('name')
                return m
            src_ft = field_to_table(src_schema)
            leg_ft = field_to_table(leg_schema)
            tgt_ft = field_to_table(tgt_schema)
            # Generate mappings via agent
            src_map = schema_mapping_agent.map_schema(filtered_source, tgt_schema)
            leg_map = schema_mapping_agent.map_schema(filtered_legacy, tgt_schema)
            for m in src_map:
                m['origin'] = 'source'
                m['source_table'] = src_ft.get(m.get('source'))
                if m.get('target'):
                    m['target_table'] = tgt_ft.get(m.get('target'))
            for m in leg_map:
                m['origin'] = 'legacy'
                m['source_table'] = leg_ft.get(m.get('source'))
                if m.get('target'):
                    m['target_table'] = tgt_ft.get(m.get('target'))
            mapping = src_map + leg_map
            set_mapping(mapping)
        except Exception:
            mapping = []
    # Backfill target_table if absent (older sessions) by reloading target schema
    try:
        if mapping and any(m.get('target') and not m.get('target_table') for m in mapping):
            import json
            target_schema_file = session.get('target_schema', 'target_schema.json')
            base_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas')
            with open(os.path.join(base_path, target_schema_file), 'r') as f:
                tgt_schema = json.load(f)
            field_to_table = {}
            for t in tgt_schema.get('tables', []):
                tname = t.get('name')
                for fld in t.get('fields', []):
                    field_to_table[fld.get('name')] = tname
            for m in mapping:
                if m.get('target') and not m.get('target_table'):
                    m['target_table'] = field_to_table.get(m.get('target'))
            set_mapping(mapping)
    except Exception:
        pass
    # Rebuild rules list strictly from mapping so each origin is isolated and table prefix guaranteed
    rebuilt_rules = []
    seen_keys = set()
    def compute_rule_examples(target, tgt_type, src_entries):
        src_names = [se.get('source') for se in src_entries if se.get('source')]
        tgt_bt = (tgt_type or '').split('(')[0].lower()
        tname = (target or '').lower()
        rule_txt = 'Direct mapping' if src_names else 'Default/static value'
        example = (f"Example: Map {src_names[0]} to {target}" if src_names else f"Example: Set {target} to '' (empty)")
        variants = []
        if len(src_names) >= 2:
            rule_txt = f"Concatenate {', '.join(src_names[:-1])} + ' ' + {src_names[-1]}"
            if any('name' in s.lower() for s in src_names) or 'name' in tname:
                example = "Example: 'John' + ' ' + 'Doe' => 'John Doe'"
            else:
                example = "Example: 'A' + ' ' + 'B' => 'A B'"
            variants.append('Trim multiple spaces after concatenation')
        if src_entries:
            src_bt = (src_entries[0].get('source_type') or '').split('(')[0].lower()
            if src_bt and tgt_bt and src_bt != tgt_bt:
                variants.append(f"Type conversion: CAST({src_names[0]} AS {tgt_bt.upper()})")
                if not src_names:
                    example = f"Example: Default to empty {tgt_bt} for {target}"
        if 'email' in tname:
            variants.append('Normalize email: LOWER(TRIM(src))')
            example = "Example: ' Alice@Example.Com ' => 'alice@example.com'"
        if 'phone' in tname:
            variants.append('Normalize phone: digits only; format E.164')
            example = "Example: '+1 (415) 555-1234' => '14155551234'"
        if 'date' in tname or 'time' in tname:
            variants.append('Parse date format YYYYMMDD -> YYYY-MM-DD')
            example = "Example: '20250107' => '2025-01-07'"
        if 'name' in tname and len(src_names) == 1:
            variants.append('Title-case and trim name')
            example = "Example: '  aLiCe  ' => 'Alice'"
        if 'address' in tname:
            variants.append('Normalize abbreviations (St, Ave); remove double spaces')
        if any(k in tname for k in ['amount','balance','rate']):
            variants.append('Round to 2 decimals')
        if src_names:
            variants.append(f"Default when null: COALESCE({src_names[0]}, '')")
        return rule_txt, example, variants
    for m in mapping:
        tgt = m.get('target'); t_tbl = m.get('target_table'); origin = m.get('origin','source')
        if not (tgt and t_tbl):
            continue
        key = (origin, t_tbl, tgt)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        src_entries = [mm for mm in mapping if mm.get('target') == tgt and mm.get('origin','source') == origin]
        sources_display = ', '.join(
            (
                f"{se.get('source_table')}.{se.get('source')}" if se.get('source_table') and se.get('source') else (se.get('source') or '')
            ) for se in src_entries if se.get('source')
        )
        rule_txt, example, variants = compute_rule_examples(tgt, m.get('target_type'), src_entries)
        rebuilt_rules.append({
            'origin': origin,
            'field': tgt,
            'field_name': tgt,
            'target_table': t_tbl,
            'field_full': f"{t_tbl}.{tgt}",
            'sources_display': sources_display,
            'rule': rule_txt,
            'example': example,
            'variants': variants
        })
    # Overlay any existing rule text or preferred selections from server-side store
    existing = get_rules() or []
    ex_map = { (r.get('origin','source'), r.get('target_table'), r.get('field')): r for r in existing }
    for r in rebuilt_rules:
        k = (r.get('origin','source'), r.get('target_table'), r.get('field'))
        if k in ex_map:
            if ex_map[k].get('rule'):
                r['rule'] = ex_map[k]['rule']
            if ex_map[k].get('preferred'):
                r['preferred'] = ex_map[k]['preferred']
    rules = rebuilt_rules
    set_rules(rules)
    # Fallback: if no rules rebuilt but mapping exists, create basic direct mapping rules
    if not rules and mapping:
        temp_seen = set()
        for m in mapping:
            tgt = m.get('target'); t_tbl = m.get('target_table'); origin = m.get('origin','source')
            if not (tgt and t_tbl):
                continue
            key = (origin, t_tbl, tgt)
            if key in temp_seen:
                continue
            temp_seen.add(key)
            rules.append({
                'origin': origin,
                'field': tgt,
                'field_name': tgt,
                'target_table': t_tbl,
                'field_full': f"{t_tbl}.{tgt}",
                'sources_display': f"{m.get('source_table')}.{m.get('source')}" if m.get('source_table') and m.get('source') else (m.get('source') or ''),
                'rule': 'Direct mapping',
                'example': ''
            })
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
    # Ensure schemas dir exists before any DB file operations
    try:
        os.makedirs(schemas_dir, exist_ok=True)
    except Exception:
        pass
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
    for p, sp in [(src_db_path, src_schema_path), (leg_db_path, leg_schema_path), (tgt_db_path, tgt_schema_path)]:
        if not os.path.exists(p):
            if _demo_create_db and _DEMO_SAMPLE_DATA is not None:
                try:
                    _demo_create_db(sp, p, _DEMO_SAMPLE_DATA)
                except Exception:
                    _fallback_create_and_populate_db(sp, p)
            else:
                _fallback_create_and_populate_db(sp, p)

    query_result = None
    query_error = None
    migration_preview = get_preview()  # previously generated validation summary

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
                # Ensure the schemas directory exists first
                try:
                    os.makedirs(schemas_dir, exist_ok=True)
                except Exception:
                    pass
                if _demo_reset:
                    try:
                        _demo_reset(src_schema_path, leg_schema_path, tgt_schema_path)
                    except Exception:
                        # Fall back if demo reset fails (e.g., path/CWD issues)
                        _fallback_reset_all_dbs(src_schema_path, leg_schema_path, tgt_schema_path)
                else:
                    _fallback_reset_all_dbs(src_schema_path, leg_schema_path, tgt_schema_path)
                clear_state('migration_preview', 'migration_result')
                session.pop('target_backup', None)
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

        # Load mapping/rules from server-side store
        mapping = get_mapping() or []
        rules = get_rules() or []
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
            # Load target schema once for holistic coverage
            import json as _json
            try:
                with open(tgt_schema_path, 'r') as _f:
                    _tgt_schema_obj = _f.read()
                tgt_schema_json = _json.loads(_tgt_schema_obj)
            except Exception:
                tgt_schema_json = {'tables': []}
            tgt_field_catalog = {t.get('name'): {f.get('name') for f in t.get('fields', [])} for t in tgt_schema_json.get('tables', [])}
            # Track mapped fields per target table & origin counts
            mapped_fields_by_table = {}
            origin_counts_by_table = {}
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
                    # Holistic tracking
                    mapped_fields_by_table.setdefault(t_tbl, set()).update([u['target_col'] for u in uniq])
                    origin_counts_by_table.setdefault(t_tbl, {}).setdefault(origin, 0)
                    origin_counts_by_table[t_tbl][origin] += row_count
            # Build holistic coverage list
            holistic = []
            for t_name, tgt_fields in tgt_field_catalog.items():
                mapped = mapped_fields_by_table.get(t_name, set())
                total_target_fields = len(tgt_fields)
                coverage_pct = round((len(mapped) / total_target_fields)*100, 1) if total_target_fields else 0.0
                holistic.append({
                    'target_table': t_name,
                    'mapped_field_count': len(mapped),
                    'total_field_count': total_target_fields,
                    'coverage_pct': coverage_pct,
                    'origins': origin_counts_by_table.get(t_name, {}),
                    'mapped_fields': sorted(mapped)
                })
            holistic.sort(key=lambda x: (-x['coverage_pct'], x['target_table']))
            migration_preview = {
                'total_table_pairs': len(summary_items),
                'total_field_mappings': total_fields,
                'estimated_total_rows': total_rows_est,
                'items': summary_items,
                'holistic': holistic
            }
            set_preview(migration_preview)
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
            # Run entity-based deduplicating migration
            agent_res = migration_execution_agent.run(mapping)
            details = []
            table_counts = {}
            # Post-migration metrics: count rows per impacted target table
            try:
                with sqlite3.connect(tgt_db_path) as c:
                    tgt_tables = sorted({m.get('target_table') for m in mapping if m.get('target_table')})
                    for t in tgt_tables:
                        try:
                            cur = c.execute(f'SELECT COUNT(1) FROM "{t}"')
                            table_counts[t] = cur.fetchone()[0]
                        except Exception as e:
                            table_counts[t] = f'err: {e}'
            except Exception:
                pass
            # Build summary and details from agent logs
            if agent_res.get('status') == 'error':
                summary_text = f"Migration error: {agent_res.get('error')}"
            else:
                # Compute total merged entities across tables if available
                if agent_res.get('global_unique_entities') is not None:
                    total_entities = int(agent_res.get('global_unique_entities') or 0)
                else:
                    total_entities = 0
                    try:
                        for _t, info in (agent_res.get('summary') or {}).items():
                            total_entities += int(info.get('entities') or 0)
                    except Exception:
                        pass
                summary_text = f"Migration complete with entity-based merge. Global unique entities: {total_entities}."
            details.extend(agent_res.get('logs') or [])
            unmatched_sources = len([m for m in mapping if m.get('target') and not m.get('source')])
            unmatched_targets = len([m for m in mapping if m.get('source') and not m.get('target')])
            set_result({
                'summary': summary_text,
                'details': details,
                'preview': migration_preview,
                'table_counts': table_counts,
                'unmatched_sources': unmatched_sources,
                'unmatched_targets': unmatched_targets,
                'agent_result': agent_res
            })
            clear_state('migration_preview')
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
                    return render_template('poc4/page5_reconciliation.html', error=f'Rollback failed: {e}', migration_result=get_result(), reconciliation_result=session.get('reconciliation_result'), target_backup=backup)
                # Fall through to render after successful rollback
                pass
            return render_template('poc4/page5_reconciliation.html', migration_result=get_result(), reconciliation_result=session.get('reconciliation_result'), target_backup=backup)
        # Approve/export action
        try:
            result = reconciliation_agent.approve(get_result() or {})
            session['reconciliation_result'] = result
        except Exception as e:
            return render_template('poc4/page5_reconciliation.html', error=str(e), migration_result=get_result(), reconciliation_result=session.get('reconciliation_result'), target_backup=session.get('target_backup'))
    return render_template('poc4/page5_reconciliation.html', migration_result=get_result(), reconciliation_result=session.get('reconciliation_result'), target_backup=session.get('target_backup'))

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

@poc4_bp.route('/db', methods=['GET'])
def db_browser():
    """Unified browser for source, legacy, and target SQLite DBs.
    Query params:
      db = source|legacy|target (which database to inspect)
      table = table name within that DB (optional)
      limit = row limit (default 50)
    """
    import json
    schemas_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas')
    # Ensure schemas dir exists
    try:
        os.makedirs(schemas_dir, exist_ok=True)
    except Exception:
        pass
    db_files = {
        'source': ('source_schema.json', 'source.db'),
        'legacy': ('legacy_schema.json', 'legacy.db'),
        'target': ('target_schema.json', 'target.db')
    }
    # Ensure DBs exist (lazy create using schema + SAMPLE_DATA if available)
    try:
        from sqlite_data_transfer_demo import create_and_populate_db, SAMPLE_DATA
    except Exception:
        create_and_populate_db = None
        SAMPLE_DATA = None
    for key, (schema_json, db_name) in db_files.items():
        db_path = os.path.join(schemas_dir, db_name)
        if not os.path.exists(db_path):
            schema_path = os.path.join(schemas_dir, schema_json)
            if create_and_populate_db and SAMPLE_DATA is not None:
                try:
                    create_and_populate_db(schema_path, db_path, SAMPLE_DATA or {})
                except Exception:
                    _fallback_create_and_populate_db(schema_path, db_path)
            else:
                _fallback_create_and_populate_db(schema_path, db_path)
    # Gather table listings + counts
    db_tables = {}
    for key, (_schema_json, db_name) in db_files.items():
        path = os.path.join(schemas_dir, db_name)
        tables = []
        if os.path.exists(path):
            try:
                with sqlite3.connect(path) as conn:
                    for (t_name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"):
                        try:
                            (cnt,) = conn.execute(f'SELECT COUNT(1) FROM "{t_name}"').fetchone()
                        except Exception:
                            cnt = 'err'
                        tables.append({'name': t_name, 'count': cnt})
            except Exception:
                pass
        db_tables[key] = tables
    # Selected table view
    sel_db = request.args.get('db', 'target')
    if sel_db not in db_files:
        sel_db = 'target'
    sel_table = request.args.get('table')
    limit = min(max(int(request.args.get('limit', '50') or 50), 1), 500)
    rows = []
    columns = []
    error = None
    if sel_table:
        db_path = os.path.join(schemas_dir, db_files[sel_db][1])
        if os.path.exists(db_path):
            try:
                with sqlite3.connect(db_path) as conn:
                    # Basic defensive quoting (no injection due to simple name whitelist)
                    qtable = sel_table.replace('"', '')
                    cur = conn.execute(f'SELECT * FROM "{qtable}" LIMIT {limit}')
                    columns = [d[0] for d in cur.description] if cur.description else []
                    for r in cur.fetchall():
                        rows.append(dict(zip(columns, r)))
            except Exception as e:
                error = str(e)
    return render_template('poc4/db_browser.html',
                           db_tables=db_tables,
                           sel_db=sel_db,
                           sel_table=sel_table,
                           rows=rows,
                           columns=columns,
                           limit=limit,
                           error=error)

