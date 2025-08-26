# --- Static Schema Preview Route ---
import os
import sys
import sqlite3
import shutil  # added for backup/rollback
from flask import send_file, abort
import uuid
from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from .agents import (
    schema_mapping_agent,
    transformation_rule_agent,
    validation_agent,
    migration_execution_agent,
    reconciliation_agent,
    chatbot_agent,
    TargetModelDesignAgent
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

def get_target_model_draft():
    return _state().get('target_model_draft')

def set_target_model_draft(doc):
    _state()['target_model_draft'] = doc

def get_target_model_final():
    return _state().get('target_model_final')

def set_target_model_final(doc):
    _state()['target_model_final'] = doc

@poc4_bp.route('/target_model', methods=['GET', 'POST'])
def target_model():
    """Dedicated page to design, preview, approve, and download the Target Model."""
    # Common locations
    schemas_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas')
    try:
        os.makedirs(schemas_dir, exist_ok=True)
    except Exception:
        pass
    # Resolve selected schema files from Page 1 (Origin Systems) only.
    # Do NOT fall back to defaults here; use only explicit selections to honor user intent.
    src_schema_file = session.get('source_schema')
    leg_schema_file = session.get('legacy_schema')
    extras_files = session.get('extra_sources', []) or []
    tgt_schema_file = session.get('target_schema', 'target_schema.json')  # for display/download context only

    def _load_json(path):
        import json as _json
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                return _json.load(fh)
        except Exception:
            return {"tables": []}

    # Build a temporary mapping for display if none exists in session
    def _ephemeral_mapping():
        mapping = get_mapping() or []
        if mapping:
            return mapping
        try:
            # Load schemas
            src_doc = _load_json(os.path.join(schemas_dir, src_schema_file)) if src_schema_file else {"tables": []}
            leg_doc = _load_json(os.path.join(schemas_dir, leg_schema_file)) if leg_schema_file else {"tables": []}
            tgt_doc = _load_json(os.path.join(schemas_dir, tgt_schema_file)) if tgt_schema_file else {"tables": []}
            extras_docs = []
            for ef in (extras_files or []):
                if ef:
                    extras_docs.append((ef, _load_json(os.path.join(schemas_dir, ef))))
            # Helpers
            def _build_ft_map(doc):
                ft = {}
                for t in (doc.get('tables') or []):
                    for f in (t.get('fields') or []):
                        nm = f.get('name')
                        if nm:
                            ft[nm] = t.get('name')
                return ft
            src_ft = _build_ft_map(src_doc)
            leg_ft = _build_ft_map(leg_doc)
            tgt_ft = _build_ft_map(tgt_doc)
            # Run mappers
            src_map = schema_mapping_agent.map_schema(src_doc, tgt_doc) if src_doc else []
            leg_map = schema_mapping_agent.map_schema(leg_doc, tgt_doc) if leg_doc else []
            ext_maps = []
            for i, (fn, edoc) in enumerate(extras_docs):
                try:
                    m = schema_mapping_agent.map_schema(edoc, tgt_doc)
                except Exception:
                    m = []
                for mm in m:
                    mm['origin'] = f'ds{i+3}'
                    # backfill source_table
                    # build once per extra
                    eft = _build_ft_map(edoc)
                    mm['source_table'] = eft.get(mm.get('source'))
                    if mm.get('target'):
                        mm['target_table'] = tgt_ft.get(mm.get('target'))
                ext_maps.extend(m)
            # Backfill core maps
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
            return src_map + leg_map + ext_maps
        except Exception:
            return []

    # Build a Mermaid graph for current mapping (source -> target field edges)
    def _build_mapping_mermaid():
        """Build a Mermaid flowchart for Source ➜ Target mapping with:
        - Color-coding by origin (source/legacy/dsN) for left-side nodes and edge strokes.
        - Grouping of target fields into subgraphs per target table.
        - A compact legend showing only the origins present.
        """
        import re
        mapping = get_mapping() or _ephemeral_mapping() or []
        # Build a target field->table map from the selected Target schema
        tgt_schema_path = os.path.join(schemas_dir, tgt_schema_file)
        tgt_doc = _load_json(tgt_schema_path)
        tgt_ft = {}
        try:
            for t in (tgt_doc.get('tables') or []):
                for f in (t.get('fields') or []):
                    nm = f.get('name')
                    if nm:
                        tgt_ft[nm] = t.get('name')
        except Exception:
            pass
        # Collect nodes and edges with origin context
        edges: list[tuple[str, str, str]] = []  # (left_id, right_id, origin)
        left_nodes: dict[str, tuple[str, str]] = {}  # id -> (label, origin)
        # Right nodes grouped by target table
        right_groups: dict[str, dict[str, str]] = {}  # table -> {id: label}

        def sid(s: str) -> str:
            return re.sub(r'[^a-zA-Z0-9_]', '_', s or '')[:80] or 'x'

        for m in mapping:
            s_tbl = m.get('source_table') or ''
            s_col = m.get('source') or ''
            t_tbl = m.get('target_table') or tgt_ft.get((m.get('target') or '')) or ''
            t_col = m.get('target') or ''
            origin = (m.get('origin') or 'source').lower()
            if not (s_col and t_col):
                continue
            origin_cap = 'Source' if origin == 'source' else ('Legacy' if origin == 'legacy' else origin.upper())
            l_label = f"{origin_cap}:{s_tbl}.{s_col}" if s_tbl else f"{origin_cap}:{s_col}"
            r_label = f"{t_tbl}.{t_col}" if t_tbl else f"{t_col}"
            l_id = f"L_{sid(origin)}_{sid(s_tbl)}_{sid(s_col)}"
            r_id = f"R_{sid(t_tbl)}_{sid(t_col)}"
            left_nodes[l_id] = (l_label, origin)
            group = t_tbl or 'UNASSIGNED'
            right_groups.setdefault(group, {})[r_id] = r_label
            edges.append((l_id, r_id, origin))

        if not edges:
            return ''

        # Color palette per origin (nodes + edge strokes)
        def _colors_for_origin(o: str):
            base = {
                'source': ('#E8F0FF', '#3A7BDA', '#0B3A79'),  # fill, stroke, text
                'legacy': ('#FFF3E0', '#FB8C00', '#4A2A00'),
                'ds3':   ('#E8F5E9', '#43A047', '#1B5E20'),
                'ds4':   ('#F3E5F5', '#8E24AA', '#4A148C'),
                'ds5':   ('#E0F7FA', '#00838F', '#004D40'),
            }
            # Any dsN beyond predefined gets a rotating set
            if o in base:
                return base[o]
            if o.startswith('ds'):
                # derive a color deterministically from the number
                try:
                    n = int(o[2:])
                except Exception:
                    n = 6
                palette = [
                    ('#FFFDE7', '#FBC02D', '#5F3700'),  # amber
                    ('#EDE7F6', '#5E35B1', '#311B92'),  # deep purple
                    ('#E1F5FE', '#039BE5', '#01579B'),  # light blue
                    ('#FCE4EC', '#D81B60', '#880E4F'),  # pink
                ]
                return palette[(n - 3) % len(palette)]
            # default gray
            return ('#F5F5F5', '#9E9E9E', '#424242')

        origins_present = sorted({o for _, _, o in edges})

        lines = ["flowchart LR"]

        # Define classes per origin
        for o in origins_present:
            fill, stroke, text = _colors_for_origin(o)
            lines.append(f"  classDef {sid(o)} fill:{fill},stroke:{stroke},stroke-width:1px,color:{text};")

        # Origins cluster with class applied per node
        lines.append("  subgraph Source")
        lines.append("  direction TB")
        for nid, (lbl, o) in sorted(left_nodes.items()):
            ocls = sid(o)
            lines.append(f"  {nid}[\"{lbl}\"]")
            lines.append(f"  class {nid} {ocls};")
        lines.append("  end")

        # Target cluster grouped by table
        lines.append("  subgraph Target")
        lines.append("  direction TB")
        for tbl in sorted(right_groups.keys()):
            sub_name = sid(f"TBL_{tbl}")
            pretty = tbl if tbl != 'UNASSIGNED' else 'Other/Unassigned'
            lines.append(f"  subgraph {sub_name}[\"{pretty}\"]")
            lines.append("  direction TB")
            for nid, lbl in sorted(right_groups[tbl].items()):
                lines.append(f"  {nid}[\"{lbl}\"]")
            lines.append("  end")
        lines.append("  end")

        # Edges and per-edge styles matching origin color
        for l, r, _ in edges:
            lines.append(f"  {l} --> {r}")
        # linkStyle indices are in the order edges are declared in the diagram
        for idx, (_, _, o) in enumerate(edges):
            _, stroke, _ = _colors_for_origin(o)
            lines.append(f"  linkStyle {idx} stroke:{stroke},stroke-width:2px,opacity:0.9;")

        # Legend showing origins present
        if origins_present:
            lines.append("  subgraph LEGEND")
            lines.append("  direction LR")
            for o in origins_present:
                lid = f"LEG_{sid(o)}"
                lines.append(f"  {lid}[\"{o.title()}\"]")
                lines.append(f"  class {lid} {sid(o)};")
            lines.append("  end")

        return "\n".join(lines)

    # Build a simple table-friendly list of mapping pairs
    def _build_mapping_pairs():
        mapping = get_mapping() or _ephemeral_mapping() or []
        # Build field->table map from selected Target schema for backfill
        tgt_schema_path = os.path.join(schemas_dir, tgt_schema_file)
        tgt_doc = _load_json(tgt_schema_path)
        tgt_ft = {}
        try:
            for t in (tgt_doc.get('tables') or []):
                for f in (t.get('fields') or []):
                    nm = f.get('name')
                    if nm:
                        tgt_ft[nm] = t.get('name')
        except Exception:
            pass
        rows = []
        for m in mapping:
            s_tbl = m.get('source_table') or ''
            s_col = m.get('source') or ''
            t_tbl = m.get('target_table') or tgt_ft.get((m.get('target') or '')) or ''
            t_col = m.get('target') or ''
            if not (s_col and t_col):
                continue
            origin = (m.get('origin') or 'source').lower()
            rows.append({
                'origin': origin,
                'source_table': s_tbl,
                'source': s_col,
                'target_table': t_tbl,
                'target': t_col,
            })
        rows.sort(key=lambda r: (
            r.get('target_table') or '',
            r.get('target') or '',
            r.get('origin') or '',
            r.get('source_table') or '',
            r.get('source') or ''
        ))
        return rows

    # Actions
    if request.method == 'POST':
        action = request.form.get('action') or 'design_target_model'
        if action == 'design_target_model':
            docs = []
            for f in (src_schema_file, leg_schema_file):
                if not f:
                    continue
                p = os.path.join(schemas_dir, f)
                if os.path.exists(p):
                    docs.append(_load_json(p))
            # Include any explicitly selected extra data sources from Page 1
            for ef in extras_files:
                if not ef:
                    continue
                p = os.path.join(schemas_dir, ef)
                if os.path.exists(p):
                    docs.append(_load_json(p))
            try:
                agent = TargetModelDesignAgent()
                draft = agent.design(docs)
                set_target_model_draft(draft)
            except Exception as e:
                set_target_model_draft({"error": str(e)})
        elif action == 'approve_target_model':
            draft = get_target_model_draft()
            if draft:
                set_target_model_final(draft)
        # After POST, fall through to a single render below

    # GET: optionally auto-generate when ?auto=1 is present
    if request.method == 'GET' and request.args.get('auto') == '1':
        docs = []
        for f in (src_schema_file, leg_schema_file):
            if not f:
                continue
            p = os.path.join(schemas_dir, f)
            if os.path.exists(p):
                docs.append(_load_json(p))
        # Include any explicitly selected extra data sources from Page 1
        for ef in extras_files:
            if not ef:
                continue
            p = os.path.join(schemas_dir, ef)
            if os.path.exists(p):
                docs.append(_load_json(p))
        try:
            agent = TargetModelDesignAgent()
            draft = agent.design(docs)
            set_target_model_draft(draft)
        except Exception as e:
            set_target_model_draft({"error": str(e)})

    # Single render for both GET/POST
    return render_template(
        'poc4/page_target_model.html',
        target_model_draft=get_target_model_draft(),
        target_model_final=get_target_model_final(),
        mapping_mermaid=_build_mapping_mermaid(),
        mapping_pairs=_build_mapping_pairs()
    )

# --- Schema metadata helpers (categorization: source|legacy|target) ---
def _schemas_dir():
    return os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas')

def _meta_path():
    return os.path.join(_schemas_dir(), 'schemas_meta.json')

def _load_meta():
    import json as _json
    try:
        os.makedirs(_schemas_dir(), exist_ok=True)
    except Exception:
        pass
    p = _meta_path()
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = _json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}

def _save_meta(meta: dict):
    import json as _json
    try:
        os.makedirs(_schemas_dir(), exist_ok=True)
    except Exception:
        pass
    try:
        with open(_meta_path(), 'w', encoding='utf-8') as f:
            _json.dump(meta or {}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# --- Optional demo helpers import with fallbacks ---
try:
    from sqlite_data_transfer_demo import (
        create_and_populate_db as _demo_create_db,
        SAMPLE_DATA as _DEMO_SAMPLE_DATA,
        reset_all_dbs as _demo_reset,
        create_and_populate_cards_ds3_db as _demo_create_ds3
    )
except Exception:
    _demo_create_db = None
    _DEMO_SAMPLE_DATA = None
    _demo_reset = None
    _demo_create_ds3 = None

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
        # Capture schema selections with explicit opt-in for Source/Legacy (no defaults)
        enable_source = request.form.get('enable_source') is not None
        enable_legacy = request.form.get('enable_legacy') is not None
        source_schema = request.form.get('source_schema')
        legacy_schema = request.form.get('legacy_schema')
        ds3_schema = request.form.get('ds3_schema')
        target_schema = request.form.get('target_schema') or 'target_schema.json'
        # Extra data sources (DS3, DS4, ...). Each select uses the same name 'extra_sources'.
        extra_sources = [x for x in request.form.getlist('extra_sources') if x]
        if enable_source and source_schema:
            session['source_schema'] = source_schema
        else:
            session.pop('source_schema', None)
        if enable_legacy and legacy_schema:
            session['legacy_schema'] = legacy_schema
        else:
            session.pop('legacy_schema', None)
        if ds3_schema:
            session['ds3_schema'] = ds3_schema
        session['target_schema'] = target_schema
        # Persist extra sources list (may be empty)
        if extra_sources:
            session['extra_sources'] = extra_sources
        else:
            session.pop('extra_sources', None)
        # Clear any previous selections
        for k in ['selected_source_fields', 'selected_legacy_fields', 'selected_target_fields']:
            session.pop(k, None)
        clear_state('mapping', 'rules', 'migration_preview', 'migration_result', 'target_backup')
        return redirect(url_for('poc4.page1_fields'))
    return render_template('poc4/page1_upload.html', extra_sources=session.get('extra_sources', []))

@poc4_bp.route('/page1/save_selections', methods=['POST'])
def page1_save_selections():
    """Lightweight endpoint to persist Page 1 selections without navigating away.
    Used by the "Design Target Model" quick action on Page 1 to pre-save choices
    before redirecting to Page 4 for auto-design.
    """
    try:
        enable_source = request.form.get('enable_source') in ('true', '1', 'on') or (request.form.get('enable_source') is not None)
        enable_legacy = request.form.get('enable_legacy') in ('true', '1', 'on') or (request.form.get('enable_legacy') is not None)
        source_schema = request.form.get('source_schema')
        legacy_schema = request.form.get('legacy_schema')
        target_schema = request.form.get('target_schema')
        ds3_schema = request.form.get('ds3_schema')
        # Persist if explicitly enabled/selected
        if enable_source and source_schema:
            session['source_schema'] = source_schema
        if not enable_source:
            session.pop('source_schema', None)
        if enable_legacy and legacy_schema:
            session['legacy_schema'] = legacy_schema
        if not enable_legacy:
            session.pop('legacy_schema', None)
        if target_schema:
            session['target_schema'] = target_schema
        if ds3_schema:
            session['ds3_schema'] = ds3_schema
        # Also accept any extra sources passed by the client (optional)
        extra_sources = [x for x in request.form.getlist('extra_sources') if x]
        if extra_sources:
            session['extra_sources'] = extra_sources
        # Note: extra sources/DS3 can be saved via the main Page 1 POST; quick save focuses on core picks
        session.modified = True
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

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
    source_schema_file = session.get('source_schema')
    legacy_schema_file = session.get('legacy_schema')
    extra_sources_files = session.get('extra_sources', [])
    # Ensure DS3 single-pick is treated as an extra source if set
    ds3_schema_file = session.get('ds3_schema')
    if ds3_schema_file:
        try:
            if ds3_schema_file not in (extra_sources_files or []):
                extra_sources_files = (extra_sources_files or []) + [ds3_schema_file]
        except Exception:
            extra_sources_files = [ds3_schema_file]
    target_schema_file = session.get('target_schema', 'target_schema.json')
    source_schema = load_schema(source_schema_file) if source_schema_file else {"tables": []}
    legacy_schema = load_schema(legacy_schema_file) if legacy_schema_file else {"tables": []}
    target_schema = load_schema(target_schema_file)
    if request.method == 'POST':
        # Collect selected fields
        selected_source = request.form.getlist('source_field')
        selected_legacy = request.form.getlist('legacy_field')
        selected_target = request.form.getlist('target_field')
        # Extras are submitted with name pattern extra_field::<index>
        selected_extras = {}
        for i, fname in enumerate(extra_sources_files or []):
            vals = request.form.getlist(f'extra_field::{i}')
            if vals:
                selected_extras[fname] = vals
        session['selected_source_fields'] = selected_source
        session['selected_legacy_fields'] = selected_legacy
        session['selected_target_fields'] = selected_target
        session['selected_extra_fields'] = selected_extras
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
    # Build extras flattened fields list
    extras_flat = []  # list of dicts { index, file, fields: [ {table,name,type,id} ] }
    for i, f in enumerate(extra_sources_files or []):
        extras_flat.append({
            'index': i,
            'file': f,
            'fields': flatten(load_schema(f), f'DS{i+3}') if f else []
        })
    return render_template(
        'poc4/page1_fields_select.html',
        source_fields=flatten(source_schema, 'SRC'),
        legacy_fields=flatten(legacy_schema, 'LEG'),
        target_fields=flatten(target_schema, 'TGT'),
        source_schema_file=source_schema_file,
        legacy_schema_file=legacy_schema_file,
        target_schema_file=target_schema_file,
        extra_sources=extra_sources_files,
        extra_sources_fields=extras_flat
    )

@poc4_bp.route('/page2', methods=['GET', 'POST'])
def page2():
    import json
    from .agents import schema_mapping_agent
    # Load schemas
    source_schema_file = session.get('source_schema')
    legacy_schema_file = session.get('legacy_schema')
    extra_sources_files = session.get('extra_sources', [])
    # Ensure DS3 single-pick is treated as an extra source if set
    ds3_schema_file = session.get('ds3_schema')
    if ds3_schema_file:
        try:
            if ds3_schema_file not in (extra_sources_files or []):
                extra_sources_files = (extra_sources_files or []) + [ds3_schema_file]
        except Exception:
            extra_sources_files = [ds3_schema_file]
    target_schema_file = session.get('target_schema', 'target_schema.json')
    base_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas')
    def load_schema(fname):
        try:
            with open(os.path.join(base_path, fname), 'r') as f:
                return json.load(f)
        except Exception:
            return {'tables': []}
    source_schema = load_schema(source_schema_file) if source_schema_file else {'tables': []}
    legacy_schema = load_schema(legacy_schema_file) if legacy_schema_file else {'tables': []}
    extra_schemas = [(fn, load_schema(fn)) for fn in (extra_sources_files or [])]
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
    selected_extra = session.get('selected_extra_fields', {})  # map filename -> [fields]
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
    filtered_extras = [(fn, filter_schema(schema, set(selected_extra.get(fn, [])))) for fn, schema in extra_schemas]
    # Generate mapping for each origin separately (only once if not in session)
    mapping = get_mapping()
    if not mapping:
        source_mapping = schema_mapping_agent.map_schema(filtered_source, target_schema)
        legacy_mapping = schema_mapping_agent.map_schema(filtered_legacy, target_schema)
        extra_mappings = []
        for idx, (fn, sch) in enumerate(filtered_extras):
            try:
                m = schema_mapping_agent.map_schema(sch, target_schema)
            except Exception:
                m = []
            for mm in m:
                mm['origin'] = f'ds{idx+3}'
            extra_mappings.extend(m)
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
        # Annotate extras with source_table/target_table
        for mm in extra_mappings:
            # For extra schemas, build a map for this schema
            fn = None
            try:
                # Find its index from origin name
                if mm.get('origin', '').startswith('ds'):
                    idx = int(mm['origin'][2:]) - 3
                    if 0 <= idx < len(extra_schemas):
                        fn, sch = extra_schemas[idx]
                        ftm = build_field_table_map(sch)
                        mm['source_table'] = ftm.get(mm.get('source'))
            except Exception:
                pass
            if mm.get('target'):
                mm['target_table'] = target_field_table_map.get(mm.get('target'))
            if mm.get('target') and not mm.get('justification'):
                sim = mm.get('similarity')
                if isinstance(sim, (int, float)):
                    mm['justification'] = f"Auto-mapped by name similarity {round(sim*100,1)}% between {mm.get('source')} and {mm.get('target')}"
                else:
                    mm['justification'] = f"Auto-mapped {mm.get('source')} to {mm.get('target')} based on heuristic"
        mapping = source_mapping + legacy_mapping + extra_mappings
    else:
        # If a previous mapping exists, append mappings for any new extras (e.g., DS3)
        existing_origins = { (mm.get('origin') or 'source').lower() for mm in mapping }
        appended = False
        for idx, (fn, sch) in enumerate(filtered_extras):
            origin_name = f'ds{idx+3}'
            if origin_name in existing_origins:
                continue
            try:
                mnew = schema_mapping_agent.map_schema(sch, target_schema)
            except Exception:
                mnew = []
            for mm in mnew:
                mm['origin'] = origin_name
                # annotate tables
                try:
                    ftm = build_field_table_map(sch)
                    mm['source_table'] = ftm.get(mm.get('source'))
                except Exception:
                    pass
                if mm.get('target'):
                    mm['target_table'] = target_field_table_map.get(mm.get('target'))
                # justification
                if mm.get('target') and not mm.get('justification'):
                    sim = mm.get('similarity')
                    if isinstance(sim, (int, float)):
                        mm['justification'] = f"Auto-mapped by name similarity {round(sim*100,1)}% between {mm.get('source')} and {mm.get('target')}"
                    else:
                        mm['justification'] = f"Auto-mapped {mm.get('source')} to {mm.get('target')} based on heuristic"
            if mnew:
                mapping.extend(mnew)
                appended = True
        # Backfill justification if previously stored mapping lacks it
        for m in mapping:
            if m.get('target') and not m.get('justification'):
                sim = m.get('similarity')
                if isinstance(sim, (int, float)):
                    m['justification'] = f"Auto-mapped by name similarity {round(sim*100,1)}% between {m.get('source')} and {m.get('target')}"
                else:
                    m['justification'] = f"Auto-mapped {m.get('source')} to {m.get('target')} based on heuristic"
        if appended:
            set_mapping(mapping)
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
    # DS3 (cards) schema/db paths (make schema selectable/persistent)
    ds3_schema_file = session.get('ds3_schema', 'source_cards_schema_ds3.json')
    src_schema_path = os.path.join(schemas_dir, src_schema_file)
    leg_schema_path = os.path.join(schemas_dir, leg_schema_file)
    tgt_schema_path = os.path.join(schemas_dir, tgt_schema_file)
    src_db_path = os.path.join(schemas_dir, 'source.db')
    leg_db_path = os.path.join(schemas_dir, 'legacy.db')
    tgt_db_path = os.path.join(schemas_dir, 'target.db')
    ds3_db_path = os.path.join(schemas_dir, 'cards_ds3.db')

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
    # Ensure DS3 DB exists (specialized creator when available)
    try:
        ds3_schema_path = os.path.join(schemas_dir, ds3_schema_file)
        if not os.path.exists(ds3_db_path):
            if 'source_cards_schema_ds3.json' and os.path.exists(ds3_schema_path) and (_demo_create_ds3 is not None):
                try:
                    _demo_create_ds3(ds3_schema_path, ds3_db_path, rows_per_table=20)
                except Exception:
                    _fallback_create_and_populate_db(ds3_schema_path, ds3_db_path)
            else:
                _fallback_create_and_populate_db(ds3_schema_path, ds3_db_path)
    except Exception:
        pass

    query_result = None
    query_error = None
    migration_preview = get_preview()  # previously generated validation summary

    # Helper: list schemas for dropdowns
    def _list_schema_files():
        files = [f for f in os.listdir(schemas_dir) if f.endswith('.json')]
        return files

    # Persist DB selection (Apply button) without triggering validate/execute
    if request.method == 'POST' and (request.form.get('action') is None) and any(k in request.form for k in ('source_db','legacy_db','ds3_db','target_db')) and (request.form.get('query') is None):
        src_sel = request.form.get('source_db')
        leg_sel = request.form.get('legacy_db')
        ds3_sel = request.form.get('ds3_db')
        tgt_sel = request.form.get('target_db')
        if src_sel:
            session['source_schema'] = src_sel
        if leg_sel:
            session['legacy_schema'] = leg_sel
        if ds3_sel:
            session['ds3_schema'] = ds3_sel
        if tgt_sel:
            session['target_schema'] = tgt_sel
        # Refresh local variables from session
        src_schema_file = session.get('source_schema', 'source_schema.json')
        leg_schema_file = session.get('legacy_schema', 'legacy_schema.json')
        tgt_schema_file = session.get('target_schema', 'target_schema.json')
        ds3_schema_file = session.get('ds3_schema', 'source_cards_schema_ds3.json')
        source_schemas = _list_schema_files()
        legacy_schemas = list(source_schemas)
        ds3_schemas = list(source_schemas)
        target_schemas = _list_schema_files()
        return render_template('poc4/page4_validation.html',
                               source_schemas=source_schemas,
                               legacy_schemas=legacy_schemas,
                               ds3_schemas=ds3_schemas,
                               target_schemas=target_schemas,
                               selected_source_db=src_schema_file,
                               selected_legacy_db=leg_schema_file,
                               selected_ds3_db=ds3_schema_file,
                               selected_target_db=tgt_schema_file,
                               migration_preview=migration_preview,
                               target_model_draft=get_target_model_draft(),
                               target_model_final=get_target_model_final())

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
        source_schemas = _list_schema_files()
        legacy_schemas = list(source_schemas)
        ds3_schemas = list(source_schemas)
        target_schemas = _list_schema_files()
        return render_template('poc4/page4_validation.html',
                               source_schemas=source_schemas,
                               legacy_schemas=legacy_schemas,
                               ds3_schemas=ds3_schemas,
                               target_schemas=target_schemas,
                               selected_source_db=src_schema_file,
                               selected_legacy_db=leg_schema_file,
                               selected_ds3_db=ds3_schema_file,
                               selected_target_db=tgt_schema_file,
                               query_result=query_result,
                               query_error=query_error,
                               migration_preview=migration_preview,
                               target_model_draft=get_target_model_draft(),
                               target_model_final=get_target_model_final())

    # Migration / reset actions
    if request.method == 'POST':
        action = request.form.get('action') or 'validate'
        # Design target model from selected source schemas
        if action == 'design_target_model':
            import json as _json
            docs = []
            for f in [src_schema_path, leg_schema_path]:
                try:
                    with open(f, 'r', encoding='utf-8') as fh:
                        docs.append(_json.load(fh))
                except Exception:
                    pass
            # Include DS3 schema if selected
            try:
                with open(os.path.join(schemas_dir, ds3_schema_file), 'r', encoding='utf-8') as fh:
                    docs.append(_json.load(fh))
            except Exception:
                pass
            try:
                agent = TargetModelDesignAgent()
                draft = agent.design(docs)
                set_target_model_draft(draft)
            except Exception as e:
                set_target_model_draft({"error": str(e)})
            source_schemas = _list_schema_files()
            legacy_schemas = list(source_schemas)
            ds3_schemas = list(source_schemas)
            target_schemas = _list_schema_files()
            return render_template('poc4/page4_validation.html',
                                   source_schemas=source_schemas,
                                   legacy_schemas=legacy_schemas,
                                   ds3_schemas=ds3_schemas,
                                   target_schemas=target_schemas,
                                   selected_source_db=src_schema_file,
                                   selected_legacy_db=leg_schema_file,
                                   selected_ds3_db=ds3_schema_file,
                                   selected_target_db=tgt_schema_file,
                                   migration_preview=migration_preview,
                                   target_model_draft=get_target_model_draft())
        if action == 'approve_target_model':
            # Persist draft as final and offer download link
            draft = get_target_model_draft()
            if draft:
                set_target_model_final(draft)
            source_schemas = _list_schema_files()
            legacy_schemas = list(source_schemas)
            ds3_schemas = list(source_schemas)
            target_schemas = _list_schema_files()
            return render_template('poc4/page4_validation.html',
                                   source_schemas=source_schemas,
                                   legacy_schemas=legacy_schemas,
                                   ds3_schemas=ds3_schemas,
                                   target_schemas=target_schemas,
                                   selected_source_db=src_schema_file,
                                   selected_legacy_db=leg_schema_file,
                                   selected_ds3_db=ds3_schema_file,
                                   selected_target_db=tgt_schema_file,
                                   migration_preview=migration_preview,
                                   target_model_draft=get_target_model_draft(),
                                   target_model_final=get_target_model_final())
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
            source_schemas = _list_schema_files()
            legacy_schemas = list(source_schemas)
            ds3_schemas = list(source_schemas)
            target_schemas = _list_schema_files()
            return render_template('poc4/page4_validation.html',
                                   source_schemas=source_schemas,
                                   legacy_schemas=legacy_schemas,
                                   ds3_schemas=ds3_schemas,
                                   target_schemas=target_schemas,
                                   selected_source_db=src_schema_file,
                                   selected_legacy_db=leg_schema_file,
                                   selected_ds3_db=ds3_schema_file,
                                   selected_target_db=tgt_schema_file,
                                   migration_preview=migration_preview,
                                   query_error=query_error,
                                   target_model_draft=get_target_model_draft(),
                                   target_model_final=get_target_model_final())

        # Load mapping/rules from server-side store
        mapping = get_mapping() or []
        rules = get_rules() or []
        if not mapping:
            source_schemas = _list_schema_files()
            legacy_schemas = list(source_schemas)
            ds3_schemas = list(source_schemas)
            target_schemas = _list_schema_files()
            return render_template('poc4/page4_validation.html',
                                   source_schemas=source_schemas,
                                   legacy_schemas=legacy_schemas,
                                   ds3_schemas=ds3_schemas,
                                   target_schemas=target_schemas,
                                   selected_source_db=src_schema_file,
                                   selected_legacy_db=leg_schema_file,
                                   selected_ds3_db=ds3_schema_file,
                                   selected_target_db=tgt_schema_file,
                                   query_error='No mapping available. Please complete mapping on Page 2.',
                                   target_model_draft=get_target_model_draft(),
                                   target_model_final=get_target_model_final())

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
            # Track totals separately: non-DS3 sum and DS3 aggregated per target table (unique entities)
            total_rows_non_ds3 = 0
            ds3_max_by_target = {}
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
            with sqlite3.connect(src_db_path) as s_conn, sqlite3.connect(leg_db_path) as l_conn, sqlite3.connect(ds3_db_path) as d3_conn:
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
                    # Row count estimate; for DS3, count unique cardholders linked to the source table
                    row_count = 0
                    try:
                        # Route row count by origin; support ds3 specifically
                        if origin == 'source':
                            conn_tmp = s_conn
                        elif origin == 'legacy':
                            conn_tmp = l_conn
                        elif origin.startswith('ds'):
                            # DS3: compute unique entity count instead of raw rows
                            # Use table-aware distinct anchors
                            if s_tbl == 'cardholder':
                                cur_tmp = d3_conn.execute('SELECT COUNT(DISTINCT cardholder_id) FROM "cardholder"')
                                row_count = cur_tmp.fetchone()[0]
                                conn_tmp = None
                            elif s_tbl == 'card_account':
                                cur_tmp = d3_conn.execute('SELECT COUNT(DISTINCT cardholder_id) FROM "card_account" ca JOIN cardholder ch ON ch.cardholder_id = ca.cardholder_id')
                                row_count = cur_tmp.fetchone()[0]
                                conn_tmp = None
                            elif s_tbl == 'card':
                                cur_tmp = d3_conn.execute('SELECT COUNT(DISTINCT ca.cardholder_id) FROM "card" c JOIN card_account ca ON ca.card_account_id = c.card_account_id')
                                row_count = cur_tmp.fetchone()[0]
                                conn_tmp = None
                            elif s_tbl == 'card_auth':
                                cur_tmp = d3_conn.execute('SELECT COUNT(DISTINCT ca.cardholder_id) FROM card_auth a JOIN card c ON c.card_id = a.card_id JOIN card_account ca ON ca.card_account_id = c.card_account_id')
                                row_count = cur_tmp.fetchone()[0]
                                conn_tmp = None
                            elif s_tbl == 'card_txn':
                                cur_tmp = d3_conn.execute('SELECT COUNT(DISTINCT ca.cardholder_id) FROM card_txn t JOIN card c ON c.card_id = t.card_id JOIN card_account ca ON ca.card_account_id = c.card_account_id')
                                row_count = cur_tmp.fetchone()[0]
                                conn_tmp = None
                            elif s_tbl == 'dispute_case':
                                cur_tmp = d3_conn.execute('SELECT COUNT(DISTINCT ca.cardholder_id) FROM dispute_case dc JOIN card_txn t ON t.txn_id = dc.txn_id JOIN card c ON c.card_id = t.card_id JOIN card_account ca ON ca.card_account_id = c.card_account_id')
                                row_count = cur_tmp.fetchone()[0]
                                conn_tmp = None
                            elif s_tbl == 'merchant':
                                # Count distinct cardholders who have transactions with the merchant
                                cur_tmp = d3_conn.execute('SELECT COUNT(DISTINCT ca.cardholder_id) FROM merchant m JOIN card_txn t ON t.merchant_id = m.merchant_id JOIN card c ON c.card_id = t.card_id JOIN card_account ca ON ca.card_account_id = c.card_account_id')
                                row_count = cur_tmp.fetchone()[0]
                                conn_tmp = None
                            else:
                                conn_tmp = d3_conn
                        else:
                            conn_tmp = l_conn
                        if conn_tmp is not None:
                            cur_tmp = conn_tmp.execute(f'SELECT COUNT(1) FROM "{s_tbl}"')
                            row_count = cur_tmp.fetchone()[0]
                        # Aggregate totals: DS3 counts are unique entities, so only count once per target table (take max)
                        if origin.startswith('ds'):
                            # Use max to approximate union of holders across DS3 tables
                            try:
                                prev = ds3_max_by_target.get(t_tbl, 0)
                                if isinstance(row_count, int):
                                    ds3_max_by_target[t_tbl] = max(prev, row_count)
                                else:
                                    ds3_max_by_target[t_tbl] = prev
                            except Exception:
                                pass
                        else:
                            total_rows_non_ds3 += row_count
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
            # Compute final estimated total rows with DS3 aggregated by target table
            total_rows_est = total_rows_non_ds3 + sum(v for v in ds3_max_by_target.values())
            migration_preview = {
                'total_table_pairs': len(summary_items),
                'total_field_mappings': total_fields,
                'estimated_total_rows': total_rows_est,
                'items': summary_items,
                'holistic': holistic
            }
            set_preview(migration_preview)
            source_schemas = _list_schema_files()
            legacy_schemas = list(source_schemas)
            ds3_schemas = list(source_schemas)
            target_schemas = _list_schema_files()
            return render_template('poc4/page4_validation.html',
                                   source_schemas=source_schemas,
                                   legacy_schemas=legacy_schemas,
                                   ds3_schemas=ds3_schemas,
                                   target_schemas=target_schemas,
                                   selected_source_db=src_schema_file,
                                   selected_legacy_db=leg_schema_file,
                                   selected_ds3_db=ds3_schema_file,
                                   selected_target_db=tgt_schema_file,
                                   migration_preview=migration_preview,
                                   target_model_draft=get_target_model_draft(),
                                   target_model_final=get_target_model_final())

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
    source_schemas = _list_schema_files()
    legacy_schemas = list(source_schemas)
    ds3_schemas = list(source_schemas)
    target_schemas = _list_schema_files()
    return render_template('poc4/page4_validation.html',
                           source_schemas=source_schemas,
                           legacy_schemas=legacy_schemas,
                           ds3_schemas=ds3_schemas,
                           target_schemas=target_schemas,
                           selected_source_db=src_schema_file,
                           selected_legacy_db=leg_schema_file,
                           selected_ds3_db=ds3_schema_file,
                           selected_target_db=tgt_schema_file,
                           migration_preview=migration_preview,
                           target_model_draft=get_target_model_draft(),
                           target_model_final=get_target_model_final())

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
        'target': ('target_schema.json', 'target.db'),
        # Third data source (cards dataset)
        'ds3': ('source_cards_schema_ds3.json', 'cards_ds3.db')
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
            # Use specialized DS3 creator when appropriate, else fall back to generic/demo
            if key == 'ds3' and _demo_create_ds3 is not None:
                try:
                    _demo_create_ds3(schema_path, db_path, rows_per_table=20)
                except Exception:
                    _fallback_create_and_populate_db(schema_path, db_path)
            else:
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
    # Column selection: support repeated cols params (?cols=a&cols=b) or comma-separated (?cols=a,b)
    raw_cols = request.args.getlist('cols') or []
    if not raw_cols:
        csv = (request.args.get('cols') or '').strip()
        if csv:
            raw_cols = [c.strip() for c in csv.split(',') if c.strip()]
    selected_cols = [c for c in raw_cols if c]
    rows = []
    columns = []
    error = None
    available_columns = []
    if sel_table:
        db_path = os.path.join(schemas_dir, db_files[sel_db][1])
        if os.path.exists(db_path):
            try:
                with sqlite3.connect(db_path) as conn:
                    qtable = sel_table.replace('"', '')
                    # Fetch available columns via PRAGMA
                    try:
                        ac = conn.execute(f'PRAGMA table_info("{qtable}")').fetchall()
                        available_columns = [a[1] for a in ac]
                    except Exception:
                        available_columns = []
                    # Build safe column list
                    if selected_cols:
                        # Whitelist against available_columns
                        safe_cols = [c for c in selected_cols if c in available_columns]
                        if not safe_cols:
                            safe_cols = available_columns[:]
                    else:
                        safe_cols = []  # empty => select all
                    if safe_cols:
                        col_sql = ', '.join([f'"{c}"' for c in safe_cols])
                        cur = conn.execute(f'SELECT {col_sql} FROM "{qtable}" LIMIT {limit}')
                        columns = [d[0] for d in cur.description] if cur.description else safe_cols
                    else:
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
                           error=error,
                           available_columns=available_columns,
                           selected_cols=selected_cols)

@poc4_bp.route('/download/target_model.json')
def download_target_model():
    import json as _json
    doc = get_target_model_final() or get_target_model_draft()
    if not doc:
        abort(404)
    try:
        tmp_path = os.path.join(_schemas_dir(), f"target_model_{uuid.uuid4().hex[:8]}.json")
        with open(tmp_path, 'w', encoding='utf-8') as f:
            _json.dump(doc, f, ensure_ascii=False, indent=2)
        return send_file(tmp_path, as_attachment=True, download_name='target_model.json', mimetype='application/json')
    except Exception:
        abort(500)

@poc4_bp.route('/upload_schema', methods=['POST'])
def upload_schema():
    """Upload a schema JSON file into the static schemas directory and return its filename.
    Frontend uses this to "Add more" schemas into the dropdown on Page 1.
    """
    try:
        file = request.files.get('schemaFile')
        if not file:
            return {"ok": False, "error": "No file provided."}, 400
        # Basic validation: ensure JSON by attempting to parse
        import json as _json
        try:
            data = _json.loads(file.read().decode('utf-8'))
            # Simple structure check
            if not isinstance(data, dict) or 'tables' not in data:
                # Still allow, but enforce a dict with tables
                data = {"tables": data if isinstance(data, list) else []}
        except Exception:
            return {"ok": False, "error": "Invalid JSON."}, 400
        # Reset file stream (we'll re-dump JSON to ensure normalized content)
        file.stream.seek(0)
        # Compute a safe filename
        raw_name = request.form.get('name') or file.filename or 'schema.json'
        safe = secure_filename(raw_name)
        if not safe.lower().endswith('.json'):
            safe += '.json'
        # Ensure uniqueness
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas')
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            pass
        dest_path = os.path.join(base_dir, safe)
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(safe)
            safe = f"{name}_{uuid.uuid4().hex[:6]}{ext}"
            dest_path = os.path.join(base_dir, safe)
        # Persist normalized JSON
        try:
            with open(dest_path, 'w', encoding='utf-8') as f:
                _json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            return {"ok": False, "error": f"Save failed: {e}"}, 500
        # Optional categorize
        kind = (request.form.get('kind') or '').strip().lower()
        if kind in ('source','legacy','target'):
            meta = _load_meta()
            meta[safe] = {'kind': kind}
            _save_meta(meta)
        return {"ok": True, "filename": safe}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

@poc4_bp.route('/rename_schema', methods=['POST'])
def rename_schema():
    """Rename an existing schema JSON file in the static schemas directory.
    Expects form fields: oldName, newName. Returns the finalized new filename.
    """
    try:
        old_name = (request.form.get('oldName') or '').strip()
        new_name = (request.form.get('newName') or '').strip()
        if not old_name or not new_name:
            return {"ok": False, "error": "oldName and newName are required."}, 400
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas')
        old_safe = secure_filename(old_name)
        new_safe = secure_filename(new_name)
        if not new_safe.lower().endswith('.json'):
            new_safe += '.json'
        old_path = os.path.join(base_dir, old_safe)
        if not os.path.exists(old_path):
            return {"ok": False, "error": "Original file not found."}, 404
        new_path = os.path.join(base_dir, new_safe)
        if os.path.exists(new_path):
            name, ext = os.path.splitext(new_safe)
            new_safe = f"{name}_{uuid.uuid4().hex[:6]}{ext}"
            new_path = os.path.join(base_dir, new_safe)
        os.rename(old_path, new_path)
        # Update meta mapping
        meta = _load_meta()
        if old_safe in meta:
            meta[new_safe] = meta.pop(old_safe)
            _save_meta(meta)
        return {"ok": True, "filename": new_safe}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

@poc4_bp.route('/list_schemas', methods=['GET'])
def list_schemas():
    """Return a categorized list of schema JSON filenames.
    Response: { ok: true, filesByKind: { source:[], legacy:[], target:[], uncategorized:[] }, files:[all] }
    """
    try:
        base_dir = _schemas_dir()
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            pass
        files = sorted([f for f in os.listdir(base_dir) if f.lower().endswith('.json')])
        by = {'source': [], 'legacy': [], 'target': [], 'uncategorized': []}
        meta = _load_meta()
        for f in files:
            # Skip meta file itself
            if f == 'schemas_meta.json':
                continue
            kind = (meta.get(f) or {}).get('kind')
            if kind in by:
                by[kind].append(f)
            else:
                # Heuristic
                name = f.lower()
                if 'legacy' in name or 'mainframe' in name or 'ds2' in name:
                    by['legacy'].append(f)
                elif 'target' in name:
                    by['target'].append(f)
                elif 'source' in name or 'relational' in name or 'ds1' in name:
                    by['source'].append(f)
                else:
                    by['uncategorized'].append(f)
        # Ensure defaults appear even if missing in dir
        return {"ok": True, "filesByKind": by, "files": files}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

@poc4_bp.route('/delete_schema', methods=['POST'])
def delete_schema():
    """Delete a schema JSON from the static schemas directory, with safeguards for default demo files."""
    try:
        name = (request.form.get('name') or '').strip()
        if not name:
            return {"ok": False, "error": "name is required."}, 400
        protected = { 'source_schema.json', 'legacy_schema.json', 'target_schema.json' }
        safe = secure_filename(name)
        if safe in protected:
            return {"ok": False, "error": "Deletion of protected demo schema is not allowed."}, 403
        base_dir = _schemas_dir()
        path = os.path.join(base_dir, safe)
        if not os.path.exists(path):
            return {"ok": False, "error": "File not found."}, 404
        os.remove(path)
        # Remove from meta
        meta = _load_meta()
        if safe in meta:
            meta.pop(safe, None)
            _save_meta(meta)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

