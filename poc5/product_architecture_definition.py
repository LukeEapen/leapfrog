
from flask import Flask, render_template, request, session
from flask_session import Session
import os
import json
import csv
import re
# Optional heavy deps will be imported inside helper functions to avoid static import errors

# Serve shared root-level static assets (../static) so theme CSS/images are available
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATIC_DIR = os.path.join(ROOT_DIR, 'static')
app = Flask(
    __name__,
    static_folder=STATIC_DIR,       # points to repo-level /static
    static_url_path='/static'       # keep URL paths the same
)
app.secret_key = "your_secret_key"
# Use server-side sessions to avoid oversized cookie warnings when storing PRD text/diagrams
app.config.update(
    SESSION_TYPE='filesystem',
    SESSION_PERMANENT=False,
    SESSION_FILE_DIR=os.path.join(os.path.dirname(__file__), '.flask_session'),
)
Session(app)

AGENT_PROMPTS = {
    "agent_5_1": "agents/agent_5_1.txt",
    "agent_5_2": "agents/agent_5_2.txt",
    "agent_5_3": "agents/agent_5_3.txt",
    "agent_5_4": "agents/agent_5_4.txt",
    "agent_5_5": "agents/agent_5_5.txt",
}

def sanitize_node_label(text: str) -> str:
    """Sanitize a string for safe display in Mermaid node labels."""
    if not text:
        return ""
    t = str(text)
    # Avoid Mermaid bracket nesting issues
    t = t.replace('[', '(').replace(']', ')')
    # Replace parentheses with full-width to avoid node shape parsing inside labels
    t = t.replace('(', '﹙').replace(')', '﹚')
    # Trim overly long labels
    return t if len(t) <= 80 else (t[:77] + '...')

def call_agent(agent_key, context):
    with open(os.path.join(os.path.dirname(__file__), AGENT_PROMPTS[agent_key]), "r", encoding="utf-8") as f:
        prompt = f.read()
    return f"{prompt}\n\n{context}"

def read_text_safely(file_path: str) -> str:
    """Extract text from file_path. Supports .txt, .md, .docx, .pdf. Fallback to binary read."""
    _, ext = os.path.splitext(file_path.lower())
    try:
        if ext in ('.txt', '.md', '.json'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        if ext == '.docx':
            try:
                import docx as docx_mod
                d = docx_mod.Document(file_path)
                return '\n'.join(p.text for p in d.paragraphs)
            except Exception:
                pass
        if ext == '.pdf':
            try:
                import importlib
                pdf_high = importlib.import_module('pdfminer.high_level')
                return getattr(pdf_high, 'extract_text')(file_path) or ''
            except Exception:
                pass
    except Exception:
        pass
    # Fallback
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return data.decode('utf-8', errors='ignore')
    except Exception:
        return ''

def parse_system_mapping(file_path: str):
    """Parse a system mapping file into nodes and edges.
    Supports:
    - CSV with headers: source,target[,label]
    - JSON with { nodes:[{id,label?}], edges:[{source,target,label?}] }
    - TXT as adjacency list lines: A->B, B->C (comma/semicolon separated)
    Returns dict: { nodes:set(str), edges:list[(src,dst,label?)] }
    """
    nodes = set()
    edges = []
    _, ext = os.path.splitext(file_path.lower())
    try:
        if ext == '.csv':
            with open(file_path, newline='', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                # Fallback if no headers
                if reader.fieldnames is None:
                    f.seek(0)
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 2:
                            s, t = row[0].strip(), row[1].strip()
                            if s and t:
                                nodes.update([s, t])
                                edges.append((s, t, None))
                else:
                    for row in reader:
                        s = (row.get('source') or row.get('from') or '').strip()
                        t = (row.get('target') or row.get('to') or '').strip()
                        if not s and len(row) >= 1:
                            # try positional
                            vals = list(row.values())
                            s = (vals[0] or '').strip()
                            t = (vals[1] or '').strip() if len(vals) > 1 else ''
                        if s and t:
                            label = (row.get('label') or '').strip() or None
                            nodes.update([s, t])
                            edges.append((s, t, label))
        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            for n in data.get('nodes', []):
                nid = (n.get('id') if isinstance(n, dict) else str(n)).strip()
                if nid:
                    nodes.add(nid)
            for e in data.get('edges', []):
                s = (e.get('source') or e.get('from') or '').strip()
                t = (e.get('target') or e.get('to') or '').strip()
                if s and t:
                    edges.append((s, t, (e.get('label') or None)))
                    nodes.update([s, t])
        else:
            text = read_text_safely(file_path)
            # split lines, accept A->B pairs per token
            for line in text.splitlines():
                parts = [p.strip() for p in line.replace(';', ',').split(',') if p.strip()]
                for token in parts:
                    if '->' in token:
                        s, t = [x.strip() for x in token.split('->', 1)]
                        if s and t:
                            nodes.update([s, t])
                            edges.append((s, t, None))
    except Exception:
        pass
    # Truncate if very large for UI stats
    truncated = False
    MAX_EDGES = 300
    if len(edges) > MAX_EDGES:
        edges = edges[:MAX_EDGES]
        truncated = True
    return {
        'nodes': sorted(nodes),
        'edges': edges,
        'stats': {
            'node_count': len(nodes),
            'edge_count': len(edges),
            'truncated': truncated,
        }
    }

def extract_decisions_with_llm(prd_text: str, context_hints: dict | None = None) -> str:
    """Use Agent 5_2 prompt to extract decisions; falls back if OpenAI not available."""
    system_prompt = (
        "You are an Architecture Decision Extractor. Read the PRD text and extract:\n"
        "- High-Level Architecture Decision Points (bulleted)\n"
        "- One-way door decisions (irreversible or costly-to-reverse) (bulleted)\n"
        "- Desired Architecture Style (single line)\n"
        "- Other Relevant Questions (bulleted)\n"
        "Return sections with clear, markdown headings: '# High-Level Architecture Decision Points', '# One-way Door Decisions', '# Desired Architecture Style', '# Other Relevant Questions'."
    )
    user_payload = prd_text or ''
    if context_hints:
        try:
            user_payload = json.dumps({'prd': prd_text, 'hints': context_hints})
        except Exception:
            pass
    # Try OpenAI
    try:
        import openai
        # Use same model pattern as repo
        chat = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        return chat.choices[0].message.content
    except Exception:
        # Fallback: reuse local agent prompt concatenation or simple heuristics
        try:
            return call_agent('agent_5_2', system_prompt + "\n\n" + user_payload)
        except Exception:
            head = (prd_text or '')[:2000]
            return (
                "# High-Level Architecture Decision Points\n"
                "- Review data domains\n- Identify SLAs and latency requirements\n\n"
                "# One-way Door Decisions\n- Choose database family\n- Pick messaging backbone\n\n"
                "# Desired Architecture Style\n- TBD\n\n"
                "# Other Relevant Questions\n- Compliance constraints?\n- Multi-region needs?\n\n---\nExcerpt:\n" + head
            )

@app.route('/', methods=['GET', 'POST'])
def tabbed_workbench():
    active_tab = 0
    # Handle form submission across tabs with file upload support
    if request.method == 'POST':
        action = request.form.get('action', 'save')
        # Persist simple form fields
        for key in request.form:
            if key == 'action':
                continue
            session[key] = request.form.get(key, '')
        # Handle PRD document upload
        prd_file = request.files.get('prd_file')
        if prd_file and prd_file.filename:
            uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            safe_name = prd_file.filename.replace('..','_').replace('/','_').replace('\\','_')
            file_path = os.path.join(uploads_dir, safe_name)
            prd_file.save(file_path)
            session['prd_file_name'] = safe_name
            # Extract text using proper parser
            prd_text = read_text_safely(file_path)
            session['prd_text'] = prd_text

        # Handle System Mapping upload
        sys_map = request.files.get('system_mapping_file')
        if sys_map and sys_map.filename:
            uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            safe_name = sys_map.filename.replace('..','_').replace('/','_').replace('\\','_')
            map_path = os.path.join(uploads_dir, safe_name)
            sys_map.save(map_path)
            session['system_mapping_name'] = safe_name
            parsed = parse_system_mapping(map_path)
            session['system_mapping'] = json.dumps(parsed)

        # Auto-lock Tab 1 once both uploads are present (can be unlocked manually)
        if session.get('prd_file_name') and session.get('system_mapping_name') and action != 'unlock_tab1':
            session['lock_tab1'] = '1'

        # Optional: decision extraction from PRD using agent
        if action == 'extract_decisions':
            prd_text = session.get('prd_text', '')
            hints = {
                'business_goals': session.get('business_goals',''),
                'legacy_system': session.get('legacy_system',''),
                'constraints': session.get('constraints',''),
            }
            extracted = extract_decisions_with_llm(prd_text, hints)
            # Try to parse sections to populate fields
            session['high_level_decision_points'] = extracted
            try:
                sections = parse_extracted_decisions(extracted)
                if sections.get('high_level'):
                    session['high_level_decision_points'] = '\n'.join(sections['high_level'])
                if sections.get('one_way'):
                    session['one_way_door_decisions'] = '\n'.join(sections['one_way'])
                if sections.get('style'):
                    session['desired_architecture_style'] = sections['style']
                    # Also sync the primary dropdown if empty
                    if not session.get('architecture_style'):
                        session['architecture_style'] = sections['style']
                if sections.get('other_qs'):
                    session['other_relevant_questions'] = '\n'.join(sections['other_qs'])
            except Exception:
                pass
            # Keep user on tab 1 to review questions
            active_tab = 0
        elif action == 'lock_tab1':
            session['lock_tab1'] = '1'
            active_tab = 0
        elif action == 'unlock_tab1':
            session['lock_tab1'] = ''
            active_tab = 0
        elif action == 'generate_initial_blueprint':
            # Build a very simple initial diagram from available context
            business = (session.get('business_goals','') or '').strip()
            style = (session.get('architecture_style','') or '').strip() or (session.get('desired_architecture_style','') or '')
            legacy = (session.get('legacy_system','') or '').strip()

            # Selections to drive diagram
            cloud = (session.get('cloud_provider','') or '').strip()
            region = (session.get('region_strategy','') or '').strip()
            deploy = (session.get('deployment_model','') or '').strip()
            dbfam = (session.get('database_family','') or '').strip()
            msg = (session.get('messaging_backbone','') or '').strip()
            idp = (session.get('identity_provider','') or '').strip()
            api_style = (session.get('api_style','') or '').strip()
            caching = (session.get('caching_strategy','') or '').strip()
            observability = (session.get('observability_stack','') or '').strip()

            # Extract top 3-5 service names from decision points (bullets)
            def top_services_from_decisions(text: str, max_n: int = 4):
                items = []
                for line in (text or '').splitlines():
                    m = re.match(r"^\s*[-*]\s*(.+)$", line)
                    if m:
                        name = re.sub(r"[^A-Za-z0-9 ]+", '', m.group(1)).strip()
                        if len(name) > 0:
                            # Condense name to a compact service label
                            words = name.split()
                            short = ''.join(w.capitalize() for w in words[:3])
                            items.append(short)
                            if len(items) >= max_n:
                                break
                return items

            svc_names = top_services_from_decisions(session.get('high_level_decision_points','')) or ['CoreServiceA','CoreServiceB','CoreServiceC']

            # Professional layered diagram with subgraphs and class definitions
            mermaid_lines = []
            mermaid_lines.append("flowchart TB")
            mermaid_lines.append("classDef svc fill:#e3f2fd,stroke:#1976d2,stroke-width:1px,color:#0d47a1")
            mermaid_lines.append("classDef db fill:#fff3cd,stroke:#ff9800,color:#e65100")
            mermaid_lines.append("classDef gw fill:#fce4ec,stroke:#c2185b,color:#880e4f")
            mermaid_lines.append("classDef msg fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20")
            mermaid_lines.append("classDef ext fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c")
            mermaid_lines.append("classDef idp fill:#ede7f6,stroke:#5e35b1,color:#311b92")

            mermaid_lines.append("subgraph Clients")
            mermaid_lines.append("User[User]")
            mermaid_lines.append("end")

            mermaid_lines.append("subgraph Edge")
            apigw_label = "API Gateway"
            if api_style:
                low = api_style.lower()
                if low in ("rest","graphql","grpc"):
                    apigw_label = f"{api_style.upper()} Gateway" if low != 'graphql' else 'GraphQL Gateway'
                else:
                    apigw_label = sanitize_node_label(api_style)
            mermaid_lines.append(f"APIGW[{sanitize_node_label(apigw_label)}]:::gw")
            if idp:
                mermaid_lines.append("IDP[" + sanitize_node_label(idp) + "]:::idp")
            mermaid_lines.append("end")

            mermaid_lines.append("subgraph Services")
            for i, n in enumerate(svc_names, start=1):
                mermaid_lines.append(f"S{i}[{sanitize_node_label(n)}]:::svc")
            mermaid_lines.append("end")

            mermaid_lines.append("subgraph Data")
            # Choose DB icons by family selection
            db_labels = ['Relational DB','NoSQL DB','Warehouse'] if not dbfam else [dbfam + ' - Primary','Cache','Analytics']
            for i, lbl in enumerate(db_labels, start=1):
                mermaid_lines.append(f"DB{i}(({sanitize_node_label(lbl)})):::db")
            mermaid_lines.append("end")

            # Optional caching
            if caching and caching.lower() != 'none':
                mermaid_lines.append("subgraph Caching")
                mermaid_lines.append(f"CACHE(({sanitize_node_label(caching)} Cache))")
                mermaid_lines.append("end")
                mermaid_lines.append("classDef cache fill:#e0f7fa,stroke:#00838f,color:#006064")
                mermaid_lines.append("class CACHE cache")

            mermaid_lines.append("subgraph Messaging")
            if msg:
                mnode = 'Kafka' if 'kafka' in msg.lower() else ('RabbitMQ' if 'rabbit' in msg.lower() else msg)
                mermaid_lines.append(f"BUS(({sanitize_node_label(mnode)})):::msg")
            else:
                mermaid_lines.append("BUS((Event Bus)):::msg")
            mermaid_lines.append("end")

            mermaid_lines.append("subgraph External")
            # External nodes from mapping will be added here later
            mermaid_lines.append("EXT_PLACEHOLDER[External Systems]:::ext")
            mermaid_lines.append("end")

            # Core flows
            mermaid_lines.append("User-->APIGW")
            if idp:
                mermaid_lines.append("IDP-->|Auth|APIGW")
            for i in range(1, len(svc_names)+1):
                mermaid_lines.append(f"APIGW-->S{i}")
            # Service to DB
            for i in range(1, min(len(svc_names), len(db_labels))+1):
                mermaid_lines.append(f"S{i}-->DB{i}")
            # Messaging
            for i in range(1, len(svc_names)+1):
                mermaid_lines.append(f"S{i}-->|pub/sub|BUS")

            # Caching links
            if caching and caching.lower() != 'none':
                for i in range(1, len(svc_names)+1):
                    mermaid_lines.append(f"S{i}-. get/set .->CACHE")

            # Observability
            if observability:
                mermaid_lines.append("subgraph Observability")
                mermaid_lines.append(f"OBS[{sanitize_node_label(observability)}]")
                mermaid_lines.append("end")
                mermaid_lines.append("classDef obs fill:#fffde7,stroke:#f9a825,color:#f57f17")
                mermaid_lines.append("class OBS obs")
                for i in range(1, len(svc_names)+1):
                    mermaid_lines.append(f"S{i}-. |traces/metrics| .->OBS")

            # Optional subtitles/comments
            if cloud or region or deploy:
                mermaid_lines.append(f"%% Cloud: {cloud or 'n/a'} | Region: {region or 'n/a'} | Deploy: {deploy or 'n/a'}")

            mermaid = "\n".join(mermaid_lines)
            # If system mapping present, extend diagram with nodes/edges
            try:
                mapping = json.loads(session.get('system_mapping', '{}'))
            except Exception:
                mapping = {}
            nodes = mapping.get('nodes') or []
            edges = mapping.get('edges') or []
            if nodes or edges:
                mermaid += "\n%% System Mapping"
            # Build safe aliases for nodes to avoid punctuation in Mermaid IDs
            alias_map: dict[str, str] = {}
            shown_nodes_limit = 40
            shown_edges_limit = 40
            def get_alias(name: str) -> str:
                nonlocal alias_map
                if name in alias_map:
                    return alias_map[name]
                alias = f"N{len(alias_map)+1}"
                alias_map[name] = alias
                return alias
            def sanitize_label(text: str) -> str:
                if not text:
                    return ""
                t = str(text)
                # Mermaid link labels cannot contain '|'; replace and trim overly long labels
                t = t.replace('|', '/')
                if len(t) > 80:
                    t = t[:77] + '...'
                return t
            # Pre-assign aliases for a subset of declared nodes for explicit rendering under External subgraph
            ext_nodes = nodes[:shown_nodes_limit]
            if ext_nodes:
                # Inject external nodes within External subgraph block by appending after EXT_PLACEHOLDER
                extra = []
                for n in ext_nodes:
                    alias = get_alias(n)
                    extra.append(f"\n{alias}[{sanitize_node_label(n)}]:::ext")
                mermaid = mermaid.replace(
                    "EXT_PLACEHOLDER[External Systems]:::ext",
                    "EXT_SUMMARY[External Systems]:::ext" + "".join(extra)
                )
            # Add edges and ensure endpoints have aliases
            for e in edges[:shown_edges_limit]:
                s = e[0] if len(e) > 0 else None
                t = e[1] if len(e) > 1 else None
                lbl = e[2] if len(e) > 2 else None
                if not s or not t:
                    continue
                sa = get_alias(s)
                ta = get_alias(t)
                label = sanitize_label(lbl)
                label_part = f"|{label}|" if label else ""
                # Route external edges; preserve correct Mermaid edge syntax
                mermaid += f"\n{sa}-->{label_part}{ta}"
            # Add an Integration Layer node to connect Externals to Services
            mermaid += "\nIL[Integration Layer]:::svc"
            for orig, alias in list(alias_map.items())[:min(len(alias_map), 10)]:
                mermaid += f"\n{alias}-->IL"
            for i in range(1, len(svc_names)+1):
                mermaid += f"\nIL-->|adapters|S{i}"
            # Add legend
            mermaid += "\n%% Auto-generated initial blueprint\n%% Style: {}".format(style or 'unspecified')
            session['architecture_diagram'] = mermaid
            # Switch to tab 2
            active_tab = 1
        else:
            # Default save keeps current tab 1
            active_tab = 0

    return render_template(
        'tabbed_architecture_workbench.html',
        active_tab=active_tab,
        # Context for binding fields
        business_goals=session.get('business_goals',''),
        legacy_system=session.get('legacy_system',''),
        constraints=session.get('constraints',''),
        architecture_style=session.get('architecture_style',''),
        desired_architecture_style=session.get('desired_architecture_style',''),
        cloud_provider=session.get('cloud_provider',''),
        region_strategy=session.get('region_strategy',''),
        deployment_model=session.get('deployment_model',''),
        database_family=session.get('database_family',''),
        messaging_backbone=session.get('messaging_backbone',''),
        identity_provider=session.get('identity_provider',''),
    api_style=session.get('api_style',''),
    caching_strategy=session.get('caching_strategy',''),
    observability_stack=session.get('observability_stack',''),
        data_residency=session.get('data_residency',''),
        compliance_standards=session.get('compliance_standards',''),
        additional_requirements=session.get('additional_requirements',''),
        prd_file_name=session.get('prd_file_name',''),
        prd_text=session.get('prd_text',''),
    system_mapping_name=session.get('system_mapping_name',''),
    system_mapping_stats=json.loads(session.get('system_mapping','{}')).get('stats') if session.get('system_mapping') else None,
        high_level_decision_points=session.get('high_level_decision_points',''),
        one_way_door_decisions=session.get('one_way_door_decisions',''),
        other_relevant_questions=session.get('other_relevant_questions',''),
        # Tab 2
        blueprint=session.get('architecture_diagram', ''),
        major_components=session.get('major_components',''),
        interface_definitions=session.get('interface_definitions',''),
        data_schemas=session.get('data_schemas',''),
        reusable_patterns=session.get('reusable_patterns',''),
        # Tab 3+
        decisions=session.get('architectural_decisions', ''),
        rationale_tradeoffs=session.get('rationale_tradeoffs',''),
        blueprint_references=session.get('blueprint_references',''),
        communication=session.get('communication_protocols', ''),
        data_flow_diagrams=session.get('data_flow_diagrams',''),
        integration_points=session.get('integration_points',''),
        pros_cons=session.get('swot_analysis', ''),
        risks_mitigation=session.get('risks_mitigation',''),
        doc=session.get('compiled_document', ''),
        stakeholder_checklist=session.get('stakeholder_checklist',''),
        lock_tab1=session.get('lock_tab1','')
    )

def parse_extracted_decisions(md: str) -> dict:
    """Parse markdown sections from extractor into dict lists and strings."""
    sections = {'high_level': [], 'one_way': [], 'style': '', 'other_qs': []}
    if not md:
        return sections
    cur = None
    for line in md.splitlines():
        l = line.strip()
        if re.match(r"^#\s*High-Level Architecture Decision Points", l, re.I):
            cur = 'high_level'; continue
        if re.match(r"^#\s*One-way Door Decisions", l, re.I):
            cur = 'one_way'; continue
        if re.match(r"^#\s*Desired Architecture Style", l, re.I):
            cur = 'style'; continue
        if re.match(r"^#\s*Other Relevant Questions", l, re.I):
            cur = 'other_qs'; continue
        if cur in ('high_level','one_way','other_qs'):
            m = re.match(r"^[-*]\s*(.+)$", l)
            if m:
                sections[cur].append(m.group(1).strip())
        elif cur == 'style' and l:
            # First non-empty line under style
            if not sections['style']:
                sections['style'] = l
    return sections

if __name__ == "__main__":
    app.run(port=6001, debug=True)
