from flask import Flask, render_template, request, session
from flask_session import Session
import os
import json
import csv
import re
from jinja2 import ChoiceLoader, FileSystemLoader

# Serve shared root-level static assets (../static) and templates (../templates)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATIC_DIR = os.path.join(ROOT_DIR, 'static')
TEMPLATES_DIR = os.path.join(ROOT_DIR, 'templates')

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='/static', template_folder=TEMPLATES_DIR)
app.secret_key = "your_secret_key"
app.config.update(
    SESSION_TYPE='filesystem',
    SESSION_PERMANENT=False,
    SESSION_FILE_DIR=os.path.join(os.path.dirname(__file__), '.flask_session'),
)
Session(app)

# Prefer POC5 templates, but fall back to root-level templates
POC5_TEMPLATES = os.path.join(os.path.dirname(__file__), 'templates')
app.jinja_loader = ChoiceLoader([
    FileSystemLoader(POC5_TEMPLATES),
    FileSystemLoader(TEMPLATES_DIR),
])

AGENT_PROMPTS = {
    "agent_5_1": "agents/agent_5_1.txt",
    "agent_5_2": "agents/agent_5_2.txt",
    "agent_5_3": "agents/agent_5_3.txt",
    "agent_5_4": "agents/agent_5_4.txt",
    "agent_5_5": "agents/agent_5_5.txt",
}


def sanitize_node_label(text: str) -> str:
    if not text:
        return ""
    t = str(text)
    t = t.replace('[', '(').replace(']', ')')
    t = t.replace('(', '﹙').replace(')', '﹚')
    return t if len(t) <= 80 else (t[:77] + '...')


def _safe_set_session(key: str, value: str | None):
    if value and not session.get(key):
        session[key] = value


def _detect_first(text: str, patterns: list[tuple[str, str]]):
    if not text:
        return None
    low = text.lower()
    for rx, canonical in patterns:
        try:
            if re.search(rx, low, re.I):
                return canonical
        except re.error:
            continue
    return None


def _extract_simple_fields_from_prd(prd_text: str) -> dict:
    fields = {}
    fields['architecture_style'] = _detect_first(prd_text, [
        (r"microservice|micro-service|microservices", "Microservices"),
        (r"event[- ]driven|eda\b", "Event-Driven"),
        (r"serverless|functions?\s+as\s+a\s+service|lambda", "Serverless"),
        (r"monolith(ic)?", "Monolith"),
        (r"soa\b|service[- ]oriented", "SOA"),
        (r"hexagonal|ports\s+and\s+adapters", "Hexagonal"),
        (r"cqrs|event\s+sourcing", "CQRS / Event Sourcing"),
    ])
    fields['cloud_provider'] = _detect_first(prd_text, [
        (r"\baws\b|amazon web services", "AWS"),
        (r"\bazure\b|microsoft azure", "Azure"),
        (r"gcp|google cloud", "GCP"),
        (r"oracle cloud|oci\b", "Oracle Cloud"),
        (r"on[- ]prem|on[- ]premises|data center", "On-Premises"),
    ])
    fields['region_strategy'] = _detect_first(prd_text, [
        (r"multi[- ]region|active[- ]active", "Multi-Region / Active-Active"),
        (r"dr\b|disaster recovery|active[- ]passive", "Active-Passive with DR"),
        (r"single[- ]region", "Single Region"),
    ])
    fields['deployment_model'] = _detect_first(prd_text, [
        (r"kubernetes|k8s|eks|aks|gke", "Kubernetes"),
        (r"ecs|fargate", "ECS/Fargate"),
        (r"vm\b|virtual machine|ec2", "VMs"),
        (r"serverless|lambda|functions?", "Serverless"),
    ])
    fields['database_family'] = _detect_first(prd_text, [
        (r"postgres|mysql|sql server|oracle\b|mariadb", "Relational"),
        (r"mongo|dynamo|cassandra|cosmos|couch", "NoSQL"),
        (r"warehouse|snowflake|redshift|bigquery", "Warehouse"),
    ])
    fields['messaging_backbone'] = _detect_first(prd_text, [
        (r"kafka", "Kafka"),
        (r"rabbitmq|amqp", "RabbitMQ"),
        (r"sqs|sns|eventbridge", "AWS Messaging"),
        (r"pub/sub|pubsub", "Pub/Sub"),
    ])
    fields['identity_provider'] = _detect_first(prd_text, [
        (r"okta", "Okta"),
        (r"auth0", "Auth0"),
        (r"azure\s+ad|entra id", "Azure AD"),
        (r"cognito", "Amazon Cognito"),
        (r"keycloak", "Keycloak"),
    ])
    fields['api_style'] = _detect_first(prd_text, [
        (r"\brest\b|http api|json api", "REST"),
        (r"graphql", "GraphQL"),
        (r"grpc", "gRPC"),
        (r"soap|wsdl|xml[- ]rpc", "SOAP/XML"),
    ])
    fields['caching_strategy'] = _detect_first(prd_text, [
        (r"redis|memcached|cache", "Cache-Aside / Redis"),
        (r"cdn|cloudfront|akamai", "CDN"),
    ])
    fields['observability_stack'] = _detect_first(prd_text, [
        (r"prometheus|grafana|otel|open[- ]telemetry", "Prometheus/Grafana + OpenTelemetry"),
        (r"elk|elastic(searc)?h|logstash|kibana", "ELK Stack"),
        (r"datadog|new relic|splunk", "APM (Datadog/New Relic/Splunk)"),
    ])
    fields['data_residency'] = _detect_first(prd_text, [
        (r"gdpr|eu[- ]only|within eu", "EU / GDPR"),
        (r"india|in[- ]country", "India"),
        (r"us[- ]only|within us", "US"),
    ])
    comps = []
    for name, rx in [
        ("GDPR", r"gdpr"), ("HIPAA", r"hipaa"), ("PCI DSS", r"pci(\s*dss)?"),
        ("SOC 2", r"soc\s*2"), ("ISO 27001", r"iso\s*27001")
    ]:
        if re.search(rx, prd_text or '', re.I):
            comps.append(name)
    if comps:
        fields['compliance_standards'] = ', '.join(sorted(set(comps)))
    m = re.search(r"(?:non[- ]functional|nfr|requirements?)[:\-]?\s*(.+)\n", prd_text or '', re.I)
    if m:
        fields['additional_requirements'] = m.group(1)[:200]
    return {k: v for k, v in fields.items() if v}


def _extract_core_sections_from_prd(prd_text: str) -> dict:
    text = prd_text or ''
    sections = {'business_goals': '', 'constraints': '', 'legacy_system': ''}
    patterns = {
        'business_goals': r"(?i)^(?:business\s+goals|objectives|goals)\b",
        'constraints': r"(?i)^(?:constraints|non[- ]functional\s+requirements|nfrs?)\b",
        'legacy_system': r"(?i)^(?:legacy\s+(?:systems?|context)|existing\s+system)\b",
    }
    lines = text.splitlines()
    n = len(lines)
    idxs = {}
    for i, line in enumerate(lines):
        for key, rx in patterns.items():
            if key in idxs:
                continue
            if re.match(rx, line.strip()):
                idxs[key] = i
    def grab(start: int) -> str:
        buf = []
        for j in range(start+1, min(start+9, n)):
            l = lines[j]
            if re.match(r"^\s*#{1,6}\s+.+$", l):
                break
            if re.match(r"^\S[^:]{0,40}:\s*$", l):
                break
            buf.append(l)
        return "\n".join(buf).strip()
    for key, start in idxs.items():
        val = grab(start)
        if val:
            sections[key] = val
    return sections


def _top_services_from_decisions(text: str, max_n: int = 5) -> list[str]:
    items = []
    for line in (text or '').splitlines():
        m = re.match(r"^\s*[-*]\s*(.+)$", line)
        if not m:
            continue
        name = re.sub(r"[^A-Za-z0-9 ]+", '', m.group(1)).strip()
        if not name:
            continue
        words = name.split()
        short = ''.join(w.capitalize() for w in words[:3])
        if short and short not in items:
            items.append(short)
            if len(items) >= max_n:
                break
    return items


def generate_blueprint_from_session() -> str:
    style = (session.get('architecture_style') or session.get('desired_architecture_style') or '').strip()
    cloud = (session.get('cloud_provider') or '').strip()
    region = (session.get('region_strategy') or '').strip()
    deploy = (session.get('deployment_model') or '').strip()
    dbfam = (session.get('database_family') or '').strip()
    msg = (session.get('messaging_backbone') or '').strip()
    idp = (session.get('identity_provider') or '').strip()
    api_style = (session.get('api_style') or '').strip()
    caching = (session.get('caching_strategy') or '').strip()
    observability = (session.get('observability_stack') or '').strip()

    svc_names = _top_services_from_decisions(session.get('high_level_decision_points', '')) or [
        'CoreServiceA', 'CoreServiceB', 'CoreServiceC'
    ]

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
    db_labels = ['Relational DB','NoSQL DB','Warehouse'] if not dbfam else [dbfam + ' - Primary','Cache','Analytics']
    for i, lbl in enumerate(db_labels, start=1):
        mermaid_lines.append(f"DB{i}(({sanitize_node_label(lbl)})):::db")
    mermaid_lines.append("end")

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
    mermaid_lines.append("EXT_PLACEHOLDER[External Systems]:::ext")
    mermaid_lines.append("end")

    # Core flows
    mermaid_lines.append("User-->APIGW")
    if idp:
        mermaid_lines.append("IDP-->|Auth|APIGW")
    for i in range(1, len(svc_names)+1):
        mermaid_lines.append(f"APIGW-->S{i}")
    for i in range(1, min(len(svc_names), 3)+1):
        mermaid_lines.append(f"S{i}-->DB{i}")
    for i in range(1, len(svc_names)+1):
        mermaid_lines.append(f"S{i}-->|pub/sub|BUS")
    if caching and caching.lower() != 'none':
        for i in range(1, len(svc_names)+1):
            mermaid_lines.append(f"S{i}-. get/set .->CACHE")
    if observability:
        mermaid_lines.append("subgraph Observability")
        mermaid_lines.append(f"OBS[{sanitize_node_label(observability)}]")
        mermaid_lines.append("end")
        mermaid_lines.append("classDef obs fill:#fffde7,stroke:#f9a825,color:#f57f17")
        mermaid_lines.append("class OBS obs")
        for i in range(1, len(svc_names)+1):
            mermaid_lines.append(f"S{i}-. |traces/metrics| .->OBS")

    if cloud or region or deploy:
        mermaid_lines.append(f"%% Cloud: {cloud or 'n/a'} | Region: {region or 'n/a'} | Deploy: {deploy or 'n/a'}")

    mermaid = "\n".join(mermaid_lines)

    # Integrate mapping nodes/edges
    try:
        mapping = json.loads(session.get('system_mapping', '{}'))
    except Exception:
        mapping = {}
    nodes = mapping.get('nodes') or []
    edges = mapping.get('edges') or []
    if nodes or edges:
        mermaid += "\n%% System Mapping"

    alias_map: dict[str, str] = {}
    shown_nodes_limit = 40
    shown_edges_limit = 40

    def get_alias(name: str) -> str:
        if name in alias_map:
            return alias_map[name]
        alias = f"N{len(alias_map)+1}"
        alias_map[name] = alias
        return alias

    def sanitize_label(text: str) -> str:
        if not text:
            return ""
        t = str(text)
        t = t.replace('|', '/')
        if len(t) > 80:
            t = t[:77] + '...'
        return t

    ext_nodes = nodes[:shown_nodes_limit]
    if ext_nodes:
        extra = []
        for n in ext_nodes:
            alias = get_alias(n)
            extra.append(f"\n{alias}[{sanitize_node_label(n)}]:::ext")
        mermaid = mermaid.replace(
            "EXT_PLACEHOLDER[External Systems]:::ext",
            "EXT_SUMMARY[External Systems]:::ext" + "".join(extra)
        )

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
        mermaid += f"\n{sa}-->{label_part}{ta}"

    if alias_map:
        mermaid += "\nIL[Integration Layer]:::svc"
        for orig, alias in list(alias_map.items())[:min(len(alias_map), 10)]:
            mermaid += f"\n{alias}-->IL"
        for i in range(1, len(svc_names)+1):
            mermaid += f"\nIL-->|adapters|S{i}"

    mermaid += "\n%% Auto-generated initial blueprint\n%% Style: {}".format(style or 'unspecified')
    return mermaid


def _ensure_default_dropdowns():
    defaults = {
        'cloud_provider': 'AWS',
        'region_strategy': 'Single Region',
        'deployment_model': 'Serverless',
        'database_family': 'Relational',
        'messaging_backbone': 'Kafka',
        'identity_provider': 'Okta',
        'api_style': 'REST',
        'caching_strategy': 'Cache-Aside / Redis',
        'observability_stack': 'Prometheus/Grafana + OpenTelemetry',
        'data_residency': 'US',
    }
    for k, v in defaults.items():
        _safe_set_session(k, v)


def call_agent(agent_key, context):
    with open(os.path.join(ROOT_DIR, AGENT_PROMPTS[agent_key]), "r", encoding="utf-8") as f:
        prompt = f.read()
    return f"{prompt}\n\n{context}"


def read_text_safely(file_path: str) -> str:
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
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return data.decode('utf-8', errors='ignore')
    except Exception:
        return ''


def parse_system_mapping(file_path: str):
    nodes = set()
    edges = []
    _, ext = os.path.splitext(file_path.lower())
    try:
        if ext == '.csv':
            with open(file_path, newline='', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
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
    try:
        import openai
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


def parse_extracted_decisions(md: str) -> dict:
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
            if not sections['style']:
                sections['style'] = l
    return sections


@app.route('/', methods=['GET', 'POST'])
def tabbed_workbench():
    active_tab = 0
    if request.method == 'POST':
        action = request.form.get('action', 'save')
        for key in request.form:
            if key == 'action':
                continue
            session[key] = request.form.get(key, '')
        # PRD upload
        prd_file = request.files.get('prd_file')
        if prd_file and prd_file.filename:
            uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            safe_name = prd_file.filename.replace('..','_').replace('/','_').replace('\\','_')
            file_path = os.path.join(uploads_dir, safe_name)
            prd_file.save(file_path)
            session['prd_file_name'] = safe_name
            session['prd_text'] = read_text_safely(file_path)
        # Mapping upload
        sys_map = request.files.get('system_mapping_file')
        if sys_map and sys_map.filename:
            uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            safe_name = sys_map.filename.replace('..','_').replace('/','_').replace('\\','_')
            map_path = os.path.join(uploads_dir, safe_name)
            sys_map.save(map_path)
            session['system_mapping_name'] = safe_name
            session['system_mapping'] = json.dumps(parse_system_mapping(map_path))
        # Lock Tab 1 when both present
        if session.get('prd_file_name') and session.get('system_mapping_name') and action != 'unlock_tab1':
            session['lock_tab1'] = '1'

        # Actions
        if action == 'extract_decisions':
            prd_text = session.get('prd_text', '')
            hints = {
                'business_goals': session.get('business_goals',''),
                'legacy_system': session.get('legacy_system',''),
                'constraints': session.get('constraints',''),
            }
            extracted = extract_decisions_with_llm(prd_text, hints)
            session['high_level_decision_points'] = extracted
            try:
                sections = parse_extracted_decisions(extracted)
                if sections.get('high_level'):
                    session['high_level_decision_points'] = '\n'.join(sections['high_level'])
                if sections.get('one_way'):
                    session['one_way_door_decisions'] = '\n'.join(sections['one_way'])
                if sections.get('style'):
                    session['desired_architecture_style'] = sections['style']
                    if not session.get('architecture_style'):
                        session['architecture_style'] = sections['style']
                if sections.get('other_qs'):
                    session['other_relevant_questions'] = '\n'.join(sections['other_qs'])
            except Exception:
                pass
            active_tab = 0
        elif action == 'lock_tab1':
            session['lock_tab1'] = '1'
            active_tab = 0
        elif action == 'unlock_tab1':
            session['lock_tab1'] = ''
            active_tab = 0
        elif action == 'generate_initial_blueprint':
            session['architecture_diagram'] = generate_blueprint_from_session()
            active_tab = 1
        else:
            active_tab = 0

    # Auto-populate once both uploads available
    if session.get('prd_text') and session.get('system_mapping'):
        prd_text = session.get('prd_text', '')
        hints = {
            'business_goals': session.get('business_goals',''),
            'legacy_system': session.get('legacy_system',''),
            'constraints': session.get('constraints',''),
        }
        extracted = extract_decisions_with_llm(prd_text, hints)
        session['high_level_decision_points'] = extracted
        try:
            sections = parse_extracted_decisions(extracted)
            if sections.get('high_level'):
                session['high_level_decision_points'] = '\n'.join(sections['high_level'])
            if sections.get('one_way'):
                session['one_way_door_decisions'] = '\n'.join(sections['one_way'])
            if sections.get('style'):
                _safe_set_session('desired_architecture_style', sections['style'])
                if not session.get('architecture_style'):
                    session['architecture_style'] = sections['style']
            if sections.get('other_qs'):
                session['other_relevant_questions'] = '\n'.join(sections['other_qs'])
        except Exception:
            pass

        for k, v in _extract_simple_fields_from_prd(prd_text).items():
            _safe_set_session(k, v)
        core = _extract_core_sections_from_prd(prd_text)
        for k in ['business_goals','constraints','legacy_system']:
            _safe_set_session(k, core.get(k))

        # Mapping-derived content
        try:
            mapping = json.loads(session.get('system_mapping','{}'))
        except Exception:
            mapping = {}
        nodes = mapping.get('nodes') or []
        edges = mapping.get('edges') or []
        svc_names = _top_services_from_decisions(session.get('high_level_decision_points','')) or []
        comp_lines = [f"- {n}" for n in svc_names] + [f"- External: {n}" for n in nodes[:20]]
        _safe_set_session('major_components', '\n'.join(comp_lines) or 'Auto from PRD/Mapping')
        iface_lines = []
        for e in edges[:50]:
            s = e[0] if len(e) > 0 else ''
            t = e[1] if len(e) > 1 else ''
            lbl = e[2] if len(e) > 2 else None
            if s and t:
                iface_lines.append(f"- {s} -> {t}" + (f" [{lbl}]" if lbl else ''))
        _safe_set_session('interface_definitions', '\n'.join(iface_lines) or 'Auto from Mapping')
        _safe_set_session('data_schemas', '\n'.join([f"- {n}" for n in nodes[:30]]) or 'Entities from Mapping')

        patt = []
        if session.get('architecture_style','').lower().startswith('micro'):
            patt += ['- Circuit Breaker', '- Saga / Orchestration', '- API Gateway']
        if session.get('messaging_backbone'):
            patt += ['- Event-Driven (Pub/Sub)', '- Outbox / Idempotency']
        if session.get('caching_strategy'):
            patt += ['- Cache-Aside']
        _safe_set_session('reusable_patterns', '\n'.join(patt) or '- Layered Architecture')

        _safe_set_session('architectural_decisions', extracted)
        style = session.get('architecture_style') or session.get('desired_architecture_style') or 'Unspecified'
        dbfam = session.get('database_family') or 'Relational/NoSQL'
        _safe_set_session('rationale_tradeoffs', f"Chose {style} with {dbfam} storage based on PRD goals and constraints.")
        _safe_set_session('blueprint_references', f"PRD: {session.get('prd_file_name','')} | Mapping: {session.get('system_mapping_name','')}")

        api_style_upper = (session.get('api_style') or '').upper()
        comm = 'HTTP/REST + JSON' if api_style_upper == 'REST' else ('gRPC' if api_style_upper == 'GRPC' else ('GraphQL' if api_style_upper == 'GRAPHQL' else 'HTTP APIs'))
        _safe_set_session('communication_protocols', comm)

        _safe_set_session('data_flow_diagrams', 'See generated Blueprint; flows inferred from Mapping edges.')
        _safe_set_session('integration_points', '\n'.join([f"- {n}" for n in nodes[:10]]) or 'Derived from Mapping')

        pros = ['- Scalability potential', '- Clear service boundaries']
        cons = ['- Operational complexity', '- Distributed tracing required']
        _safe_set_session('swot_analysis', '\n'.join(['Pros:'] + pros + ['','Cons:'] + cons))
        risks = ['- Compliance and data residency', '- Message ordering/duplication'] if session.get('messaging_backbone') else ['- Compliance and data residency']
        _safe_set_session('risks_mitigation', '\n'.join(["Risks:"] + [f"- {r}" for r in risks] + ["","Mitigations:","- Automate controls & monitoring","- Adopt Idempotency & retries","- Backup/DR runbooks"]))

        excerpt = (prd_text or '')[:500]
        _safe_set_session('compiled_document', f"Auto-extracted from PRD and Mapping.\n\nExcerpt:\n{excerpt}")
        _safe_set_session('stakeholder_checklist', '\n'.join([
            '- Security sign-off', '- Compliance sign-off', '- SRE runbooks', '- Architecture review'
        ]))

        session['architecture_diagram'] = generate_blueprint_from_session()
        _ensure_default_dropdowns()

    return render_template(
        'tabbed_architecture_workbench.html',
        active_tab=active_tab,
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
        blueprint=session.get('architecture_diagram', ''),
        major_components=session.get('major_components',''),
        interface_definitions=session.get('interface_definitions',''),
        data_schemas=session.get('data_schemas',''),
        reusable_patterns=session.get('reusable_patterns',''),
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


if __name__ == "__main__":
    app.run(port=6001, debug=True)
