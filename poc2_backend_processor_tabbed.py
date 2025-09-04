#!/usr/bin/env python3
"""
Tabbed Layout Backend Processor - CLEANED
Removed duplicated/truncated sections that caused IndentationError.
Single authoritative implementation with enforced gpt-4o usage.
"""

from dotenv import load_dotenv
load_dotenv()

import os, sys, logging, traceback, json, tempfile, csv, io, time, re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
from openai import OpenAI
from werkzeug.utils import secure_filename
import docx, PyPDF2

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    logging.warning("tiktoken not available - using fallback token counting")
    TIKTOKEN_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('tabbed_backend.log'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

ALLOWED_MODEL = "gpt-4o"

# ---------------- Token & Model Utilities ----------------

def count_tokens(text, model=ALLOWED_MODEL):
    if not TIKTOKEN_AVAILABLE:
        return len(text)//4
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return len(text)//4

def choose_model(_purpose:str)->str:
    return ALLOWED_MODEL

def safe_chat_completion(client, messages, purpose:str, max_tokens=1000, temperature=0.7):
    return client.chat.completions.create(model=choose_model(purpose), messages=messages, max_tokens=max_tokens, temperature=temperature)

# ---------------- Flask App Setup ----------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config.update(
    SESSION_TYPE='filesystem', SESSION_PERMANENT=True, PERMANENT_SESSION_LIFETIME=3600,
    SESSION_USE_SIGNER=True, SESSION_KEY_PREFIX='tabbed_workbench:',
    SESSION_FILE_DIR=os.path.join(tempfile.gettempdir(), 'tabbed_workbench_sessions')
)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
Session(app)

# ---------------- File Utilities ----------------
ALLOWED_EXTENSIONS = {'txt','pdf','docx','md','csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def safe_read(file_obj):
    try:
        if not file_obj: return ""
        filename = secure_filename(file_obj.filename)
        ext = filename.rsplit('.',1)[1].lower() if '.' in filename else ''
        if ext in ['txt','md','csv']:
            return file_obj.read().decode('utf-8', errors='ignore').strip()
        if ext == 'docx':
            d = docx.Document(file_obj)
            return '\n'.join(p.text for p in d.paragraphs).strip()
        if ext == 'pdf':
            reader = PyPDF2.PdfReader(file_obj)
            pages = []
            for p in reader.pages:
                try: pages.append(p.extract_text() or '')
                except Exception: pass
            return '\n'.join(pages).strip()
        return file_obj.read().decode('utf-8', errors='ignore').strip()
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return ""

# ---------------- Basic Routes ----------------
@app.route('/')
def landing_page():
    return render_template('landing_page.html')

@app.route('/tabbed-layout')
def tabbed_layout():
    try:
        if request.args.get('clear_session') == 'true':
            session.clear(); session.modified = True
        return render_template('poc2_tabbed_workbench.html')
    except Exception as e:
        logger.error(f"Error loading layout: {e}")
        return f"Error: {e}", 500

@app.route('/health')
def health():
    return jsonify({"status":"healthy","timestamp":datetime.now().isoformat(),"openai_configured":bool(os.getenv('OPENAI_API_KEY'))})

@app.route('/debug-info')
def debug_info():
    d = app.config['SESSION_FILE_DIR']
    files = os.listdir(d) if os.path.exists(d) else []
    return jsonify({
        "environment": os.environ.get('FLASK_ENV','development'),
        "openai_key_set": bool(os.getenv('OPENAI_API_KEY')),
        "session_directory": d,
        "session_files_count": len(files),
        "session_data": {
            "keys": list(session.keys()),
            "epics_count": len(session.get('generated_epics', [])),
            "current_epic": session.get('current_epic', {}).get('title','None')
        }
    })

@app.route('/debug-session')
def debug_session():
    try:
        epics = session.get('generated_epics', [])
        return jsonify({"session_id": request.cookies.get('session','None'),"epics_count": len(epics),"keys": list(session.keys()),"epic_ids": [e.get('id') for e in epics]})
    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------- Epic Generation Context ----------------

def create_epic_generation_context(prd_content, additional_content="", context_notes=""):
    ctx = f"""
You are an expert product manager. Based on the PRD generate 3-5 comprehensive epics.
PRD Content:\n{prd_content}
""".strip()
    if additional_content:
        ctx += f"\n\nAdditional Documentation:\n{additional_content}"
    if context_notes:
        ctx += f"\n\nUser Context & Instructions:\n{context_notes}"
    ctx += """
\nFormat:
Epic 1: <Title>
Description: <Detailed description>
Priority: High/Medium/Low

Repeat for 3-5 epics (significant capability, independent value, clear scope).
"""
    return ctx

def create_epic_generation_context_with_system_mapping(prd_content, system_mapping, context_notes=""):
    return create_epic_generation_context(prd_content, "", context_notes) + f"\n\nSystem Mapping Content:\n{system_mapping}\n(Incorporate architecture & integration points.)"

# ---------------- Upload & Epic Generation ----------------
@app.route('/tabbed-upload-files', methods=['POST'])
def tabbed_upload_files():
    try:
        prd_file = request.files.get('prd_file')
        sys_file = request.files.get('system_mapping_file')
        context_notes = request.form.get('context_notes','')
        if not prd_file:
            return jsonify({"success": False, "error": "PRD file is required"})
        prd_content = safe_read(prd_file)
        if not prd_content:
            return jsonify({"success": False, "error": "Could not read PRD content"})
        system_mapping = safe_read(sys_file) if sys_file and sys_file.filename else ''
        session['prd_content'] = prd_content
        session['system_mapping_content'] = system_mapping
        session['context_notes'] = context_notes
        if system_mapping:
            session['system_info'] = system_mapping
        combined = prd_content + (f"\n\n--- System Mapping ---\n{system_mapping}" if system_mapping else '') + (f"\n\n--- User Context ---\n{context_notes}" if context_notes else '')
        session['combined_content'] = combined; session.modified = True
        if system_mapping:
            enhanced = create_epic_generation_context_with_system_mapping(prd_content, system_mapping, context_notes)
        else:
            enhanced = create_epic_generation_context(prd_content, '', context_notes)
        max_tokens = 2000; ctx_limit = 8192
        def est(t): return max(1, len(t)//4)
        if est(enhanced)+max_tokens > ctx_limit:
            enhanced = enhanced[: (ctx_limit - max_tokens)*4 ]
        sys_msg = "You are an expert product manager. Generate comprehensive epics."
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        resp = client.chat.completions.create(model=ALLOWED_MODEL, messages=[{"role":"system","content":sys_msg},{"role":"user","content":enhanced}], max_tokens=max_tokens, temperature=0.7)
        epics_text = resp.choices[0].message.content
        epics = parse_epics_from_response(epics_text)
        if not epics:
            return jsonify({"success": False, "error": "Failed to parse epics"})
        session['generated_epics'] = epics; session.modified = True
        return jsonify({"success": True, "epics": epics, "message": f"Generated {len(epics)} epics"})
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"success": False, "error": str(e)})

# ---------------- Parse Epics ----------------

def parse_epics_from_response(text: str):
    try:
        epics = []
        parts = text.split('Epic ')
        counter = 1
        for i, section in enumerate(parts):
            if i == 0: continue
            lines = [l.strip() for l in section.strip().split('\n') if l.strip()]
            if not lines: continue
            first = lines[0]
            title_part = first.split(':',1)[1].strip() if ':' in first else first
            desc = ''
            priority = 'Medium'
            for ln in lines[1:]:
                if ln.startswith('Description:'):
                    desc = ln.split(':',1)[1].strip()
                elif ln.startswith('Priority:'):
                    priority = ln.split(':',1)[1].strip()
                elif not desc and not ln.lower().startswith('priority') and not ln.lower().startswith('epic'):
                    desc = ln
            if title_part and desc:
                epics.append({'id': f'epic_{counter}','title': title_part,'description': desc,'priority': priority or 'Medium','estimated_stories': 'TBD','estimated_effort': 'TBD'})
                counter += 1
        logger.info(f"Parsed {len(epics)} epics")
        return epics
    except Exception as e:
        logger.error(f"Parse epics error: {e}")
        return []

# ---------------- Select Epic & Generate Stories ----------------
@app.route('/tabbed-select-epic', methods=['POST'])
def tabbed_select_epic():
    try:
        data = request.get_json() or {}
        epic_id = data.get('epic_id')
        epics = session.get('generated_epics', [])
        if not epic_id:
            return jsonify({"success": False, "error": "Epic ID required"})
        selected = next((e for e in epics if e.get('id') == epic_id), None)
        if not selected:
            return jsonify({"success": False, "error": "Epic not found"})
        session['current_epic'] = selected
        stories = generate_user_stories_for_epic(selected, session.get('combined_content',''))
        if stories:
            session['generated_user_stories'] = stories; session.modified = True
            return jsonify({"success": True, "epic": selected, "user_stories": stories, "message": f"Generated {len(stories)} user stories"})
        return jsonify({"success": False, "error": "Failed to generate user stories"})
    except Exception as e:
        logger.error(f"Select epic error: {e}")
        return jsonify({"success": False, "error": str(e)})

# ---------------- Story Generation ----------------

def generate_user_stories_for_epic(epic, prd_content):
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        system_info = session.get('system_info','')
        prd_snippet = (prd_content or '')[:6000] or 'No PRD context'
        prompt = f"""
You are an expert product manager. Generate detailed user stories for the epic as STRICT JSON array (no markdown) with 5-8 items.
Schema Example:\n[{{"id":"string(optional)","title":"string","description":"As a <user> I want ... so that ...","acceptance_criteria":["Given ... When ... Then ..."],"priority":"High|Medium|Low","estimated_effort":"Small|Medium|Large","responsible_systems":"PrimarySystem"}}]
Epic Title: {epic.get('title','')}
Epic Description: {epic.get('description','')}
System Mapping (truncated):\n{system_info[:3000] if system_info else 'None'}
Relevant PRD (truncated):\n{prd_snippet}
Return ONLY JSON.
"""
        resp = client.chat.completions.create(model=ALLOWED_MODEL, messages=[{"role":"system","content":"Return ONLY raw JSON when asked for JSON."},{"role":"user","content":prompt}], max_tokens=2500, temperature=0.55)
        raw = resp.choices[0].message.content.strip()
        logger.info(f"User story raw (first 300): {raw[:300]}")
        stories = []
        try:
            cleaned = raw
            if cleaned.startswith('```'):
                cleaned = '\n'.join(l for l in cleaned.splitlines() if not l.strip().startswith('```'))
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                for idx, s in enumerate(parsed,1):
                    if not isinstance(s, dict): continue
                    desc = (s.get('description') or '').strip()
                    if not desc: continue
                    ac = s.get('acceptance_criteria', [])
                    if isinstance(ac, str): ac=[ac]
                    ac = [a.strip() for a in ac if a.strip()]
                    if not ac: ac=["Given context When action Then outcome"]
                    stories.append({'id': f"story_{idx}", 'title': s.get('title') or f"Story {idx}", 'description': desc, 'acceptance_criteria': '\n'.join(f"• {a}" for a in ac), 'priority': s.get('priority','Medium'), 'estimated_effort': s.get('estimated_effort','Medium'), 'responsible_systems': extract_primary_system(s.get('responsible_systems','TBD')), 'epic_id': epic.get('id'), 'epic_title': epic.get('title')})
        except Exception as je:
            logger.warning(f"Primary JSON parse failed: {je}")
        if not stories:
            stories = parse_user_stories_from_response(raw, epic)
        if not stories:
            stories=[{'id':'story_fallback_1','title': f"Initial {epic.get('title')} story",'description': f"As a user I want core capability of {epic.get('title')} so that value is delivered.",'acceptance_criteria':'• Given the epic context\n• When the core action is performed\n• Then the outcome is successful','priority':'Medium','estimated_effort':'Medium','responsible_systems':'TBD','epic_id':epic.get('id'),'epic_title':epic.get('title')}]        
        return stories
    except Exception as e:
        logger.error(f"Story generation error: {e}\n{traceback.format_exc()}")
        return []

def parse_user_stories_from_response(content, epic):
    try:
        if not content: return []
        sections = re.split(r"(?:^|\n)(?:User Story|Story)\s+\d+[:\-]\s*", content)
        if len(sections) <= 1:
            sections = re.split(r"\n\d+\.\s+", content)
        out=[]; idx=1
        for sec in sections:
            sec=sec.strip()
            if not sec: continue
            lines=[l.strip() for l in sec.split('\n') if l.strip()]
            if not lines: continue
            title=lines[0][:140]
            desc_lines=[]; ac=[]; priority='Medium'; effort='Medium'; resp='TBD'; mode='desc'
            for ln in lines[1:]:
                low=ln.lower()
                if low.startswith('acceptance criteria'):
                    mode='ac'; continue
                if low.startswith('priority:'):
                    priority=ln.split(':',1)[1].strip().title() or 'Medium'; continue
                if low.startswith('effort:'):
                    effort=ln.split(':',1)[1].strip().title() or 'Medium'; continue
                if low.startswith('responsible systems:'):
                    resp=extract_primary_system(ln.split(':',1)[1].strip() or 'TBD'); continue
                if mode=='ac' and (ln.startswith('-') or ln.startswith('*') or ln.startswith('•') or ln.lower().startswith('given')):
                    ac.append(ln.lstrip('-*• ').strip())
                elif mode=='desc':
                    desc_lines.append(ln)
            description=' '.join(desc_lines) if desc_lines else title
            if not ac: ac=["Given context When action Then outcome"]
            out.append({'id': f'story_fallback_{idx}','title': title,'description': description,'acceptance_criteria': '\n'.join(f"• {c}" for c in ac),'priority': priority,'estimated_effort': effort,'responsible_systems': resp,'epic_id': epic.get('id'),'epic_title': epic.get('title')}); idx+=1
        logger.info(f"Fallback parser extracted {len(out)} stories")
        return out
    except Exception as e:
        logger.error(f"Fallback parse error: {e}")
        return []

# ---------------- Story Selection & Details ----------------
@app.route('/tabbed-select-story', methods=['POST'])
def tabbed_select_story():
    try:
        data = request.get_json() or {}
        story_id = data.get('story_id')
        story_obj = data.get('story')
        stories = session.get('generated_user_stories', [])
        selected = None
        if story_obj:
            if story_obj.get('id'):
                selected = next((s for s in stories if s.get('id')==story_obj.get('id')), story_obj)
            else:
                selected = story_obj
        elif story_id:
            selected = next((s for s in stories if s.get('id')==story_id), None)
        if not selected:
            return jsonify({"success": False, "error": "Story not found"})
        session['current_user_story']=selected
        details = generate_story_details(selected)
        session['story_details']=details; session.modified=True
        return jsonify({"success": True, "story": selected, "story_details": details})
    except Exception as e:
        logger.error(f"Select story error: {e}")
        return jsonify({"success": False, "error": str(e)})

def ask_assistant_from_file_optimized(agent_file, user_prompt):
    try:
        path = os.path.join('agents', agent_file)
        with open(path,'r',encoding='utf-8') as f: system_text = f.read()
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        resp = safe_chat_completion(client,[{"role":"system","content":system_text},{"role":"user","content":user_prompt}],purpose='requirements',max_tokens=3000,temperature=0.7)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Assistant file interaction error: {e}")
        return f"Error: {e}"

def generate_story_details(user_story):
    try:
        current_epic = session.get('current_epic', {})
        story_name = user_story.get('title','')
        story_description = user_story.get('description','')
        ac_text = user_story.get('acceptance_criteria','')
        if not ac_text or ac_text == 'Acceptance criteria to be defined':
            prompt = f"Generate at least 7 testable acceptance criteria as JSON array for story: {story_name}\nDescription: {story_description}"
            raw = ask_assistant_from_file_optimized('poc2_agent4_acceptanceCriteria_gen', prompt)
            generated=[]
            try:
                possible = raw.strip()
                if possible.startswith('['):
                    generated = json.loads(possible)
                    generated = [g if isinstance(g,str) else str(g) for g in generated]
            except Exception:
                for line in raw.splitlines():
                    line=line.strip('-*• ').strip()
                    if line.lower().startswith('given'):
                        generated.append(line)
            if len(generated)<7:
                generated += [
                    f"Given valid context When {story_name.lower()} Then success",
                    f"Given invalid input When {story_name.lower()} Then error is shown",
                    f"Given system error When {story_name.lower()} Then graceful handling",
                    f"Given security constraints When {story_name.lower()} Then access is controlled",
                    f"Given audit requirement When {story_name.lower()} Then event is logged",
                    f"Given performance target When {story_name.lower()} Then response within threshold",
                    f"Given edge case When {story_name.lower()} Then behaves correctly"
                ][:7-len(generated)]
            ac_text='\n'.join(f"• {c}" for c in generated)
        tagged = session.get('tagged_requirements', [])
        if not tagged:
            req_prompt = f"Extract >=6 functional & non-functional requirements (ID prefix REQ-) from PRD + story. Return JSON array of strings. Story: {story_name}. Description: {story_description}\nPRD:\n{session.get('prd_content','')[:4000]}"
            raw_req = ask_assistant_from_file_optimized('poc2_agent1_tagged_requirements', req_prompt)
            parsed=[]
            try:
                if raw_req.strip().startswith('['):
                    parsed = json.loads(raw_req)
                    parsed=[r if isinstance(r,str) else str(r) for r in parsed]
            except Exception:
                pass
            if not parsed:
                for l in raw_req.splitlines():
                    if re.match(r'^REQ-[A-Z]+-[0-9]{3}:', l.strip()):
                        parsed.append(l.strip())
            if not parsed:
                parsed=["REQ-FUNC-001: Core functional behavior.","REQ-SEC-001: Enforce authentication/authorization.","REQ-DATA-001: Validate and sanitize inputs.","REQ-ERR-001: Log and surface errors.","REQ-PERF-001: Meet performance thresholds.","REQ-AUD-001: Record audit trail."]
            seen=set(); cleaned=[]
            for r in parsed:
                r=re.sub(r'\s+',' ',r).strip()
                if r and r not in seen:
                    seen.add(r); cleaned.append(r)
            tagged = cleaned[:25]
            session['tagged_requirements']=tagged; session.modified=True
        trace_prompt = f"Create a traceability matrix mapping story to requirements. Story: {story_name}\nDescription: {story_description}\nAcceptance Criteria: {ac_text}"
        trace_resp = ask_assistant_from_file_optimized('poc2_traceability_agent', trace_prompt)
        return {'title': story_name,'description': story_description,'acceptance_criteria': ac_text,'priority': user_story.get('priority','Medium'),'estimated_effort': user_story.get('estimated_effort','Medium'),'epic_title': current_epic.get('title','Unknown Epic'),'epic_description': current_epic.get('description',''),'responsible_systems': extract_primary_system(user_story.get('responsible_systems','TBD')),'tagged_requirements': tagged,'traceability_matrix': trace_resp}
    except Exception as e:
        logger.error(f"Story details error: {e}")
        return {'title': user_story.get('title',''), 'description': user_story.get('description',''), 'acceptance_criteria': ac_text or 'Acceptance criteria to be defined'}

@app.route('/tabbed-user-story-details-page', methods=['GET','POST'])
def tabbed_user_story_details_page():
    try:
        if request.method == 'POST':
            story_id = request.form.get('selected_story_id')
            story_name = request.form.get('selected_story_name')
            story_desc = request.form.get('selected_story_description')
            stories = session.get('generated_user_stories', [])
            story = next((s for s in stories if s.get('id')==story_id or s.get('title')==story_name), {'id':story_id,'title':story_name,'description':story_desc})
            details = generate_story_details(story)
            session['current_user_story']=story; session['story_details']=details; session.modified=True
        else:
            story = session.get('current_user_story', {})
            if not story:
                return redirect(url_for('tabbed_layout'))
            details = session.get('story_details') or generate_story_details(story)
        ac = details.get('acceptance_criteria','')
        ac_list = [l.strip() for l in ac.split('\n') if l.strip()] if isinstance(ac,str) else ac
        return render_template('poc2_user_story_details.html',
                               epic_title=session.get('current_epic',{}).get('title',''),
                               epic_description=session.get('current_epic',{}).get('description',''),
                               user_story_name=story.get('title',''),
                               user_story_description=story.get('description',''),
                               acceptance_criteria=ac_list,
                               priority=details.get('priority','Medium'),
                               responsible_systems=extract_primary_system(details.get('responsible_systems','CAPS')),
                               tagged_requirements=details.get('tagged_requirements',[]),
                               traceability_matrix=details.get('traceability_matrix','Not available'))
    except Exception as e:
        logger.error(f"Details page error: {e}")
        return f"Error: {e}", 500

# ---------------- Chat Endpoints ----------------
@app.route('/tabbed-epic-chat', methods=['POST'])
def tabbed_epic_chat():
    try:
        msg = (request.get_json() or {}).get('message','').strip()
        if not msg: return jsonify({"success": False, "error": "Message required"})
        epics = session.get('generated_epics', [])
        if not epics: return jsonify({"success": False, "error": "No epics available"})
        ctx = "Current Epics:\n" + '\n'.join(f"{i+1}. {e.get('title')}\n   {e.get('description')}" for i,e in enumerate(epics))
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        resp = client.chat.completions.create(model=ALLOWED_MODEL, messages=[{"role":"system","content":f"You refine epics.\n{ctx}"},{"role":"user","content":msg}], max_tokens=900, temperature=0.7)
        return jsonify({"success": True, "response": resp.choices[0].message.content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/tabbed-story-chat', methods=['POST'])
def tabbed_story_chat():
    try:
        msg = (request.get_json() or {}).get('message','').strip()
        stories = session.get('generated_user_stories', [])
        epic = session.get('current_epic', {})
        if not stories: return jsonify({"success": False, "error": "No user stories"})
        ctx = f"Epic: {epic.get('title','')}\n" + '\n'.join(f"{i+1}. {s.get('title')} - {s.get('description')}" for i,s in enumerate(stories))
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        resp = client.chat.completions.create(model=ALLOWED_MODEL, messages=[{"role":"system","content":f"You refine user stories.\n{ctx}"},{"role":"user","content":msg}], max_tokens=900, temperature=0.7)
        return jsonify({"success": True, "response": resp.choices[0].message.content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/tabbed-details-chat', methods=['POST'])
def tabbed_details_chat():
    try:
        msg = (request.get_json() or {}).get('message','').strip()
        story = session.get('current_user_story', {})
        if not story: return jsonify({"success": False, "error": "No story selected"})
        ctx = f"Story: {story.get('title')}\nDescription: {story.get('description')}\nAC: {story.get('acceptance_criteria','')}"[:3500]
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        resp = client.chat.completions.create(model=ALLOWED_MODEL, messages=[{"role":"system","content":f"You refine story details.\n{ctx}"},{"role":"user","content":msg}], max_tokens=900, temperature=0.7)
        return jsonify({"success": True, "response": resp.choices[0].message.content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ---------------- Traceability Chat ----------------
@app.route('/chat_traceability', methods=['POST'])
def chat_traceability():
    try:
        data = request.get_json() or {}
        user_message = data.get('userMessage','').strip()
        story = data.get('userStory', {})
        if not user_message:
            return jsonify({"success": False, "error": "Message required"})
        context_text = f"Story Title: {story.get('title','')}\nDescription: {story.get('description','')}\nAcceptance Criteria: {story.get('acceptanceCriteria','')}\nCurrent Traceability: {data.get('currentTraceability','')}"
        system_prompt = f"You are a traceability specialist. Improve matrices.\n{context_text}\nFormat clear professional output."[:6000]
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        resp = client.chat.completions.create(model=ALLOWED_MODEL, messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_message}], max_tokens=1500, temperature=0.7)
        return jsonify({"success": True, "message": resp.choices[0].message.content.strip(), "userMessage": user_message})
    except Exception as e:
        logger.error(f"Traceability chat error: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

# ---------------- Requirements Chat ----------------
@app.route('/chat_requirements', methods=['POST'])
def chat_requirements():
    try:
        data = request.get_json() or {}
        msg = data.get('userMessage','').strip()
        story = data.get('userStory', {})
        if not msg: return jsonify({"success": False, "error": "Message required"})
        context = f"Story: {story.get('title','')}\nDescription: {story.get('description','')}\nCurrent Requirements: {data.get('currentRequirements', [])}"[:5500]
        sys_prompt = f"You are a requirements engineer. Provide tagged requirements (REQ-<CAT>-NNN).\n{context}"[:6000]
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        resp = client.chat.completions.create(model=ALLOWED_MODEL, messages=[{"role":"system","content":sys_prompt},{"role":"user","content":msg}], max_tokens=1200, temperature=0.7)
        return jsonify({"success": True, "message": resp.choices[0].message.content.strip(), "userMessage": msg})
    except Exception as e:
        logger.error(f"Requirements chat error: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

# ---------------- Jira Submission ----------------
@app.route('/tabbed-submit-jira', methods=['POST'])
def tabbed_submit_jira():
    try:
        current_epic = session.get('current_epic', {})
        current_story = session.get('current_user_story', {})
        story_details = session.get('story_details', {})
        if not current_story:
            return jsonify({"success": False, "error": "No story selected"})
        summary = current_story.get('title','')
        description = current_story.get('description','')
        ac_block=''
        if story_details.get('acceptance_criteria'):
            ac_items = story_details['acceptance_criteria'] if isinstance(story_details['acceptance_criteria'], list) else [l.strip('• ').strip() for l in story_details['acceptance_criteria'].split('\n') if l.strip()]
            ac_block = "\n\n*Acceptance Criteria:*\n" + '\n'.join(f"• {c}" for c in ac_items)
        req_block=''
        if story_details.get('tagged_requirements'):
            req_block = "\n\n*Tagged Requirements:*\n" + '\n'.join(f"• {r}" for r in story_details['tagged_requirements'])
        trace_block=''
        if story_details.get('traceability_matrix'):
            trace_block=f"\n\n*Traceability Matrix:*\n{story_details['traceability_matrix']}"
        full_desc = description + ac_block + req_block + trace_block + (f"\n\n*Responsible Systems:* {story_details.get('responsible_systems','CAPS')}")
        try:
            from jira import JIRA
            api_token = os.getenv('JIRA_API_TOKEN')
            if not api_token:
                return jsonify({"success": False, "error": "JIRA_API_TOKEN not configured"})

            # Configurable Jira connection (supports corporate CA bundles)
            server = os.getenv('JIRA_SERVER', 'https://lalluluke.atlassian.net/')
            email = os.getenv('JIRA_EMAIL', 'lalluluke@gmail.com')
            verify_env = (os.getenv('JIRA_VERIFY', 'true') or 'true').lower()
            ca_bundle = os.getenv('JIRA_CA_BUNDLE') or os.getenv('REQUESTS_CA_BUNDLE')

            # Determine SSL verification behavior: path to CA bundle, True, or False (not recommended)
            verify = True
            if ca_bundle:
                verify = ca_bundle
            elif verify_env in ('false', '0', 'no'):
                verify = False

            options = { 'server': server, 'verify': verify }
            jira = JIRA(options=options, basic_auth=(email, api_token))

            issue_dict = {'project': {'key':'SCRUM'}, 'summary': summary, 'description': full_desc, 'issuetype': {'name':'Story'}}
            new_issue = jira.create_issue(fields=issue_dict)
            base_browse = server.rstrip('/') + '/browse/'
            return jsonify({"success": True, "ticket_id": new_issue.key, "jira_url": f"{base_browse}{new_issue.key}"})
        except ImportError:
            return jsonify({"success": False, "error": "jira library not installed"})
        except Exception as je:
            return jsonify({"success": False, "error": f"Jira error: {je}"})
    except Exception as e:
        logger.error(f"Jira submit error: {e}")
        return jsonify({"success": False, "error": str(e)})

# ---------------- Legacy PRD Upload ----------------
@app.route('/tabbed-upload-prd', methods=['POST'])
def tabbed_upload_prd():
    return tabbed_upload_files()

# ---------------- Session Management ----------------
@app.route('/clear-session', methods=['POST'])
def clear_session():
    session.clear(); session.modified=True
    return jsonify({"success": True, "message": "Session cleared"})

# ---------------- System Mapping ----------------
@app.route('/upload-system-info', methods=['POST'])
def upload_system_info():
    try:
        f = request.files.get('file')
        if not f or not f.filename:
            return jsonify({"success": False, "error": "No file provided"}), 400
        content = safe_read(f)
        if not content: return jsonify({"success": False, "error": "Empty or unreadable file"}), 400
        session['system_info']=content; session['system_info_filename']=f.filename; session['system_info_uploaded']=datetime.now().isoformat(); session.modified=True
        preview='\n'.join(content.split('\n')[:10])
        if len(content.split('\n'))>10: preview += "\n... (truncated)"
        return jsonify({"success": True, "filename": f.filename, "preview": preview[:500]})
    except Exception as e:
        logger.error(f"Upload system info error: {e}")
        return jsonify({"success": False, "error": "Failed to upload"}), 500

@app.route('/get-system-info')
def get_system_info():
    return jsonify({"success": True, "has_system_info": bool(session.get('system_info')),"filename": session.get('system_info_filename',''),"uploaded_time": session.get('system_info_uploaded',''),"preview": (session.get('system_info','')[:500] if session.get('system_info') else '')})

@app.route('/clear-system-info', methods=['POST'])
def clear_system_info():
    for k in ['system_info','system_info_filename','system_info_uploaded']:
        session.pop(k, None)
    session.modified=True
    return jsonify({"success": True, "message": "System info cleared"})

# ---------------- System Mapping Helpers ----------------

def map_responsible_systems_from_csv(user_story_description, user_story_title):
    try:
        sys_info = session.get('system_info','')
        if not sys_info: return 'TBD'
        reader = csv.DictReader(io.StringIO(sys_info))
        rows=list(reader)
        if not rows: return 'TBD'
        cols={c.lower():c for c in rows[0].keys()}
        system_col=None; desc_col=None
        for k,v in cols.items():
            if not system_col and any(tok in k for tok in ['system','component','service']): system_col=v
            if not desc_col and any(tok in k for tok in ['description','function','purpose']): desc_col=v
        if not system_col: system_col=list(rows[0].keys())[0]
        search = f"{user_story_title} {user_story_description}".lower()
        matches=[]
        for row in rows:
            name=row.get(system_col,'').strip()
            if not name: continue
            score=0
            if name.lower() in search: score+=10
            for w in name.lower().split():
                if len(w)>2 and w in search: score+=5
            if desc_col:
                desc=row.get(desc_col,'').lower()
                for w in [w for w in desc.split() if len(w)>3]:
                    if w in search: score+=2
            if score>0: matches.append((name,score))
        if matches:
            matches.sort(key=lambda x:x[1], reverse=True)
            return matches[0][0]
        return rows[0].get(system_col,'').strip() or 'TBD'
    except Exception as e:
        logger.error(f"System map error: {e}")
        return 'TBD'

def extract_primary_system(responsible_systems_str):
    if not responsible_systems_str or responsible_systems_str.strip() in ['TBD','To be determined']:
        return 'CAPS'
    parts=[p.strip() for p in responsible_systems_str.split(',') if p.strip()]
    if not parts: return 'CAPS'
    primary = parts[0].replace('System:','').replace('Component:','').strip()
    return primary or 'CAPS'

# ---------------- System Mapping UI and Chat ----------------
@app.route('/system-mapping', methods=['GET'])
def system_mapping_page():
    try:
        return render_template('system_mapping.html')
    except Exception as e:
        logger.error(f"Error serving system mapping UI: {e}")
        return f"Error: {e}", 500

@app.route('/system-mapping-chat', methods=['POST'])
def system_mapping_chat():
    try:
        data = request.get_json() or {}
        user_message = (data.get('message') or '').strip()
        current_systems = data.get('current_systems', []) or []
        available_systems = data.get('available_systems', []) or []
        if not user_message:
            return jsonify({"success": False, "error": "Message is required"}), 400
        # Build concise system context
        def fmt_list(lst):
            return ('\n- ' + '\n- '.join(lst)) if lst else ' None'
        context = f"""
You are a technical architect AI that recommends systems for a target architecture.

User message:
{user_message}

Currently selected systems:{fmt_list(current_systems)}

Available systems (names only):{fmt_list(available_systems)}

Return guidance as a short paragraph. When appropriate, include a JSON blob with keys:
message: short guidance string
suggestions: array of objects {{ system: string, reason: string }} limited to top 5
warnings: array of strings for risks or gaps (0-3 items)
If you include JSON, ensure it is valid (no markdown fences).
""".strip()
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        resp = client.chat.completions.create(
            model=ALLOWED_MODEL,
            messages=[
                {"role": "system", "content": "Be concise, professional, and helpful."},
                {"role": "user", "content": context}
            ],
            max_tokens=700,
            temperature=0.5
        )
        answer = resp.choices[0].message.content.strip()
        return jsonify({"success": True, "message": answer})
    except Exception as e:
        logger.error(f"System mapping chat error: {e}")
        return jsonify({"success": False, "error": "Internal error"}), 500

# Friendly alias for back navigation from system_mapping.html
@app.route('/epic-results', methods=['GET'])
def epic_results_alias():
    try:
        # Render the epics page with any stored epics if available
        epics = session.get('generated_epics', [])
        return render_template('poc2_tabbed_workbench.html')
    except Exception as e:
        logger.error(f"Epic results alias error: {e}")
        return redirect(url_for('tabbed_layout'))

# ---------------- Main ----------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    debug = os.environ.get('FLASK_ENV','development').lower() != 'production'
    host = '0.0.0.0' if not debug else '127.0.0.1'
    logger.info(f"Starting Tabbed Backend on port {port}")
    print(f"Server running: http://localhost:{port}/tabbed-layout")
    app.run(host=host, port=port, debug=debug)
