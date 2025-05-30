# --- main.py ---
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
from dotenv import load_dotenv
from docx import Document
from io import BytesIO
import openai, os, logging, time, json, uuid, asyncio
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup 
from markdown2 import markdown
from bs4 import BeautifulSoup
from docx.shared import Pt
from docx.enum.style import WD_STYLE_TYPE

# Try Redis; fallback to file storage
try:
    import redis
    redis_test = redis.Redis(host='localhost', port=6379, db=0)
    redis_test.ping()
    redis_client = redis_test
    USING_REDIS = True
except Exception:
    import tempfile
    TEMP_DIR = tempfile.gettempdir()
    USING_REDIS = False

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.secret_key = 'replace-with-secure-key'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

executor = ThreadPoolExecutor()

ASSISTANTS = {
    'agent_1': 'asst_EvIwemZYiG4cCmYc7GnTZoQZ',
    'agent_1_1': 'asst_sW7IMhE5tQ78Ylx0zQkh6YnZ',
    'agent_2': 'asst_t5hnaKy1wPvD48jTbn8Mx45z',
    'agent_3': 'asst_EkihtJQe9qFiztRdRXPhiy2G',
    'agent_4_1': 'asst_Ed8s7np19IPmjG5aOpMAYcPM', # Agent 4.1: Product Requirements / User Stories Generator – System Instructions
    'agent_4_2': 'asst_CLBdcKGduMvSBM06MC1OJ7bF', # Agent 4.2: Operational Business Requirements Generator – System Instructions
    'agent_4_3': 'asst_61ITzgJTPqkQf4OFnnMnndNb', # Agent 4.3: Capability-Scoped Non-Functional Requirements Generator – System Instructions
    'agent_4_4': 'asst_pPFGsMMqWg04OSHNmyQ5oaAy', # Agent 4.4: Data Attribute Requirement Generator – System Instructions
    'agent_4_5': 'asst_wwgc1Zbl5iknlDtcFLOuTIjd'  # Agent 4.5: LRC: Legal, Regulatory, and Compliance Synthesizer – System Instructions
}

def store_data(data):
    session_id = str(uuid.uuid4())
    if USING_REDIS:
        redis_client.setex(session_id, 3600, json.dumps(data))
    else:
        with open(os.path.join(TEMP_DIR, f'prd_session_{session_id}.json'), 'w') as f:
            json.dump(data, f)
    return session_id

def get_data(session_id):
    if USING_REDIS:
        data = redis_client.get(session_id)
        return json.loads(data) if data else None
    else:
        path = os.path.join(TEMP_DIR, f'prd_session_{session_id}.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    return None

def call_agent(agent_id, input_text):
    try:
        logging.info(f"[CALL START] Calling agent {agent_id} with input size {len(input_text)}")
        start_time = time.time()
        thread = openai.beta.threads.create()
        openai.beta.threads.messages.create(thread_id=thread.id, role="user", content=input_text)
        run = openai.beta.threads.runs.create(thread_id=thread.id, assistant_id=agent_id)

        wait_start = time.time()
        while True:
            status = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if status.status == "completed":
                break
            time.sleep(0.25)
        wait_end = time.time()

        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        result = messages.data[0].content[0].text.value

        logging.info(f"[CALL END] {agent_id} done in {time.time()-start_time:.2f}s (waited {wait_end-wait_start:.2f}s) and result : {result}")
        return result
    except Exception as e:
        logging.error(f"[ERROR] Agent {agent_id} failed: {e}")
        return f"Error: {str(e)}"

async def call_agents_parallel(agent_calls):
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(executor, call_agent, agent_id, input_text) for agent_id, input_text in agent_calls]
    return await asyncio.gather(*tasks)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == "admin" and password == "secure123":  # Replace with proper authentication
            session['logged_in'] = True
            return redirect(url_for('page1'))
        else:
            return render_template('page0_login.html', error=True)
    return render_template('page0_login.html', error=False)
@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        inputs = {key: request.form[key] for key in ['industry', 'sector', 'geography', 'intent', 'features']}
        session.update(inputs)

        context = "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in inputs.items())
        session_id = store_data({
            "inputs": inputs,
            "status": "processing"
        })
        session['data_key'] = session_id

        def run_agents():
            try:
                #a1 = call_agent(ASSISTANTS['agent_1'], context)
                a11, a2, a3 = asyncio.run(call_agents_parallel([
                    (ASSISTANTS['agent_1_1'], context),
                    (ASSISTANTS['agent_2'], context),
                    (ASSISTANTS['agent_3'], context)
                ]))
                final_data = {
                    #"agent_1_output": a1,
                    "product_overview": a11,
                    "feature_overview": a2,
                    "highest_order": a3,
                    "status": "complete"
                }
                if USING_REDIS:
                    redis_client.setex(session_id, 3600, json.dumps(final_data))
                else:
                    with open(os.path.join(TEMP_DIR, f'prd_session_{session_id}.json'), 'w') as f:
                        json.dump(final_data, f)
                logging.info(f"[AGENTS DONE] Page 1 background agents complete for session {session_id}")
            except Exception as e:
                logging.error(f"Error during agent execution: {e}")
                fail_data = {"status": "error", "message": str(e)}
                if USING_REDIS:
                    redis_client.setex(session_id, 3600, json.dumps(fail_data))
                else:
                    with open(os.path.join(TEMP_DIR, f'prd_session_{session_id}.json'), 'w') as f:
                        json.dump(fail_data, f)

        executor.submit(run_agents)

        return redirect('/page2')

    return render_template('page1_input.html')


@app.route('/page2', methods=['GET', 'POST'])
def page2():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    session_id = session.get('data_key', '')
    data = get_data(session_id) or {}

    # Wait for background job to complete
    start_time = time.time()
    timeout = 120  # max 30 seconds wait
    while True:
        if data.get("status") == "complete":
            break
        if data.get("status") == "error":
            return f"<h3>❌ Agent processing failed:</h3><pre>{data.get('message')}</pre>", 500
        time.sleep(1)
        if time.time() - start_time > timeout:
            return "Processing took too long. Please refresh the page in a few seconds.", 504
        data = get_data(session_id) or {}

    if request.method == 'POST':
        data.update({
            "product_overview": request.form.get("product_overview", ""),
            "feature_overview": request.form.get("feature_overview", "")
        })
        if USING_REDIS:
            redis_client.setex(session_id, 3600, json.dumps(data))
        return redirect('/page3')

    return render_template("page2_agents.html",
        agent11_output=data.get('product_overview', ''),
        agent2_output=data.get('feature_overview', '')
    )

@app.route('/update_content', methods=['POST'])
def update_content():
    if not session.get('logged_in'):
        return jsonify({'error': 'Not authenticated'}), 401
        
    try:
        data = request.get_json()
        content_type = data.get('type')
        new_content = data.get('content')
        
        if not content_type or not new_content:
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Get stored data
        stored_data = get_data(session.get('data_key', '')) or {}
        
        # Update content based on type
        if content_type == 'product':
            # Re-run Agent 1.1 with new content
            new_response = call_agent(ASSISTANTS['agent_1_1'], new_content)
            stored_data['product_overview'] = new_response
        elif content_type == 'feature':
            # Re-run Agent 2 with new content
            new_response = call_agent(ASSISTANTS['agent_2'], new_content)
            stored_data['feature_overview'] = new_response
            
        # Store updated data
        if USING_REDIS:
            redis_client.setex(session['data_key'], 3600, json.dumps(stored_data))
        else:
            with open(os.path.join(TEMP_DIR, f'prd_session_{session["data_key"]}.json'), 'w') as f:
                json.dump(stored_data, f)
                
        return jsonify({'success': True, 'response': new_response})
        
    except Exception as e:
        logging.error(f"Error updating content: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/page3', methods=['GET', 'POST'])
def page3():
    if not session.get('logged_in'): return redirect(url_for('login'))
    data = get_data(session.get('data_key', '')) or {}

    if request.method == 'POST':
        feature_overview = data.get("feature_overview", "")
        keys = [k for k in ASSISTANTS if k.startswith("agent_4_")]
        
        # Use asyncio to call all agents in parallel
        results = asyncio.run(call_agents_parallel([
            (ASSISTANTS[k], feature_overview) for k in keys
        ]))
        outputs = dict(zip(keys, results))

        data['combined_outputs'] = outputs
        session['combined_outputs'] = outputs

        if USING_REDIS:
            redis_client.setex(session['data_key'], 3600, json.dumps(data))
        else:
            with open(os.path.join(TEMP_DIR, f'prd_session_{session["data_key"]}.json'), 'w') as f:
                json.dump(data, f)

        return redirect('/page4')

    return render_template('page3_prompt_picker.html', highest_order=data.get('highest_order', ''))


@app.route('/page4')
def page4():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    data = get_data(session.get('data_key', '')) or {}
    outputs = data.get('combined_outputs', {})
    outputs["product_overview"] = data.get("product_overview", "")
    outputs["feature_overview"] = data.get("feature_overview", "")

    if not outputs:  # Only re-run if not already computed
        feature_overview = data.get("feature_overview", "")
        async def gather_outputs():
            keys = [k for k in ASSISTANTS if k.startswith("agent_4_")]
            results = await call_agents_parallel([(ASSISTANTS[k], feature_overview) for k in keys])
            return dict(zip(keys, results))
        
        outputs = asyncio.run(gather_outputs())
        renamed_outputs = {
            "Product Requirements / User Stories Generator": outputs.get("agent_4_1", ""),
            "Operational Business Requirements Generator": outputs.get("agent_4_2", ""),
            "Capability-Scoped Non-Functional Requirements Generator": outputs.get("agent_4_3", ""),
            "Data Attribute Requirement Generator": outputs.get("agent_4_4", ""),
            "LRC: Legal, Regulatory, and Compliance Synthesizer": outputs.get("agent_4_5", "")
        }
        data['combined_outputs'] = {
            "Product Overview (Agent 1.1)": data.get("product_overview", ""),
            "Feature Overview (Agent 2)": data.get("feature_overview", ""),
            **renamed_outputs
        }
        session['combined_outputs'] = outputs

        if USING_REDIS:
            redis_client.setex(session['data_key'], 3600, json.dumps(data))
        else:
            with open(os.path.join(TEMP_DIR, f'prd_session_{session["data_key"]}.json'), 'w') as f:
                json.dump(data, f)

    return render_template('page4_final_output.html', outputs=outputs)


@app.route("/download_doc", methods=["POST"])
def download_doc():
    doc = Document()
    try:
        content = request.get_json(force=True)
    except:
        content = {}

    data = get_data(session.get("data_key", "")) or {}
    content.setdefault("Product Overview (Agent 1.1)", data.get("product_overview", ""))
    content.setdefault("Feature Overview (Agent 2)", data.get("feature_overview", ""))

    section_order = [
        "Product Overview",
        "Feature Overview",
        "Product requirements",
        "Functional requirements",
        "Non-functional requirements",
        "Data requirements",
        "Legal, compliance, regulatory requirements"
    ]

    sorted_items = [(key, content[key]) for key in section_order if key in content]

    for title, raw_markdown in sorted_items:
        doc.add_heading(title, level=2)
        html_content = markdown(raw_markdown, extras=["fenced-code-blocks", "tables"])
        soup = BeautifulSoup(html_content, "html.parser")
        for el in soup.find_all(["h1", "h2", "h3", "strong", "p", "li", "ul", "ol"]):
            text = el.get_text(strip=True)
            if el.name == "strong":
                doc.add_paragraph(text).runs[0].bold = True
            elif el.name == "li":
                doc.add_paragraph(f"• {text}", style="List Bullet")
            elif el.name in ["h1", "h2", "h3"]:
                doc.add_heading(text, level=3)
            else:
                doc.add_paragraph(text)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="PRD_Draft.docx")



if __name__ == '__main__':
    ort = int(os.environ.get("PORT", 7001))
    app.run(host="0.0.0.0", port=ort, debug=True)
