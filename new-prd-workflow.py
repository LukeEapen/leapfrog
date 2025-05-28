# --- main.py ---
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
from dotenv import load_dotenv
from docx import Document
from io import BytesIO
import openai, os, logging, time, json, uuid, asyncio
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor

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
        if request.form['username'] == 'admin' and request.form['password'] == 'secure123':
            session['logged_in'] = True
            return redirect(url_for('page1'))
        return 'Invalid credentials', 401
    return render_template('page0_login.html')

@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if not session.get('logged_in'): return redirect(url_for('login'))
    if request.method == 'POST':
        inputs = {key: request.form[key] for key in ['industry', 'sub_industry', 'sector', 'geography', 'intent', 'features']}
        session.update(inputs)

        context = "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in inputs.items())
        a1 = call_agent(ASSISTANTS['agent_1'], context)
        a11, a2, a3 = asyncio.run(call_agents_parallel([
            (ASSISTANTS['agent_1_1'], a1),
            (ASSISTANTS['agent_2'], a1),
            (ASSISTANTS['agent_3'], a1)
        ]))

        session_id = store_data({
            "agent_1_output": a1,
            "product_overview": a11,
            "feature_overview": a2,
            "highest_order": a3
        })
        session['data_key'] = session_id
        return redirect('/page2')
    return render_template('page1_input.html')

@app.route('/page2', methods=['GET', 'POST'])
def page2():
    if not session.get('logged_in'): return redirect(url_for('login'))
    data = get_data(session.get('data_key', '')) or {}
    if request.method == 'POST':
        data.update({
            "product_overview": request.form.get("product_overview", ""),
            "feature_overview": request.form.get("feature_overview", "")
        })
        redis_client.setex(session['data_key'], 3600, json.dumps(data)) if USING_REDIS else None
        return redirect('/page3')
    return render_template("page2_agents.html",
        agent11_output=data.get('product_overview', ''),
        agent2_output=data.get('feature_overview', '')
    )

@app.route('/page3', methods=['GET', 'POST'])
def page3():
    if not session.get('logged_in'): return redirect(url_for('login'))
    data = get_data(session.get('data_key', '')) or {}

    if request.method == 'POST':
        feature_overview = data.get("feature_overview", "")
        outputs = {}
        for key in [k for k in ASSISTANTS if k.startswith("agent_4_")]:
            outputs[key] = call_agent(ASSISTANTS[key], feature_overview)
        session['combined_outputs'] = outputs
        return redirect('/page4')

    return render_template('page3_prompt_picker.html', highest_order=data.get('highest_order', ''))


@app.route('/page4')
def page4():
    if not session.get('logged_in'): return redirect(url_for('login'))
    data = get_data(session.get('data_key', '')) or {}
    feature_overview = data.get("feature_overview", "")

    async def gather_outputs():
        keys = [k for k in ASSISTANTS if k.startswith("agent_4_")]
        results = await call_agents_parallel([(ASSISTANTS[k], feature_overview) for k in keys])
        return dict(zip(keys, results))

    outputs = asyncio.run(gather_outputs())
    session['combined_outputs'] = outputs
    return render_template('page4_final_output.html', outputs=outputs)

@app.route("/download_doc", methods=["POST"])
def download_doc():
    doc = Document()

    # Add product and feature overviews
    doc.add_heading("Product Overview (Agent 1.1)", level=2)
    doc.add_paragraph(get_data(session["data_key"]).get("product_overview", ""))

    doc.add_heading("Feature Overview (Agent 2)", level=2)
    doc.add_paragraph(get_data(session["data_key"]).get("feature_overview", ""))

    # Add selective outputs from agents 4.1 to 4.5
    for key in ["agent_4_1", "agent_4_2", "agent_4_3", "agent_4_4", "agent_4_5"]:
        content = session.get("combined_outputs", {}).get(key, "")
        doc.add_heading(key.replace("_", " ").title(), level=2)
        doc.add_paragraph(content)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="PRD_Draft.docx")

if __name__ == '__main__':
    ort = int(os.environ.get("PORT", 7007))
    app.run(host="0.0.0.0", port=ort, debug=True)
