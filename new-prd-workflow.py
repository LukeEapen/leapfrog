# --- main.py ---
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from dotenv import load_dotenv
from flask import send_file
from docx import Document
from io import BytesIO
import openai, os, logging, time

load_dotenv()

app = Flask(__name__)
app.secret_key = 'replace-with-secure-key'
openai.api_key = os.getenv("OPENAI_API_KEY")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define assistant IDs for agents
ASSISTANTS = {
    'agent_1': 'asst_EvIwemZYiG4cCmYc7GnTZoQZ',
    'agent_1_1': 'asst_sW7IMhE5tQ78Ylx0zQkh6YnZ', # Agent 1.1: Product Overview Generator – System Instructions
      'agent_2': 'asst_t5hnaKy1wPvD48jTbn8Mx45z',
      'agent_3': 'asst_EkihtJQe9qFiztRdRXPhiy2G', # Agent 2 : Requirement Generator
    'agent_4_1': 'asst_Ed8s7np19IPmjG5aOpMAYcPM', # Agent 4.1: Product Requirements / User Stories Generator – System Instructions
    'agent_4_2': 'asst_CLBdcKGduMvSBM06MC1OJ7bF', # Agent 4.2: Operational Business Requirements Generator – System Instructions
    'agent_4_3': 'asst_61ITzgJTPqkQf4OFnnMnndNb', # Agent 4.3: Capability-Scoped Non-Functional Requirements Generator – System Instructions
    'agent_4_4': 'asst_pPFGsMMqWg04OSHNmyQ5oaAy', # Agent 4.4: Data Attribute Requirement Generator – System Instructions
    'agent_4_5': 'asst_wwgc1Zbl5iknlDtcFLOuTIjd', # Agent L: Legal Requirements Generator – System Instructions
    'agent_4_6': 'asst_qEfqPFiWiXDDCNPNwP7n0r0B', # Agent R: Regulatory Rules Translator – System Instructions
    'agent_4_7': 'asst_dJIQFP63GIjc2F04rTA3Vj3g'  # Agent C: Compliance Enforcer Definition Generator – System Instructions
}

# --- Helper to call assistant ---
def call_agent(agent_id, input_text):
    try:
        logging.info(f"[CALL START] Calling agent {agent_id} with input length {len(input_text)} characters")
        start_time = time.time()

        thread = openai.beta.threads.create()
        openai.beta.threads.messages.create(thread_id=thread.id, role="user", content=input_text)
        run = openai.beta.threads.runs.create(thread_id=thread.id, assistant_id=agent_id)

        wait_start = time.time()
        while True:
            status = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if status.status == "completed":
                break
            logging.info(f"Waiting for agent {agent_id}... status: {status.status}")
            time.sleep(0.25)

        wait_end = time.time()
        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        result = messages.data[0].content[0].text.value

        total_time = time.time() - start_time
        logging.info(f"[CALL END] Agent {agent_id} completed in {total_time:.2f} seconds (waiting time: {wait_end - wait_start:.2f}s)")
        return result
    except Exception as e:
        logging.error(f"[ERROR] Agent {agent_id} failed: {str(e)}")
        return f"Error: {str(e)}"

# --- Page 0: Login ---
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'secure123':
            session['logged_in'] = True
            return redirect(url_for('page1'))
        return 'Invalid credentials', 401
    return render_template('page0_login.html')

# --- Page 1: Input Form ---
@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if not session.get('logged_in'): return redirect(url_for('login'))

    if request.method == 'POST':
        # Collect and store input fields
        industry = request.form['industry']
        sub_industry = request.form['sub_industry']
        sector = request.form['sector']
        geography = request.form['geography']
        intent = request.form['intent']
        features = request.form['features']

        session['industry'] = industry
        session['sub_industry'] = sub_industry
        session['sector'] = sector
        session['geography'] = geography
        session['intent'] = intent
        session['features'] = features

        context = f"""Industry: {industry}
Sub-Industry: {sub_industry}
Sector: {sector}
Geography: {geography}
Product Intent: {intent}
Features: {features}
"""

        # Call Agent 1 with the full context
        agent1_response = call_agent(ASSISTANTS['agent_1'], context)

        # Use Agent 1 output to call 1.1, 2, 3
        agent11_output = call_agent(ASSISTANTS['agent_1_1'], agent1_response)
        agent2_output = call_agent(ASSISTANTS['agent_2'], agent1_response)
        agent3_output = call_agent(ASSISTANTS['agent_3'], agent1_response)

        # Store agent outputs for page 2
        session['product_overview'] = agent11_output
        session['feature_overview'] = agent2_output
        session['highest_order'] = agent3_output

        return redirect(url_for('page2'))

    return render_template('page1_input.html')


# --- Page 2: Agent 1 & Agent 2 Interaction ---
@app.route('/page2', methods=['GET', 'POST'])
def page2():
    if not session.get('logged_in'): return redirect(url_for('login'))

    if request.method == "POST":
        session["product_overview"] = request.form["product_overview"]
        session["feature_overview"] = request.form["feature_overview"]
        session["highest_order"] = request.form["highest_order"]
        return redirect("/page3")

    # Use Agent 1 output as input to Agent 3
    agent1_output = session.get("product_overview")
    agent2_output = call_agent(ASSISTANTS["agent_2"], agent1_output)
    agent3_output = call_agent(ASSISTANTS["agent_3"], agent1_output)

    session["feature_overview"] = agent2_output
    session["highest_order"] = agent3_output

    return render_template(
        "page2_agents.html",
        agent11_output=agent1_output,
        agent2_output=agent2_output,
        agent3_output=agent3_output,
    )
# --- Page 3: Prompt Selector ---
@app.route('/page3', methods=['GET', 'POST'])
def page3():
    if not session.get('logged_in'): return redirect(url_for('login'))
    if request.method == 'POST':
        session['selected_prompts'] = request.form.getlist('selected_prompts')
        return redirect(url_for('page4'))
    return render_template('page3_prompt_picker.html')

# --- Page 4: Agent 4 Results ---
@app.route('/page4')
def page4():
    if not session.get('logged_in'): return redirect(url_for('login'))
    combined_outputs = {}
    feature_overview = session.get('feature_overview', '')
    if not feature_overview:
        app.logger.warning("Missing 'feature_overview' in session during /page4")

    for key in ASSISTANTS:
        combined_outputs[key] = call_agent(ASSISTANTS[key], feature_overview)

    return render_template('page4_final_output.html', outputs=combined_outputs)

@app.route("/download_doc", methods=["POST"])
def download_doc():
    data = request.get_json()
    doc = Document()
    for title, content in data.items():
        doc.add_heading(title, level=2)
        doc.add_paragraph(content)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="PRD_Draft.docx",
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
if __name__ == '__main__':
    ort = int(os.environ.get("PORT", 7007))
    app.run(host="0.0.0.0", port=ort, debug=True)
