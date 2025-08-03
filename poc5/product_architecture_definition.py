
from flask import Flask, render_template, request, session
import os
import json

app = Flask(__name__)
app.secret_key = "your_secret_key"

AGENT_PROMPTS = {
    "agent_5_1": "agents/agent_5_1.txt",
    "agent_5_2": "agents/agent_5_2.txt",
    "agent_5_3": "agents/agent_5_3.txt",
    "agent_5_4": "agents/agent_5_4.txt",
    "agent_5_5": "agents/agent_5_5.txt",
}

def call_agent(agent_key, context):
    with open(os.path.join(os.path.dirname(__file__), AGENT_PROMPTS[agent_key]), "r", encoding="utf-8") as f:
        prompt = f.read()
    return f"{prompt}\n\n{context}"

@app.route('/', methods=['GET', 'POST'])
def tabbed_workbench():
    # Collect all fields from the tabbed form
    if request.method == 'POST':
        # Save all fields in session
        for key in request.form:
            session[key] = request.form[key]
        # Optionally, process agent calls here and save outputs
        # Example: session['blueprint'] = call_agent("agent_5_1", session.get('business_goals', ''))
    # Render the tabbed interface
    return render_template('tabbed_architecture_workbench.html',
        blueprint=session.get('architecture_diagram', ''),
        decisions=session.get('architectural_decisions', ''),
        communication=session.get('communication_protocols', ''),
        pros_cons=session.get('swot_analysis', ''),
        doc=session.get('compiled_document', '')
    )

if __name__ == "__main__":
    app.run(port=6001, debug=True)
