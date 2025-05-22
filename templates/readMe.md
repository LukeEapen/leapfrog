✅ Step 4: Running the Application
Install Flask and OpenAI:
pip install flask openai
pip install openai python-dotenv

python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt

Run your Flask app:
python app.py

pip list

Visit your UI:
http://127.0.0.1:5000


pip install openai python-dotenv
⑤ Run the interpreter
python main.py


# Agentic AI Workflow – OpenAI Assistants

This project is an **agentic AI workflow platform** that orchestrates multiple OpenAI assistants (agents) to automate and validate business requirements, compliance, architecture, and more. It features a modern web UI and a modular backend, making it easy to extend and adapt for enterprise use cases.

---

## 🚀 Features

- **Multi-Agent Orchestration:** Agents for intent clarification, requirement generation, validation, compliance, NFR, architecture, and functional parity.
- **Modern Web UI:** Bootstrap-based, responsive, and visually organized for clarity.
- **Async Flask Backend:** Fast, scalable, and ready for production.
- **Structured Agent Instructions:** Each agent has a dedicated instruction file for easy customization.
- **Markdown & Table Rendering:** Outputs are formatted for readability.
- **Easy Extensibility:** Add or modify agents and instructions as needed.

---

## 🗂️ Project Structure

```
openai-assistant-clean/
│
├── agent_instructions/
│   ├── agent_1.txt
│   ├── agent_2.txt
│   ├── agent_3.txt
│   ├── agent_5.txt
│   ├── agent_6.txt
│   ├── agent_7.txt
│   └── agent_8.txt
│
├── templates/
│   ├── index-openai.html
│   └── index-openai-v2.html
│
├── langgraph_app.py      # Main async Flask backend (LangGraph version)
├── openai-app.py         # Synchronous Flask backend (OpenAI only)
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart

### 1. **Clone the Repository**

```sh
git clone https://github.com/your-org/openai-assistant-clean.git
cd openai-assistant-clean
```

### 2. **Set Up Your Environment**

Create and activate a virtual environment:

```sh
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. **Install Dependencies**

```sh
pip install -r requirements.txt
```

### 4. **Configure Environment Variables**

Create a `.env` file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

### 5. **Run the Application**

#### For the async LangGraph backend:
```sh
python langgraph_app.py
```

#### For the synchronous OpenAI backend:
```sh
python openai-app.py
```

The app will be available at [http://localhost:5003/agenticAI](http://localhost:5003/agenticAI) (or port 5001 for openai-app.py).

---

## 🖥️ Usage

1. **Enter your business intent** in the first text area and submit to Agent 1.
2. **Review and approve** each agent's output to proceed to the next step.
3. **Agents 5–8** outputs are displayed as columns in a formatted table for easy comparison.
4. **All outputs** are structured and ready for audit, compliance, and further review.

---

## 🧑‍💻 Customizing Agents

- **Instructions:** Edit the files in `agent_instructions/` to change agent behavior, tone, or output format.
- **Add Agents:** Add new agent instructions and update the backend to include them in the workflow.
- **UI:** Modify the HTML templates in `templates/` for branding or layout changes.

---

## 📝 Example: Agent 5 Instruction

See `agent_instructions/agent_5.txt` for a detailed, compliance-first prompt.  
Each agent file contains:
- **Role summary**
- **Objectives**
- **Inputs/Outputs**
- **Success criteria**
- **Output format**
- **Behavior and tone**

---

## 🛠️ Troubleshooting

- **Dependencies:** Ensure all packages in `requirements.txt` are installed.
- **API Key:** Make sure your `.env` file is present and correct.
- **Async Issues:** Use `langgraph_app.py` for async workflows; use `openai-app.py` for synchronous.
- **Errors:** Check the `agent_responses.log` file for detailed error logs.

---

## 📦 Requirements

See [`requirements.txt`](./requirements.txt) for all dependencies.

---

## 📄 License

This project is for educational and internal enterprise prototyping.  
For production or commercial use, review all dependencies and compliance requirements.

---

## 🤝 Contributing

Pull requests and issues are welcome! Please document any new agents or workflow changes.

---

## 📬 Support

For help, open an issue or contact the project maintainer.

---