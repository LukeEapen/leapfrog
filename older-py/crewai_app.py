from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI  # Use langchain_openai with new LangChain version

# Load environment variables
load_dotenv()
app = Flask(__name__)
openai_api_key = os.getenv("OPENAI_API_KEY")

def load_instruction(agent_name):
    # If your files have .txt extension, use:
    # with open(f"agent_instructions/{agent_name}.txt", "r", encoding="utf-8") as f:
    with open(f"agent_instructions/{agent_name}.txt", "r", encoding="utf-8") as f:
        return f.read()
    
# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, openai_api_key=openai_api_key)

# Define all 8 agents
agent_1 = Agent(
    role="Intent Clarifier",
    goal="Transform vague user input into structured directive prompts for requirement generation",
    backstory=load_instruction("agent_1"),
    verbose=True,
    llm=llm,
)

agent_2 = Agent(
    role="Requirement Generator",
    goal="Generate exhaustive, testable requirements based on the structured prompt",
    backstory=load_instruction("agent_2"),
    verbose=True,
    llm=llm,
)

agent_3 = Agent(
    role="Validator Agent",
    goal="Validator Agent within a multi-agent pipeline supporting Capital One’s credit card core modernization. It serves as the final QA checkpoint for requirement documents labeled as “Highest Order Requirements.” The primary responsibility is to evaluate completeness, clarity, alignment, and soundness of requirements across business, technical, and regulatory dimensions—ensuring they are enterprise-ready for planning, execution, and compliance audit.",
    backstory=load_instruction("agent_3"),  # Make sure this file exists: agent_instructions/agent_3
    verbose=True,
    llm=llm,
)

agent_5 = Agent(
    role="Regulatory Analyst",
    goal="Analyze the requirement for legal, regulatory, and compliance implications",
    backstory=load_instruction("agent_5"),
    verbose=True,
    llm=llm,
)

agent_6 = Agent(
    role="NFR Specialist",
    goal="Ensure the solution meets performance, resilience, scalability, and other non-functional requirements",
    backstory=load_instruction("agent_6"),
    verbose=True,
    llm=llm,
)

agent_7 = Agent(
    role="Platform Architect",
    goal="Ensure solution aligns with modern architecture and platform decoupling principles",
    backstory=load_instruction("agent_7"),
    verbose=True,
    llm=llm,
)

agent_8 = Agent(
    role="Legacy Integration Lead", #Detailed explanation of requirements
    goal="Assess backward compatibility and legacy system integration",
    backstory=load_instruction("agent_8"),
    verbose=True,
    llm=llm,
)

agent_4 = Agent(
    role="Consolidation Reviewer",
    goal="Summarize the outputs of Agents 5–8 into a single advisory",
    backstory="Agent 4 consolidates specialist reviews into one coherent analysis for final consideration.",
    verbose=True,
    llm=llm,
)

@app.route('/api/query_agents', methods=['POST'])
def query_agents():
    data = request.get_json()
    question = data.get("question", "")

    # Sequential tasks
    task1 = Task(
        description=question,
        expected_output="A structured prompt suitable for requirement generation, framed in modernization context.",
        agent=agent_1
    )
    task2 = Task(
        description="Generate system requirements based on the clarified prompt.",
        expected_output="A detailed list of business and technical requirements covering full capability lifecycle.",
        agent=agent_2,
        context=[task1]
    )
    task3 = Task(
        description="Validator Agent within a multi-agent pipeline supporting Capital One’s credit card core modernization. It serves as the final QA checkpoint for requirement documents labeled as “Highest Order Requirements.” The primary responsibility is to evaluate completeness, clarity, alignment, and soundness of requirements across business, technical, and regulatory dimensions—ensuring they are enterprise-ready for planning, execution, and compliance audit.",
        expected_output="For Each Issue or Suggestion: Tag the affected requirement using its sub-category and row content.",
        agent=agent_3,
        context=[task2]
    )

    # Parallel specialist reviewers
    task5 = Task(
        description="Review requirements for Legal, Regulatory, and Compliance (LRC) aspects.",
        expected_output="An annotated list of LRC risks and recommendations.",
        agent=agent_5,
        context=[task2]
    )
    task6 = Task(
        description="Define Non-Functional Requirements (NFRs) for the proposed capability.",
        expected_output="A complete list of NFRs with justification and priorities.",
        agent=agent_6,
        context=[task2]
    )
    task7 = Task(
        description="Evaluate architectural alignment with platform and modernization principles.",
        expected_output="Feedback on architectural quality and modernization readiness.",
        agent=agent_7,
        context=[task2]
    )
    task8 = Task(
        description="Assess how well the plan integrates with or replaces legacy systems.",
        expected_output="Analysis of legacy dependencies and migration considerations.",
        agent=agent_8,
        context=[task2]
    )

    # Final consolidation
    agent_4_output = "\n\n".join([
        str(task5.output) if task5.output else "",
        str(task6.output) if task6.output else "",
        str(task7.output) if task7.output else "",
        str(task8.output) if task8.output else ""
    ])
        # Execute the full Crew
    crew = Crew(
        agents=[agent_1, agent_2, agent_3, agent_5, agent_6, agent_7, agent_8],
        tasks=[task1, task2, task3, task5, task6, task7, task8]
    )

    crew.kickoff()

    return jsonify({
        "agent_1": str(task1.output) if task1.output else "No output",
        "agent_2": str(task2.output) if task2.output else "No output",
        "agent_3": str(task3.output) if task3.output else "No output",
        "agent_4": agent_4_output or "No output"
    })

@app.route('/agenticAI')
def index():
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port)
