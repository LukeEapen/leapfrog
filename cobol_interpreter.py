import os
import openai
from flask import Flask, request, jsonify, render_template
import time
import logging
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Set the upload folder path
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

#Personal platform.openai.com
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

assistant_id_agent_1 = "asst_P2HdYTBuZtBZqHGZizbsii1P"

def read_cobol_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()


def split_by_paragraphs(cobol_code):
    # Simple heuristic: paragraphs usually start at column 8 with a label
    lines = cobol_code.splitlines()
    blocks = []
    block = []
    for line in lines:
        if line.strip() == "":
            continue
        if line[:6].strip().endswith("."):
            if block:
                blocks.append("\n".join(block))
                block = []
        block.append(line)
    if block:
        blocks.append("\n".join(block))
    return blocks

def call_agent(assistant_id, message):
    logging.info(f"Calling assistant with ID: {assistant_id} and message: {message}")
    start_time = time.time()

    try:
        # Step 1: Create a thread (conversation container)
        thread = openai.beta.threads.create()

        # Step 2: Add user message
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=message
        )

        # Step 3: Run the assistant
        run = openai.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        # Step 4: Poll for completion
        while True:
            run_status = openai.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_status.status == "completed":
                break
            elif run_status.status in ["failed", "cancelled", "expired"]:
                raise Exception(f"Run status: {run_status.status}")
            time.sleep(1)

        # Step 5: Get assistant's reply
        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        for msg in reversed(messages.data):
            if msg.role == "assistant":
                end_time = time.time()
                elapsed_time = end_time - start_time
                logging.info(f"Assistant {assistant_id} completed in {elapsed_time:.2f} seconds")
                return msg.content[0].text.value.strip()

    except Exception as e:
        logging.error(f"Error in call_agent for assistant {assistant_id}: {str(e)}")
        raise


# Logging configuration
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return render_template('legacy.html')

@app.route('/agenticAI')
def agentic_ai():
    return render_template('legacy.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Ensure the upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Read the file and process it
        with open(file_path, 'r') as f:
            cobol_code = f.read()

        # Call the assistant
        try:
            response = call_agent("asst_P2HdYTBuZtBZqHGZizbsii1P", cobol_code)
            return jsonify({'response': response})
        except Exception as e:
            logging.error(f"Error calling assistant: {str(e)}")
            return jsonify({'error': 'Failed to process the file'}), 500
        
def main(cobol_folder):
    for root, _, files in os.walk(cobol_folder):  # Use os.walk to traverse subfolders
        for file in files:
            if file.endswith(".cbl"):
                path = os.path.join(root, file)
                print(f"\n📄 File: {path}")
                cobol_code = read_cobol_file(path)
                blocks = split_by_paragraphs(cobol_code)
                for i, block in enumerate(blocks, start=1):
                    print(f"\n--- Paragraph {i} ---")
                    explanation = call_agent(assistant_id_agent_1 , block)
                    print(explanation)
                    

if __name__ == "__main__":
    app.run(debug=True, port=8081)
    #folder = "C:\\Users\\LukeLallu\\OneDrive - Enterprise Blueprints Limited\\Documents\\EB\\python"
    #main(folder)