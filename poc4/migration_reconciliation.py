from flask import Flask
from backend.routes import poc4_bp

import os
app = Flask(__name__, template_folder=os.path.join('frontend', 'templates'))
app.secret_key = 'your-secret-key'  # Replace with a secure key in production

# Register the POC4 Blueprint
app.register_blueprint(poc4_bp)

@app.route('/')
def index():
    return '<h2 class="mb-0">Data Modelling, Migration & Reconciliation</h2><a href="/poc4/page1">Start Workflow</a>'

if __name__ == '__main__':
    app.run(debug=True, port=5000)
