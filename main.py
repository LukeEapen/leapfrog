from flask import Flask
from prd_openai_app_v1 import v1_blueprint
from new_prd_workflow import v2_blueprint

app = Flask(__name__)
app.secret_key = 'replace-with-secure-key'

# Register versioned Blueprints
app.register_blueprint(v1_blueprint, url_prefix='/v1')
app.register_blueprint(v2_blueprint, url_prefix='/v2')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7001)