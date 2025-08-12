# app.py - Flask Microservice Entry Point
from flask import Flask
from flask_cors import CORS

# Import your blueprints/routes here
# from routes.user_routes import user_bp
# from models import db

app = Flask(__name__)
CORS(app)

# Configurations (update as needed)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'  # Or your DB URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions (uncomment if using SQLAlchemy, etc.)
# db.init_app(app)

# Register blueprints/routes
# app.register_blueprint(user_bp)

@app.route('/')
def index():
    return {'status': 'ok', 'message': 'Welcome to the Flask Microservice!'}

# Add error handlers, CLI commands, etc. as needed

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
