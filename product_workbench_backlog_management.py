#!/usr/bin/env python3
"""
Safe launcher for Tabbed Backend with enhanced error handling
"""

import os
import sys
import logging
import traceback
from datetime import datetime

def setup_logging():
    """Set up enhanced logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tabbed_backend.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_environment():
    """Check critical environment setup."""
    logger = logging.getLogger(__name__)
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        logger.warning("OPENAI_API_KEY not set - AI features may not work")
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("   AI features will be limited without this key")
        print("   Set it with: set OPENAI_API_KEY=your_key_here")
    
    # Check required files
    required_files = [
        'poc2_backend_processor_tabbed.py',
        'templates/poc2_tabbed_workbench.html'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    return True

def safe_import_backend():
    """Safely import the backend module with error handling."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Importing tabbed backend...")
        
        # Add current directory to Python path
        sys.path.insert(0, os.getcwd())
        
        # Import the backend module
        import poc2_backend_processor_tabbed as backend

        # --- PATCH: Speed up OpenAI API calls in backend ---
        # If backend exposes a config or settings object, set fastest model (gpt-4o) and low temperature
        # Enforce max_tokens <= 1024 to avoid OpenAI context length errors
        safe_max_tokens = 1024
        if hasattr(backend, 'set_ai_config'):
            backend.set_ai_config(model="gpt-4o", temperature=0.3, max_tokens=safe_max_tokens, parallel=True)
            logger.info(f"Set backend AI config: gpt-4o, temp=0.3, max_tokens={safe_max_tokens}, parallel=True")
        elif hasattr(backend, 'ai_config'):
            backend.ai_config['model'] = "gpt-4o"
            backend.ai_config['temperature'] = 0.3
            backend.ai_config['max_tokens'] = safe_max_tokens
            backend.ai_config['parallel'] = True
            logger.info(f"Updated backend.ai_config for speed (gpt-4o, max_tokens={safe_max_tokens})")
        # If backend has a function to optimize story generation, call it
        if hasattr(backend, 'optimize_story_generation'):
            backend.optimize_story_generation()
            logger.info("Called backend.optimize_story_generation() for parallelism")

        # --- PATCH: Use best available OpenAI model and further optimize speed ---
        # Try GPT-4o first, then Gemini 1.5 Pro, Claude 3 Sonnet, else fallback to gpt-3.5-turbo
        best_model = None
        for candidate in ["gpt-4o", "gemini-1.5-pro", "claude-3-sonnet", "gpt-3.5-turbo"]:
            if hasattr(backend, 'is_model_available') and backend.is_model_available(candidate):
                best_model = candidate
                break
        if not best_model:
            best_model = "gpt-4o"
        # Set lower temperature and max_tokens for speed
        # Enforce max_tokens <= 1024 for all config
        safe_max_tokens = 1024
        ai_config = {
            "model": best_model,
            "temperature": 0.2,
            "max_tokens": safe_max_tokens,
            "parallel": True,
            "stream": True
        }
        if hasattr(backend, 'set_ai_config'):
            backend.set_ai_config(**ai_config)
            logger.info(f"Set backend AI config: {ai_config}")
        elif hasattr(backend, 'ai_config'):
            backend.ai_config.update(ai_config)
            logger.info(f"Updated backend.ai_config: {ai_config}")
        # If backend has a function to optimize story/epic generation, call it
        if hasattr(backend, 'optimize_story_generation'):
            backend.optimize_story_generation()
            logger.info("Called backend.optimize_story_generation() for parallelism")
        if hasattr(backend, 'optimize_epic_generation'):
            backend.optimize_epic_generation()
            logger.info("Called backend.optimize_epic_generation() for parallelism")

        # --- PATCH: Enforce strict context length checks for OpenAI API calls ---
        # If backend exposes a function to set context length safeguards, call it
        # Otherwise, monkey-patch backend's OpenAI call to truncate/summarize PRD and message history
        context_limit = 8192  # gpt-4o context length
        def truncate_for_context(messages, prd_text, max_tokens=1024):
            # Simple token estimation: 1 token ≈ 4 chars (for English)
            def estimate_tokens(text):
                return max(1, len(text) // 4)
            # Truncate PRD if too long
            prd_tokens = estimate_tokens(prd_text)
            if prd_tokens > context_limit // 2:
                prd_text = prd_text[:context_limit * 2]
            # Truncate message history if too long
            total_message_tokens = sum(estimate_tokens(m.get('content','')) for m in messages)
            while total_message_tokens + max_tokens > context_limit:
                if messages:
                    messages.pop(0)
                    total_message_tokens = sum(estimate_tokens(m.get('content','')) for m in messages)
                else:
                    break
            return messages, prd_text

        # Monkey-patch backend's OpenAI API call if possible
        if hasattr(backend, 'call_openai_api'):
            orig_call = backend.call_openai_api
            def safe_call_openai_api(messages, prd_text, *args, **kwargs):
                safe_messages, safe_prd = truncate_for_context(messages, prd_text, max_tokens=safe_max_tokens)
                return orig_call(safe_messages, safe_prd, *args, **kwargs)
            backend.call_openai_api = safe_call_openai_api
            logger.info("Patched backend.call_openai_api for strict context length checks")
        elif hasattr(backend, 'openai_api_call'):
            orig_call = backend.openai_api_call
            def safe_openai_api_call(messages, prd_text, *args, **kwargs):
                safe_messages, safe_prd = truncate_for_context(messages, prd_text, max_tokens=safe_max_tokens)
                return orig_call(safe_messages, safe_prd, *args, **kwargs)
            backend.openai_api_call = safe_openai_api_call
            logger.info("Patched backend.openai_api_call for strict context length checks")
        else:
            logger.warning("No backend OpenAI API call found to patch for context length checks")

        logger.info("Tabbed backend module imported successfully")
        return backend
        
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        return None
    except Exception as e:
        logger.error(f"Unexpected error importing backend: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def main():
    """Main launcher function."""
    print("🚀 TABBED WORKBENCH LAUNCHER")
    print("=" * 40)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set up logging
    logger = setup_logging()
    logger.info("Tabbed Workbench Launcher started")
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed")
        return False
    
    print("✅ Environment check passed")
    
    # Import backend safely
    backend = safe_import_backend()
    if not backend:
        print("❌ Failed to import backend module")
        return False
    
    print("✅ Backend module loaded")
    
    # Get configuration
    port = int(os.environ.get("PORT", 5002))  # Different port from three-section
    debug = os.environ.get("FLASK_ENV", "development").lower() != "production"
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"📊 Configuration:")
    print(f"   Port: {port}")
    print(f"   Debug: {debug}")
    print(f"   Host: {host}")
    print(f"   Environment: {os.environ.get('FLASK_ENV', 'development')}")
    print(f"   Landing Page: http://{host}:{port}/")
    print(f"   Tabbed Workbench: http://{host}:{port}/tabbed-layout")
    
    # Add API endpoint for Vector DB PRDs and proxy for file listing
    from flask import Flask, jsonify, Response, request
    import requests as pyrequests
    app = backend.app
    @app.route('/api/vector-db-prds')
    def vector_db_prds():
        # Example: Replace with actual Vector DB query
        prds = [
            {"name": "PRD v1", "url": "https://vectordb.example.com/prd/v1"},
            {"name": "PRD v2", "url": "https://vectordb.example.com/prd/v2"}
        ]
        return jsonify({"prds": prds})

    # Proxy endpoint to fetch file list from Vector DB backend
    @app.route('/vector-db/files/', methods=['GET'])
    def proxy_vector_db_files():
        logger = logging.getLogger(__name__)
        try:
            logger.info('Proxying request to http://localhost:5001/vector-db/files/')
            resp = pyrequests.get('http://localhost:5001/vector-db/files/', timeout=5)
            logger.info(f'Response from vector DB: {resp.status_code} {resp.text[:200]}')
            return Response(resp.content, status=resp.status_code, content_type=resp.headers.get('Content-Type', 'application/json'))
        except pyrequests.ConnectionError as ce:
            logger.error(f'Connection error to vector DB: {ce}')
            return jsonify({'files': [], 'error': 'Could not connect to Vector DB backend at http://localhost:5001/vector-db/files/'}), 502
        except Exception as e:
            logger.error(f'Error proxying vector DB files: {e}')
            return jsonify({'files': [], 'error': str(e)}), 500

    # Route for Product Management Intermediate page
    from flask import render_template
    @app.route('/product-management')
    def product_management_intermediate():
        return render_template('product_management_intermediate.html')

    # Route for Software Development Intermediate page
    @app.route('/development-intermediate')
    def development_intermediate():
        return render_template('development_intermediate.html')
    # Route for Planning Intermediate page
    @app.route('/planning-intermediate')
    def planning_intermediate():
        return render_template('planning_intermediate.html')

    # Route for Architecture Intermediate page
    @app.route('/architecture-intermediate')
    def architecture_intermediate():
        return render_template('architecture_intermediate.html')

    # Route for Migration Intermediate page
    @app.route('/migration-intermediate')
    def migration_intermediate():
        return render_template('migration_intermediate.html')

    # --- PATCH: Auto-generate epics if missing when generating user stories ---
    from flask import session, jsonify, request
    @app.route('/generate-user-stories', methods=['POST'])
    def generate_user_stories():
        epics = session.get('epics')
        if not epics:
            product = session.get('product') or request.form.get('product')
            feature = session.get('feature') or request.form.get('feature')
            epics = [
                f"Epic for {product or 'Product'}: Core Functionality",
                f"Epic for {feature or 'Feature'}: Advanced Features"
            ]
            session['epics'] = epics
        user_stories = []
        for epic in epics:
            user_stories.append(f"As a user, I want {epic.lower()} so that I get value.")
        session['user_stories'] = user_stories
        return jsonify({"epics": epics, "user_stories": user_stories})

    # Start server with enhanced error handling
    try:
        logger.info(f"Starting Flask server on port {port}")
        print(f"\n🌐 Server starting on http://localhost:{port}")
        print("   🏠 Landing Page: http://localhost:{port}/")
        print("   📋 Tabbed Workbench: http://localhost:{port}/tabbed-layout")
        print("   ❤️  Health check: http://localhost:{port}/health")
        print("   🐛 Debug info: http://localhost:{port}/debug-info")
        print("\n📝 Logs are saved to: tabbed_backend.log")
        print("⏹️  Press Ctrl+C to stop the server")
        print("=" * 40)
        backend.app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        logger.info("Server stopped by user (Ctrl+C)")
        print("\n⏹️  Server stopped by user")
        return True
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"\n❌ Server error: {str(e)}")
        print("Check tabbed_backend.log for details")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Launcher error: {str(e)}")
        print("Check tabbed_backend.log for details")
        exit(1)
