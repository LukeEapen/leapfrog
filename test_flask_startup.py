#!/usr/bin/env python3
"""
Simple test to verify Flask app can start.
"""

import logging
logging.basicConfig(level=logging.INFO)

try:
    from poc2_backend_processor_three_section import app
    print("✅ Flask app imported successfully")
    
    print("🚀 Starting Flask app on port 5001...")
    app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
    
except Exception as e:
    print(f"❌ Error starting Flask app: {e}")
    import traceback
    traceback.print_exc()
