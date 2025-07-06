#!/usr/bin/env python3
"""
Safe launcher for Three Section Backend with enhanced error handling
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
            logging.FileHandler('three_section_debug.log'),
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
        'poc2_backend_processor_three_section.py',
        'templates/poc2_three_section_layout.html'
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
        logger.info("Importing three section backend...")
        
        # Add current directory to Python path
        sys.path.insert(0, os.getcwd())
        
        # Import the backend module
        import poc2_backend_processor_three_section as backend
        
        logger.info("Backend module imported successfully")
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
    print("🚀 THREE SECTION BACKEND LAUNCHER")
    print("=" * 40)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set up logging
    logger = setup_logging()
    logger.info("Three Section Backend Launcher started")
    
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
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_ENV", "development").lower() != "production"
    host = "0.0.0.0" if os.environ.get("FLASK_ENV") == "production" else "127.0.0.1"
    
    print(f"📊 Configuration:")
    print(f"   Port: {port}")
    print(f"   Debug: {debug}")
    print(f"   Host: {host}")
    print(f"   Environment: {os.environ.get('FLASK_ENV', 'development')}")
    print(f"   URL: http://{host}:{port}/three-section")
    
    # Start server with enhanced error handling
    try:
        logger.info(f"Starting Flask server on port {port}")
        print(f"\n🌐 Server starting on http://localhost:{port}")
        print("   Main UI: http://localhost:{port}/three-section")
        print("   Health check: http://localhost:{port}/health")
        print("   Debug info: http://localhost:{port}/debug-info")
        print("\n📝 Logs are saved to: three_section_debug.log")
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
        print("Check three_section_debug.log for details")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Launcher error: {str(e)}")
        print("Check three_section_debug.log for details")
        exit(1)
