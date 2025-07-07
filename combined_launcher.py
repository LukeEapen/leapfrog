#!/usr/bin/env python3
"""
Combined launcher for both Flask applications
Starts both the optimized backend (port 5000) and three-section backend (port 5001) simultaneously
"""

import os
import sys
import logging
import traceback
import threading
import time
from datetime import datetime

def setup_logging():
    """Set up enhanced logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('combined_launcher.log'),
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
        'poc2_backend_processor_optimized.py',
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

def start_optimized_backend():
    """Start the optimized backend on port 5000."""
    logger = logging.getLogger('optimized_backend')
    
    try:
        logger.info("Starting optimized backend...")
        
        # Add current directory to Python path
        sys.path.insert(0, os.getcwd())
        
        # Import and start the optimized backend
        import poc2_backend_processor_optimized as optimized_backend
        
        logger.info("Optimized backend module imported successfully")
        logger.info("Starting optimized backend Flask server on port 5000")
        
        # Start the Flask app
        optimized_backend.app.run(
            debug=False,
            host="127.0.0.1",
            port=5000,
            threaded=True,
            use_reloader=False
        )
        
    except Exception as e:
        logger.error(f"Error in optimized backend: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def start_three_section_backend():
    """Start the three-section backend on port 5001."""
    logger = logging.getLogger('three_section_backend')
    
    try:
        logger.info("Starting three-section backend...")
        
        # Add current directory to Python path
        sys.path.insert(0, os.getcwd())
        
        # Import the three-section backend
        import poc2_backend_processor_three_section as three_section_backend
        
        logger.info("Three-section backend module imported successfully")
        logger.info("Starting three-section backend Flask server on port 5001")
        
        # Start the Flask app
        three_section_backend.app.run(
            debug=False,
            host="127.0.0.1",
            port=5001,
            threaded=True,
            use_reloader=False
        )
        
    except Exception as e:
        logger.error(f"Error in three-section backend: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def main():
    """Main launcher function."""
    print("🚀 COMBINED FLASK APPLICATIONS LAUNCHER")
    print("=" * 50)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set up logging
    logger = setup_logging()
    logger.info("Combined launcher started")
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed")
        return False
    
    print("✅ Environment check passed")
    
    # Configuration
    print(f"📊 Configuration:")
    print(f"   Optimized Backend: http://127.0.0.1:5000")
    print(f"   Three-Section Backend: http://127.0.0.1:5001/three-section")
    print(f"   Environment: {os.environ.get('FLASK_ENV', 'development')}")
    
    print(f"\n🌐 Starting both Flask applications...")
    print("   📋 Optimized Backend (Port 5000) - Main application")
    print("   📋 Three-Section Backend (Port 5001) - Three-section layout")
    print(f"\n📝 Logs are saved to: combined_launcher.log")
    print("⏹️  Press Ctrl+C to stop both servers")
    print("=" * 50)
    
    # Start both backends in separate threads
    try:
        # Create threads for both backends
        optimized_thread = threading.Thread(
            target=start_optimized_backend, 
            name="OptimizedBackend",
            daemon=True
        )
        
        three_section_thread = threading.Thread(
            target=start_three_section_backend, 
            name="ThreeSectionBackend",
            daemon=True
        )
        
        # Start both threads
        logger.info("Starting optimized backend thread...")
        optimized_thread.start()
        
        # Small delay to prevent port conflicts during startup
        time.sleep(2)
        
        logger.info("Starting three-section backend thread...")
        three_section_thread.start()
        
        print("\n✅ Both applications started successfully!")
        print("\n🔗 Access URLs:")
        print("   • Optimized Backend: http://localhost:5000")
        print("   • Three-Section UI: http://localhost:5001/three-section")
        print("\n💡 Both applications are running. Use the URLs above to access them.")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
                # Check if both threads are still alive
                if not optimized_thread.is_alive():
                    logger.error("Optimized backend thread died")
                    print("❌ Optimized backend stopped unexpectedly")
                    break
                    
                if not three_section_thread.is_alive():
                    logger.error("Three-section backend thread died")
                    print("❌ Three-section backend stopped unexpectedly")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Received shutdown signal (Ctrl+C)")
            print("\n⏹️  Shutting down both servers...")
            print("   Applications will stop gracefully...")
            return True
            
    except Exception as e:
        logger.error(f"Error starting applications: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"\n❌ Error starting applications: {str(e)}")
        print("Check combined_launcher.log for details")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Launcher error: {str(e)}")
        print("Check combined_launcher.log for details")
        exit(1)
