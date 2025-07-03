#!/usr/bin/env python3
"""
Three Section Server Troubleshooting Script
Helps diagnose why the server stops when UI is accessed
"""

import subprocess
import time
import requests
import sys
import os
from datetime import datetime

def check_server_running(port=5001):
    """Check if server is running on the specified port."""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_endpoints():
    """Test critical endpoints that might cause crashes."""
    base_url = "http://localhost:5001"
    
    endpoints = [
        ("/health", "GET"),
        ("/debug-info", "GET"),
        ("/three-section", "GET"),
        ("/three-section-get-epics", "GET")
    ]
    
    print("🔍 Testing endpoints...")
    for endpoint, method in endpoints:
        try:
            print(f"Testing {method} {endpoint}...")
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
            
            print(f"  ✅ Status: {response.status_code}")
            if endpoint == "/debug-info":
                data = response.json()
                print(f"  📊 Session keys: {data.get('session_keys', [])}")
                print(f"  📊 Epic count: {data.get('epic_count', 0)}")
                print(f"  📊 OpenAI configured: {data.get('openai_configured', False)}")
            
        except requests.exceptions.ConnectionError:
            print(f"  ❌ Connection failed - server stopped!")
            return False
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            return False
    
    return True

def check_python_environment():
    """Check Python environment for potential issues."""
    print("🐍 Checking Python environment...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check critical imports
    critical_imports = [
        "flask", "openai", "json", "os", "logging", "traceback"
    ]
    
    for module in critical_imports:
        try:
            __import__(module)
            print(f"  ✅ {module} - OK")
        except ImportError:
            print(f"  ❌ {module} - MISSING")
            return False
    
    return True

def check_environment_variables():
    """Check environment variables."""
    print("🌍 Checking environment variables...")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print(f"  ✅ OPENAI_API_KEY - Set (length: {len(openai_key)})")
    else:
        print(f"  ⚠️  OPENAI_API_KEY - Not set (AI features will be limited)")
    
    flask_debug = os.getenv('FLASK_DEBUG')
    print(f"  📊 FLASK_DEBUG: {flask_debug}")

def monitor_server_startup():
    """Monitor server startup and detect when it crashes."""
    print("👁️  Monitoring server startup...")
    
    # Wait for server to start
    for i in range(10):
        if check_server_running():
            print(f"  ✅ Server is running (attempt {i+1})")
            break
        print(f"  ⏳ Waiting for server... (attempt {i+1})")
        time.sleep(2)
    else:
        print("  ❌ Server failed to start after 20 seconds")
        return False
    
    # Test endpoints
    print("\n🧪 Testing endpoints that might cause crashes...")
    return test_endpoints()

def main():
    """Main troubleshooting routine."""
    print("🔧 THREE SECTION SERVER TROUBLESHOOTING")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Basic environment checks
    if not check_python_environment():
        print("❌ Python environment issues detected")
        return False
    
    check_environment_variables()
    
    # Check if server is already running
    if check_server_running():
        print("\n✅ Server is already running!")
        print("Testing endpoints...")
        success = test_endpoints()
        if success:
            print("\n🎉 All endpoints working correctly!")
            print("🌐 Try accessing: http://localhost:5001/three-section")
        return success
    
    print("\n⚠️  Server is not running.")
    print("Please start the server first with:")
    print("  python poc2_backend_processor_three_section.py")
    print("\nThen run this script again to test endpoints.")
    
    return False

def run_diagnostic():
    """Run comprehensive diagnostic."""
    print("\n🔬 RUNNING COMPREHENSIVE DIAGNOSTIC")
    print("=" * 50)
    
    # Check if main backend file exists
    backend_file = "poc2_backend_processor_three_section.py"
    if not os.path.exists(backend_file):
        print(f"❌ Backend file not found: {backend_file}")
        return False
    print(f"✅ Backend file found: {backend_file}")
    
    # Check if template exists
    template_file = "templates/poc2_three_section_layout.html"
    if not os.path.exists(template_file):
        print(f"❌ Template file not found: {template_file}")
        return False
    print(f"✅ Template file found: {template_file}")
    
    # Check file sizes
    backend_size = os.path.getsize(backend_file)
    template_size = os.path.getsize(template_file)
    print(f"📊 Backend file size: {backend_size:,} bytes")
    print(f"📊 Template file size: {template_size:,} bytes")
    
    return True

if __name__ == "__main__":
    print("Starting troubleshooting...")
    
    # Run diagnostic first
    if not run_diagnostic():
        exit(1)
    
    # Run main troubleshooting
    success = main()
    
    if not success:
        print("\n💡 TROUBLESHOOTING TIPS:")
        print("1. Make sure OPENAI_API_KEY is set")
        print("2. Install requirements: pip install -r requirements.txt")
        print("3. Start server: python poc2_backend_processor_three_section.py")
        print("4. Check console output for error messages")
        print("5. Try accessing health check: http://localhost:5001/health")
    
    exit(0 if success else 1)
