#!/usr/bin/env python3
"""
Startup script for the Three Section UI and Backend System
This script helps users easily start the new three-section system
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """Check if all requirements are met."""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    print("✅ requirements.txt found")
    
    # Check if main backend file exists
    if not Path("poc2_backend_processor_three_section.py").exists():
        print("❌ poc2_backend_processor_three_section.py not found")
        return False
    print("✅ Backend file found")
    
    # Check if template exists
    if not Path("templates/poc2_three_section_layout.html").exists():
        print("❌ three-section template not found")
        return False
    print("✅ Template file found")
    
    return True

def check_environment():
    """Check environment variables."""
    print("\n🌍 Checking environment variables...")
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set - AI features will not work")
        api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print("✅ OpenAI API key set for this session")
        else:
            print("⚠️  Continuing without OpenAI API key")
    else:
        print("✅ OPENAI_API_KEY is set")
    
    # Set Flask environment for development
    os.environ["FLASK_DEBUG"] = "true"
    print("✅ Flask debug mode enabled")

def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Try to install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Dependency installation failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Dependency installation timed out")
        return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {str(e)}")
        return False

def start_server():
    """Start the three-section backend server."""
    print("\n🚀 Starting three-section backend server...")
    print("   Server will run on: http://localhost:5001")
    print("   Press Ctrl+C to stop the server")
    print("\n" + "="*60)
    
    try:
        # Start the server
        subprocess.run([
            sys.executable, "poc2_backend_processor_three_section.py"
        ])
    except KeyboardInterrupt:
        print("\n\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        return False
    
    return True

def main():
    """Main startup routine."""
    print("=" * 60)
    print("🎯 THREE SECTION UI & BACKEND SYSTEM STARTUP")
    print("=" * 60)
    print("This script will help you start the new three-section system")
    print("that operates independently from the existing application.")
    print()
    
    # Check system requirements
    if not check_requirements():
        print("\n❌ System requirements not met. Please fix the issues above.")
        return False
    
    # Check environment
    check_environment()
    
    # Ask about dependency installation
    install_deps = input("\n🤔 Install/update dependencies? (y/N): ").strip().lower()
    if install_deps in ('y', 'yes'):
        if not install_dependencies():
            print("\n❌ Dependency installation failed. Please install manually:")
            print("   pip install -r requirements.txt")
            return False
    
    print("\n✅ System ready to start!")
    print("\n📝 Quick Guide:")
    print("   1. The server will start on http://localhost:5001")
    print("   2. Navigate to http://localhost:5001/three-section for the new UI")
    print("   3. The three-section layout workflow:")
    print("      - Upload a PRD file or enter requirements manually")
    print("      - Generated epics appear (left section) → Click to select")
    print("      - User stories appear (middle section) → Click to select")
    print("      - Story details appear (right section) with acceptance criteria")
    print("   4. Use chat features for refinement at any level")
    print("   5. Submit polished stories to Jira when ready")
    print("\n💡 Alternative: For a tabbed interface experience:")
    print("   - Run: python launch_tabbed_workbench.py")
    print("   - Access: http://localhost:5002/tabbed-layout")
    
    start_now = input("\n🚀 Start the server now? (Y/n): ").strip().lower()
    if start_now not in ('n', 'no'):
        return start_server()
    else:
        print("\n💡 To start manually, run:")
        print("   python poc2_backend_processor_three_section.py")
        return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Startup cancelled by user")
        exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        exit(1)
