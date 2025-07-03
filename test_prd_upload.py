#!/usr/bin/env python3
"""
Test PRD Upload functionality for the Three Section System
"""

import requests
import json
import os
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:5001"
TEST_TIMEOUT = 60

def test_prd_upload():
    """Test PRD upload with sample content."""
    print("🧪 Testing PRD Upload Functionality")
    print("=" * 50)
    
    # Create a sample PRD content
    sample_prd = """
PRODUCT REQUIREMENTS DOCUMENT
=============================

Project: E-commerce Platform

OVERVIEW
--------
Build a modern e-commerce platform for online retail business.

FEATURES
--------
1. User Authentication
   - User registration and login
   - Password reset functionality
   - Social media login integration

2. Product Management
   - Product catalog browsing
   - Search and filtering
   - Product recommendations

3. Shopping Cart
   - Add/remove items
   - Quantity management
   - Save for later functionality

4. Payment Processing
   - Multiple payment methods
   - Secure checkout process
   - Order confirmation

5. Order Management
   - Order tracking
   - Order history
   - Return/refund process

6. Admin Dashboard
   - Inventory management
   - Sales analytics
   - User management

TECHNICAL REQUIREMENTS
---------------------
- Mobile responsive design
- High performance and scalability
- Security compliance
- SEO optimization

USER STORIES
-----------
- As a customer, I want to browse products easily
- As a customer, I want to make secure purchases
- As an admin, I want to manage inventory efficiently
"""

    try:
        # Test manual requirements input
        print("Testing manual requirements input...")
        response = requests.post(
            f"{BASE_URL}/three-section-upload-prd",
            data={"requirements": sample_prd},
            timeout=TEST_TIMEOUT
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                epics = data.get('epics', [])
                print(f"✅ Success! Generated {len(epics)} epics")
                
                for i, epic in enumerate(epics, 1):
                    print(f"\nEpic {i}:")
                    print(f"  Title: {epic.get('title', 'N/A')}")
                    print(f"  Priority: {epic.get('priority', 'N/A')}")
                    print(f"  Description: {epic.get('description', 'N/A')[:100]}...")
                
                return True
            else:
                print(f"❌ API returned error: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - is the server running on port 5001?")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_file_upload():
    """Test file upload functionality."""
    print("\n🧪 Testing File Upload Functionality")
    print("=" * 50)
    
    # Create a temporary PRD file
    prd_content = """
MOBILE APP PRD
==============

App Name: TaskMaster Pro

OVERVIEW
--------
A productivity app for task management and team collaboration.

CORE FEATURES
-------------
1. Task Creation and Management
2. Team Collaboration
3. Progress Tracking
4. Notifications and Reminders
5. Analytics Dashboard

USER STORIES
-----------
- As a user, I want to create and organize tasks
- As a team lead, I want to assign tasks to team members
- As a user, I want to track my progress over time
"""
    
    try:
        # Write temporary file
        with open("temp_prd.txt", "w") as f:
            f.write(prd_content)
        
        # Test file upload
        print("Testing file upload...")
        with open("temp_prd.txt", "rb") as f:
            files = {"prd_file": ("test_prd.txt", f, "text/plain")}
            response = requests.post(
                f"{BASE_URL}/three-section-upload-prd",
                files=files,
                timeout=TEST_TIMEOUT
            )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                epics = data.get('epics', [])
                print(f"✅ Success! Generated {len(epics)} epics from file")
                return True
            else:
                print(f"❌ API returned error: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    finally:
        # Clean up temporary file
        try:
            os.remove("temp_prd.txt")
        except:
            pass

def main():
    """Run PRD upload tests."""
    print("🚀 PRD Upload System Tests")
    print(f"Target URL: {BASE_URL}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n⚠️  Make sure the three-section server is running!")
    
    input("Press Enter to start tests...")
    
    tests = [
        ("Manual Requirements Input", test_prd_upload),
        ("File Upload", test_file_upload)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📝 Running: {test_name}")
        success = test_func()
        
        if success:
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            failed += 1
            print(f"❌ {test_name} FAILED")
    
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 All PRD upload tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
