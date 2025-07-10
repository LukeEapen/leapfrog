#!/usr/bin/env python3
"""
Test script to validate the system mapping functionality in the tabbed backend
"""

import requests
import json
import os

def test_system_mapping():
    """Test the complete workflow including system mapping."""
    
    base_url = "http://localhost:5002"
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Server health check: {response.status_code}")
        if response.status_code != 200:
            print("❌ Server is not running properly")
            return False
    except Exception as e:
        print(f"❌ Server connection error: {e}")
        return False
    
    # Test 2: Upload system mapping file
    try:
        with open('test_system_mapping.csv', 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/upload-system-info", files=files)
            print(f"✅ System mapping upload: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Response: {data.get('message', 'No message')}")
            else:
                print(f"   Error: {response.text}")
                
    except FileNotFoundError:
        print("❌ test_system_mapping.csv file not found")
        return False
    except Exception as e:
        print(f"❌ System mapping upload error: {e}")
        return False
    
    # Test 3: Check system info status
    try:
        response = requests.get(f"{base_url}/get-system-info")
        print(f"✅ System info status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Has system info: {data.get('has_system_info', False)}")
            print(f"   Filename: {data.get('filename', 'N/A')}")
        else:
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ System info status error: {e}")
        return False
    
    # Test 4: Test file upload with both PRD and system mapping
    try:
        with open('test_prd.txt', 'rb') as prd_file, open('test_system_mapping.csv', 'rb') as sys_file:
            files = {
                'prd_file': prd_file,
                'system_mapping_file': sys_file
            }
            data = {
                'context_notes': 'Test context for system mapping validation'
            }
            
            response = requests.post(f"{base_url}/tabbed-upload-files", files=files, data=data)
            print(f"✅ Files upload test: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Success: {result.get('success', False)}")
                print(f"   Epics generated: {len(result.get('epics', []))}")
                
                # Print first epic as sample
                epics = result.get('epics', [])
                if epics:
                    first_epic = epics[0]
                    print(f"   Sample epic: {first_epic.get('title', 'No title')}")
            else:
                print(f"   Error: {response.text}")
                
    except FileNotFoundError as e:
        print(f"❌ Required file not found: {e}")
        return False
    except Exception as e:
        print(f"❌ Files upload error: {e}")
        return False
    
    print("\n🎉 System mapping validation completed!")
    return True

if __name__ == "__main__":
    print("🧪 SYSTEM MAPPING VALIDATION TEST")
    print("=" * 50)
    test_system_mapping()
