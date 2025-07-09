#!/usr/bin/env python3
"""
Test script to verify tabbed layout epic generation functionality
"""

import requests
import json

def test_epic_display():
    """Test the complete epic generation and display flow."""
    print("🧪 Testing Tabbed Layout Epic Generation & Display")
    print("=" * 50)
    
    base_url = "http://localhost:5001"
    
    # Test data
    test_requirements = """
    Project: E-commerce Customer Management Platform
    
    Requirements:
    1. User Registration & Authentication
       - Secure user signup with email verification
       - Social media login integration (Google, Facebook)
       - Password reset functionality
       - Two-factor authentication support
    
    2. Customer Profile Management
       - Personal information management
       - Address book with multiple addresses
       - Purchase history tracking
       - Preference settings
    
    3. Product Catalog & Search
       - Product browsing with categories
       - Advanced search and filtering
       - Product recommendations
       - Wishlist functionality
    
    4. Shopping Cart & Checkout
       - Add/remove items from cart
       - Multiple payment methods (Credit card, PayPal, Apple Pay)
       - Order tracking and notifications
       - Guest checkout option
    
    5. Customer Support System
       - Live chat functionality
       - Support ticket creation and tracking
       - FAQ and knowledge base
       - Return and refund management
    
    6. Admin Dashboard
       - Sales analytics and reporting
       - Customer management tools
       - Inventory management
       - Order fulfillment tracking
    """
    
    try:
        print("Step 1: Generating epics and user stories...")
        
        # Test epic generation
        response = requests.post(f"{base_url}/generate-epics-from-text", 
                               json={
                                   "text": test_requirements,
                                   "input_type": "requirements"
                               }, 
                               timeout=60)
        
        print(f"Epic generation response: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Generated {data.get('epic_count', 0)} epics and {data.get('story_count', 0)} user stories")
                
                # Test getting epics
                print("\nStep 2: Retrieving generated epics...")
                epics_response = requests.get(f"{base_url}/get-epics")
                
                if epics_response.status_code == 200:
                    epics_data = epics_response.json()
                    if epics_data.get('success'):
                        epics = epics_data.get('epics', [])
                        print(f"✅ Retrieved {len(epics)} epics")
                        
                        # Display epics
                        for i, epic in enumerate(epics, 1):
                            print(f"\n📋 Epic {i}: {epic.get('title', 'No Title')}")
                            print(f"   Priority: {epic.get('priority', 'Unknown')}")
                            print(f"   ID: {epic.get('id', 'No ID')}")
                            print(f"   Description: {epic.get('description', 'No description')[:100]}...")
                        
                        # Test getting user stories for first epic
                        if epics:
                            first_epic_id = epics[0].get('id')
                            print(f"\nStep 3: Retrieving user stories for epic {first_epic_id}...")
                            
                            stories_response = requests.get(f"{base_url}/get-user-stories?epic_id={first_epic_id}")
                            
                            if stories_response.status_code == 200:
                                stories_data = stories_response.json()
                                if stories_data.get('success'):
                                    stories = stories_data.get('user_stories', [])
                                    print(f"✅ Retrieved {len(stories)} user stories for epic {first_epic_id}")
                                    
                                    for i, story in enumerate(stories, 1):
                                        print(f"\n   📝 Story {i}: {story.get('title', 'No Title')}")
                                        print(f"      Priority: {story.get('priority', 'Unknown')}")
                                        print(f"      Points: {story.get('story_points', 'Unknown')}")
                                        print(f"      Description: {story.get('description', 'No description')[:80]}...")
                                else:
                                    print(f"❌ Failed to get user stories: {stories_data.get('error')}")
                            else:
                                print(f"❌ HTTP error getting user stories: {stories_response.status_code}")
                    else:
                        print(f"❌ Failed to get epics: {epics_data.get('error')}")
                else:
                    print(f"❌ HTTP error getting epics: {epics_response.status_code}")
            else:
                print(f"❌ Epic generation failed: {data.get('error')}")
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the Flask app is running on localhost:5001")
        print("   Start the server with: python poc2_backend_processor_three_section.py")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_epic_display()
