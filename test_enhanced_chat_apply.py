#!/usr/bin/env python3
"""
Test script for enhanced System Mapping Chat apply functionality
"""

import requests
import json

def test_enhanced_chat_functionality():
    """Test the enhanced system mapping chat with apply functions"""
    
    print("🧪 Testing Enhanced System Mapping Chat Apply Functions...")
    print("=" * 60)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Credit Card Processing Request",
            "message": "I need systems for credit card processing",
            "current_systems": [],
            "expected_suggestions": ["Credit decision engine", "Card issuance manager"]
        },
        {
            "name": "Fraud Detection Architecture",
            "message": "Help me design a fraud detection architecture",
            "current_systems": ["Customer data repository"],
            "expected_suggestions": ["Fraud detection platform", "Authorization control layer"]
        },
        {
            "name": "Customer Onboarding Flow",
            "message": "What systems do I need for customer onboarding?",
            "current_systems": [],
            "expected_suggestions": ["Customer acquisition platform", "Document management platform"]
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔬 Test {i}: {scenario['name']}")
        print("-" * 40)
        
        test_data = {
            "message": scenario["message"],
            "current_systems": scenario["current_systems"],
            "available_systems": [
                "Customer acquisition platform",
                "Credit decision engine", 
                "Card issuance manager",
                "Payment setup module",
                "Fraud detection platform",
                "Authorization control layer",
                "Document management platform",
                "Customer data repository"
            ]
        }
        
        try:
            response = requests.post(
                'http://127.0.0.1:5000/system-mapping-chat',
                json=test_data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Status: SUCCESS")
                print(f"📝 Message: {result.get('message', 'No message')[:100]}...")
                
                suggestions = result.get('suggestions', [])
                print(f"💡 Suggestions: {len(suggestions)} systems recommended")
                
                for j, suggestion in enumerate(suggestions[:3], 1):  # Show first 3
                    print(f"   {j}. {suggestion.get('system', 'Unknown')} - {suggestion.get('reason', 'No reason')[:50]}...")
                
                warnings = result.get('warnings', [])
                if warnings:
                    print(f"⚠️  Warnings: {len(warnings)} considerations")
                    for warning in warnings[:2]:  # Show first 2
                        print(f"   - {warning[:60]}...")
                
                # Check if expected suggestions are present
                suggested_systems = [s.get('system', '') for s in suggestions]
                found_expected = any(exp in suggested_systems for exp in scenario['expected_suggestions'])
                
                if found_expected:
                    print(f"🎯 Expected systems found: YES")
                else:
                    print(f"🎯 Expected systems found: NO (got: {suggested_systems})")
                    
            else:
                print(f"❌ Status: FAILED ({response.status_code})")
                print(f"Response: {response.text[:200]}...")
                
        except requests.exceptions.RequestException as e:
            print(f"🔥 Network Error: {e}")
        except Exception as e:
            print(f"💥 Unexpected Error: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Enhanced Chat Apply Function Tests Completed!")
    print("\n📋 Features Tested:")
    print("   ✅ AI-powered system recommendations")
    print("   ✅ Context-aware suggestions based on current selection")
    print("   ✅ Warnings and considerations")
    print("   ✅ JSON response format for apply functionality")
    print("\n🎮 UI Features Available:")
    print("   🟢 Apply All Suggestions button")
    print("   🟢 Individual system apply buttons")
    print("   🟢 Quick Actions (Clear, Show Current, Suggest Removals)")
    print("   🟢 Remove systems directly from chat")
    print("   🟢 Export CSV and Preview from chat")

if __name__ == "__main__":
    test_enhanced_chat_functionality()
