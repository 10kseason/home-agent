#!/usr/bin/env python3
"""
간단한 tool handler 테스트
"""
import requests
import time

# 직접 assist_control의 _post 함수를 모방
def test_agent_connection():
    event_url = "http://127.0.0.1:8765/event"
    headers = {"Content-Type": "application/json"}
    
    test_cases = [
        "mictrans.start",
        "capture_assist.start",
        "stt.start"
    ]
    
    for event_type in test_cases:
        print(f"\n=== Testing {event_type} ===")
        try:
            payload = {"ts": time.time()}
            response = requests.post(
                event_url, 
                json={"type": event_type, "payload": payload}, 
                headers=headers, 
                timeout=3
            )
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                print("✅ SUCCESS: Agent received the event")
            else:
                print("❌ FAILED: Agent did not accept the event")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_agent_connection()