#!/usr/bin/env python3
"""
Tool handler 직접 테스트 스크립트
"""
import sys
import os

# overlay tools path 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'Overlay', 'plugins', 'tools'))

try:
    from tool.assist_control import TOOL_HANDLERS
    
    print("=== Available Tool Handlers ===")
    for name, handler in TOOL_HANDLERS.items():
        print(f"  {name}: {handler}")
    
    print("\n=== Testing mictrans_start ===")
    if 'mictrans_start' in TOOL_HANDLERS:
        handler = TOOL_HANDLERS['mictrans_start']
        result = handler({})
        print(f"Result: {result}")
    else:
        print("mictrans_start handler not found!")
    
    print("\n=== Testing capture_assist_start ===")
    if 'capture_assist_start' in TOOL_HANDLERS:
        handler = TOOL_HANDLERS['capture_assist_start']
        result = handler({})
        print(f"Result: {result}")
    else:
        print("capture_assist_start handler not found!")
        
    print("\n=== Testing stt.start (should map to mictrans) ===")
    if 'stt.start' in TOOL_HANDLERS:
        handler = TOOL_HANDLERS['stt.start']
        result = handler({})
        print(f"Result: {result}")
    else:
        print("stt.start handler not found!")
        
except Exception as e:
    print(f"Error importing tool handlers: {e}")
    import traceback
    traceback.print_exc()