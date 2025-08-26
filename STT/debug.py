import os
import sys
print("=== SUBPROCESS ENVIRONMENT ===")
print(f"Python: {sys.executable}")
print(f"Working Dir: {os.getcwd()}")
print(f"Script Location: {__file__}")
print(f"AGENT_EVENT_URL: {os.environ.get('AGENT_EVENT_URL', 'NOT SET')}")

print("\n=== MODULE TEST ===")
try:
    import numpy
    print("numpy: OK")
    import librosa  
    print("librosa: OK")
    from diarizer import SpeakerDiarizer
    print("diarizer: OK")
    print("All imports successful!")
except Exception as e:
    print(f"Import failed: {e}")

print("\n=== FILES CHECK ===")
print(f"VSRG-Ts-to-kr.py exists: {os.path.exists('VSRG-Ts-to-kr.py')}")
print(f"diarizer.py exists: {os.path.exists('diarizer.py')}")
print(f"config.yaml exists: {os.path.exists('config.yaml')}")

input("Press Enter to exit...")  # 창이 바로 안 닫히게