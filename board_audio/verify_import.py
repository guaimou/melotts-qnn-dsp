import sys
sys.path.insert(0, "/home/fibo/AI model/tts_models")
import tts
print("tts.py imported OK")
print("voices:", list(tts.QNN_VOICE_MAP.keys()))
