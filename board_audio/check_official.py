"""Check official FiboTTS model backend path."""
import sys
sys.path.insert(0, "/home/fibo/AI model/tts_models")
sys.path.insert(0, "/home/fibo/AI model/llm_models/get_output_from_log_method")
from config import TTS_MODEL

print("Model path:", TTS_MODEL)

import os
# Check the .fmodel directory
model_dir = os.path.dirname(TTS_MODEL)
print("Model dir:", model_dir)
for f in os.listdir(model_dir):
    fpath = os.path.join(model_dir, f)
    size = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
    print(f"  {f} ({size/1024/1024:.1f}MB)" if size > 0 else f"  {f} (dir)")

# Check if it uses .fmodel extension
print("\nFmodel files:")
for root, dirs, files in os.walk(TTS_MODEL):
    for f in files:
        if f.endswith('.fmodel') or f.endswith('.bin') or f.endswith('.so'):
            fpath = os.path.join(root, f)
            print(f"  {fpath} ({os.path.getsize(fpath)/1024/1024:.1f}MB)")
    break  # just top level
