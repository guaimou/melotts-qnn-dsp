"""Add AdaRound voice to TTS module and test."""
import re, sys

tts_path = '/home/fibo/AI model/tts_models/tts.py'
with open(tts_path, 'r') as f:
    content = f.read()

# Add .so map entry
old = '"a13":     "/home/fibo/melotts_qnn/lib_a13_pch/aarch64-ubuntu-gcc9.4/libmelotts_a13_pch.so",'
new = ('"a13":     "/home/fibo/melotts_qnn/lib_a13_pch/aarch64-ubuntu-gcc9.4/libmelotts_a13_pch.so",\n'
       '    "default_ada": "/home/fibo/melotts_qnn/lib_adaround/aarch64-ubuntu-gcc9.4/libmelotts_adaround.so",')
if old in content:
    content = content.replace(old, new)
    print("SO_MAP: added default_ada")
else:
    # Already added or different format
    if 'default_ada' in content:
        print("SO_MAP: default_ada already exists")

# Add voice map entry
old_v = '"a13":     (os.path.join(MELO_MODELS_DIR, "a13", "G_34400.pth"), 0),'
new_v = ('"a13":     (os.path.join(MELO_MODELS_DIR, "a13", "G_34400.pth"), 0),\n'
         '    "default_ada": (os.path.join(MELO_MODELS_DIR, "default_zh", "G_default.pth"), 1),')
if old_v in content:
    content = content.replace(old_v, new_v)
    print("VOICE_MAP: added default_ada")
elif 'default_ada' in content:
    print("VOICE_MAP: default_ada already exists")

with open(tts_path, 'w') as f:
    f.write(content)

print("\nVerify:")
for line in content.split('\n'):
    if 'default_ada' in line:
        print(f"  {line.strip()}")
