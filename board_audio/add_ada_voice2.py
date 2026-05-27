import re
with open('/home/fibo/AI model/tts_models/tts.py', 'r') as f:
    content = f.read()

old = '"a13":     (os.path.join(MELO_MODELS_DIR, "a13", "G_34400.pth"), 0),'
new = '"a13":     (os.path.join(MELO_MODELS_DIR, "a13", "G_34400.pth"), 0),\n    "default_ada": (os.path.join(MELO_MODELS_DIR, "default_zh", "G_default.pth"), 1),'
content = content.replace(old, new)

with open('/home/fibo/AI model/tts_models/tts.py', 'w') as f:
    f.write(content)
print("QNN_VOICE_MAP updated: default_ada added")

# Verify
import re
matches = re.findall(r'"default_ada".*', content)
for m in matches:
    print(" ", m)
