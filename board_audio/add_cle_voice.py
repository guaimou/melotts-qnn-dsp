"""Add default_cle voice to TTS module."""
tts_path = '/home/fibo/AI model/tts_models/tts.py'
with open(tts_path, 'r') as f:
    content = f.read()

# 1. Add .so map
old = '"default_ada_sqnr": "/home/fibo/melotts_qnn/lib_adaround_sqnr/aarch64-ubuntu-gcc9.4/libmelotts_adaround_sqnr.so",'
new_so = old + '\n    "default_cle": "/home/fibo/melotts_qnn/lib_cle_sqnr/aarch64-ubuntu-gcc9.4/libmelotts_cle_sqnr.so",'
content = content.replace(old, new_so)

# 2. Add voice map
old_v = '"default_ada_sqnr": (os.path.join(MELO_MODELS_DIR, "default_zh", "G_default.pth"), 1),'
new_v = old_v + '\n    "default_cle": (os.path.join(MELO_MODELS_DIR, "default_zh", "G_default.pth"), 1),'
content = content.replace(old_v, new_v)

# 3. Add QNN params
old_p = '"default_ada_sqnr": {"in_scale": 0.0635628774762154, "in_offset": -133, "out_scale": 0.0016934585291892, "out_offset": -117},'
new_p = old_p + '\n    "default_cle": {"in_scale": 0.0635628774762154, "in_offset": -133, "out_scale": 0.0016634501516819, "out_offset": -117},'
content = content.replace(old_p, new_p)

with open(tts_path, 'w') as f:
    f.write(content)

print("Added default_cle voice")
for line in content.split('\n'):
    if 'default_cle' in line:
        print(f"  {line.strip()}")
