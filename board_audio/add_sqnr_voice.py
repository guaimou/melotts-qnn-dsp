"""Add default_ada_sqnr voice to TTS module."""
tts_path = '/home/fibo/AI model/tts_models/tts.py'
with open(tts_path, 'r') as f:
    content = f.read()

# 1. Add .so map
old = '"default_ada": "/home/fibo/melotts_qnn/lib_adaround/aarch64-ubuntu-gcc9.4/libmelotts_adaround.so",'
new_so = '"default_ada": "/home/fibo/melotts_qnn/lib_adaround/aarch64-ubuntu-gcc9.4/libmelotts_adaround.so",\n    "default_ada_sqnr": "/home/fibo/melotts_qnn/lib_adaround_sqnr/aarch64-ubuntu-gcc9.4/libmelotts_adaround_sqnr.so",'
content = content.replace(old, new_so)

# 2. Add voice map
old_v = '"default_ada": (os.path.join(MELO_MODELS_DIR, "default_zh", "G_default.pth"), 1),'
new_v = '"default_ada": (os.path.join(MELO_MODELS_DIR, "default_zh", "G_default.pth"), 1),\n    "default_ada_sqnr": (os.path.join(MELO_MODELS_DIR, "default_zh", "G_default.pth"), 1),'
content = content.replace(old_v, new_v)

# 3. Add QNN params with SQNR encoding
old_p = '"default_ada": {"in_scale": 0.0637778565287590, "in_offset": -133, "out_scale": 0.0016753388335928, "out_offset": -115},'
new_p = '"default_ada": {"in_scale": 0.0637778565287590, "in_offset": -133, "out_scale": 0.0016753388335928, "out_offset": -115},\n    "default_ada_sqnr": {"in_scale": 0.0635628774762154, "in_offset": -133, "out_scale": 0.0016934585291892, "out_offset": -117},'
content = content.replace(old_p, new_p)

with open(tts_path, 'w') as f:
    f.write(content)

print("Added default_ada_sqnr voice")
for line in content.split('\n'):
    if 'default_ada_sqnr' in line:
        print(f"  {line.strip()}")
