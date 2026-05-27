"""Remove experimental voices, keep only default/a2/a13."""
tts_path = '/home/fibo/AI model/tts_models/tts.py'
with open(tts_path, 'r') as f:
    content = f.read()

# 1. Clean QNN_SO_MAP: remove default_ada*, default_cle
old_so = '''QNN_SO_MAP = {
    "default": "/home/fibo/melotts_qnn/lib_cle_sqnr/aarch64-ubuntu-gcc9.4/libmelotts_cle_sqnr.so",
    "a2":      "/home/fibo/melotts_qnn/lib_a2_pch/aarch64-ubuntu-gcc9.4/libmelotts_a2_pch.so",
    "a13":     "/home/fibo/melotts_qnn/lib_a13_pch/aarch64-ubuntu-gcc9.4/libmelotts_a13_pch.so",
    "default_ada": "/home/fibo/melotts_qnn/lib_adaround/aarch64-ubuntu-gcc9.4/libmelotts_adaround.so",
    "default_ada_sqnr": "/home/fibo/melotts_qnn/lib_adaround_sqnr/aarch64-ubuntu-gcc9.4/libmelotts_adaround_sqnr.so",
    "default_cle": "/home/fibo/melotts_qnn/lib_cle_sqnr/aarch64-ubuntu-gcc9.4/libmelotts_cle_sqnr.so",
}'''

new_so = '''QNN_SO_MAP = {
    "default": "/home/fibo/melotts_qnn/lib_cle_sqnr/aarch64-ubuntu-gcc9.4/libmelotts_cle_sqnr.so",
    "a2":      "/home/fibo/melotts_qnn/lib_a2_pch/aarch64-ubuntu-gcc9.4/libmelotts_a2_pch.so",
    "a13":     "/home/fibo/melotts_qnn/lib_a13_pch/aarch64-ubuntu-gcc9.4/libmelotts_a13_pch.so",
}'''
content = content.replace(old_so, new_so)

# 2. Clean QNN_VOICE_MAP
old_vm = '''QNN_VOICE_MAP = {
    "default": (os.path.join(MELO_MODELS_DIR, "default_zh", "G_default.pth"), 1),
    "a2":      (os.path.join(MELO_MODELS_DIR, "a2", "G_54600.pth"), 0),
    "a13":     (os.path.join(MELO_MODELS_DIR, "a13", "G_34400.pth"), 0),
    "default_ada": (os.path.join(MELO_MODELS_DIR, "default_zh", "G_default.pth"), 1),
    "default_ada_sqnr": (os.path.join(MELO_MODELS_DIR, "default_zh", "G_default.pth"), 1),
    "default_cle": (os.path.join(MELO_MODELS_DIR, "default_zh", "G_default.pth"), 1),
}'''

new_vm = '''QNN_VOICE_MAP = {
    "default": (os.path.join(MELO_MODELS_DIR, "default_zh", "G_default.pth"), 1),
    "a2":      (os.path.join(MELO_MODELS_DIR, "a2", "G_54600.pth"), 0),
    "a13":     (os.path.join(MELO_MODELS_DIR, "a13", "G_34400.pth"), 0),
}'''
content = content.replace(old_vm, new_vm)

# 3. Clean QNN_PARAMS
old_params = '''_QNN_PARAMS = {
    "default": {"in_scale": 0.0635628774762154, "in_offset": -133, "out_scale": 0.0016634501516819, "out_offset": -117},
    "a2":      {"in_scale": 0.0986679643392563, "in_offset": -124, "out_scale": 0.0022239189129323, "out_offset": -99},
    "a13":     {"in_scale": 0.0507732145488262, "in_offset": -133, "out_scale": 0.0021189446561038, "out_offset": -119},
    "default_ada": {"in_scale": 0.0637778565287590, "in_offset": -133, "out_scale": 0.0016753388335928, "out_offset": -115},
    "default_ada_sqnr": {"in_scale": 0.0635628774762154, "in_offset": -133, "out_scale": 0.0016934585291892, "out_offset": -117},'''

new_params = '''_QNN_PARAMS = {
    "default": {"in_scale": 0.0635628774762154, "in_offset": -133, "out_scale": 0.0016634501516819, "out_offset": -117},
    "a2":      {"in_scale": 0.0986679643392563, "in_offset": -124, "out_scale": 0.0022239189129323, "out_offset": -99},
    "a13":     {"in_scale": 0.0507732145488262, "in_offset": -133, "out_scale": 0.0021189446561038, "out_offset": -119},'''
content = content.replace(old_params, new_params)

with open(tts_path, 'w') as f:
    f.write(content)

# Verify
for section, names in [("SO_MAP", ["default", "a2", "a13", "default_ada", "default_cle"]),
                        ("VOICE_MAP", ["default", "a2", "a13", "default_ada", "default_cle"]),
                        ("PARAMS", ["default", "a2", "a13", "default_ada"])]:
    print(f"\n{section}:")
    for name in names:
        found = f'"{name}"' in content
        marker = "OK" if (name in ["default", "a2", "a13"]) == found else "BAD"
        print(f"  {name}: {'present' if found else 'removed'} [{marker}]")
