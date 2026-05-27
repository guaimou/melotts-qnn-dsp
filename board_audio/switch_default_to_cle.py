"""Switch default voice from PCH to CLE model."""
tts_path = '/home/fibo/AI model/tts_models/tts.py'
with open(tts_path, 'r') as f:
    content = f.read()

# 1. Change SO_MAP: default → CLE .so
old_so = '"default": "/home/fibo/melotts_qnn/lib_pch/aarch64-ubuntu-gcc9.4/libmelotts_pch.so",'
new_so = '"default": "/home/fibo/melotts_qnn/lib_cle_sqnr/aarch64-ubuntu-gcc9.4/libmelotts_cle_sqnr.so",'
content = content.replace(old_so, new_so)

# 2. Change QNN_PARAMS: default → CLE encoding (same input, slightly different output)
old_p = '"default": {"in_scale": 0.0635628774762154, "in_offset": -133, "out_scale": 0.0016914557199925, "out_offset": -117},'
new_p = '"default": {"in_scale": 0.0635628774762154, "in_offset": -133, "out_scale": 0.0016634501516819, "out_offset": -117},'
content = content.replace(old_p, new_p)

with open(tts_path, 'w') as f:
    f.write(content)

# Verify
print("Changes applied. Verifying:")
for line in content.split('\n'):
    if 'default' in line and ('lib_cle' in line or 'lib_pch' in line):
        print(f"  SO: {line.strip()}")
    if '"default":' in line and 'in_scale' in line:
        print(f"  PARAMS: {line.strip()}")
