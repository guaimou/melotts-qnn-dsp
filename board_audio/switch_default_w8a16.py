"""Switch default voice to W8A16."""
tts_path = '/home/fibo/AI model/tts_models/tts.py'
with open(tts_path, 'r') as f:
    content = f.read()

# Change SO_MAP default to W8A16
old = '"default": "/home/fibo/melotts_qnn/lib_cle_sqnr/aarch64-ubuntu-gcc9.4/libmelotts_cle_sqnr.so",'
new = '"default": "/home/fibo/melotts_qnn/lib_cle_w8a16/aarch64-ubuntu-gcc9.4/libmelotts_cle_w8a16.so",'
content = content.replace(old, new)

# Change PARAMS default to W8A16
old_p = '"default": {"in_scale": 0.0635628774762154, "in_offset": -133, "out_scale": 0.0016634501516819, "out_offset": -117},'
new_p = '"default": {"w8a16": True, "in_scale": 0.0002481628616806, "in_offset": -34127, "out_scale": 0.0000305175781250, "out_offset": -32768},'
content = content.replace(old_p, new_p)

with open(tts_path, 'w') as f:
    f.write(content)

print("Default voice switched to W8A16")
for line in content.split('\n'):
    if 'default' in line and ('lib_cle_w8a16' in line or '"w8a16"' in line):
        print(f"  {line.strip()}")
