tts_path = '/home/fibo/AI model/tts_models/tts.py'
with open(tts_path, 'r') as f:
    content = f.read()

old = '"a13":     {"in_scale": 0.0507732145488262, "in_offset": -133, "out_scale": 0.0021189446561038, "out_offset": -119},'
new = ('"a13":     {"in_scale": 0.0507732145488262, "in_offset": -133, "out_scale": 0.0021189446561038, "out_offset": -119},\n'
       '    "default_ada": {"in_scale": 0.0635628774762154, "in_offset": -133, "out_scale": 0.0016914557199925, "out_offset": -117},')
content = content.replace(old, new)

with open(tts_path, 'w') as f:
    f.write(content)

# Verify
for line in content.split('\n'):
    if 'default_ada' in line:
        print(line.strip())
