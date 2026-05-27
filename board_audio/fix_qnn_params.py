tts_path = '/home/fibo/AI model/tts_models/tts.py'
with open(tts_path, 'r') as f:
    content = f.read()

# Fix default_ada params to match gen_adaround_int8.cpp encoding
old_ada = '"default_ada": {"in_scale": 0.0635628774762154, "in_offset": -133, "out_scale": 0.0016914557199925, "out_offset": -117},'
new_ada = '"default_ada": {"in_scale": 0.0637778565287590, "in_offset": -133, "out_scale": 0.0016753388335928, "out_offset": -115},'
content = content.replace(old_ada, new_ada)

with open(tts_path, 'w') as f:
    f.write(content)

print("Updated default_ada QNN params:")
for line in content.split('\n'):
    if 'default_ada' in line and 'in_scale' in line:
        print(f"  {line.strip()}")
