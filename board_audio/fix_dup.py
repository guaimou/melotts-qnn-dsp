"""Remove duplicate default_ada entries."""
tts_path = '/home/fibo/AI model/tts_models/tts.py'
with open(tts_path, 'r') as f:
    content = f.read()

# Remove duplicate default_ada lines, keep only first occurrence in each section
lines = content.split('\n')
seen_ada_so = False
seen_ada_voice = False
result = []
for line in lines:
    if 'default_ada' in line and '/libmelotts_adaround.so' in line:
        if not seen_ada_so:
            seen_ada_so = True
            result.append(line)
        else:
            continue  # skip duplicate
    elif 'default_ada' in line and 'MELO_MODELS_DIR' in line:
        if not seen_ada_voice:
            seen_ada_voice = True
            result.append(line)
        else:
            continue  # skip duplicate
    else:
        result.append(line)

with open(tts_path, 'w') as f:
    f.write('\n'.join(result))

print("Dedup done. Verify:")
for line in result:
    if 'default_ada' in line:
        print(f"  {line.strip()}")
