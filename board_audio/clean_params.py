c = open('/home/fibo/AI model/tts_models/tts.py').read()
old = '    "default_cle": {"in_scale": 0.0635628774762154, "in_offset": -133, "out_scale": 0.0016634501516819, "out_offset": -117},\n'
c = c.replace(old, '')
open('/home/fibo/AI model/tts_models/tts.py', 'w').write(c)
for line in c.split('\n'):
    if 'default_cle' in line:
        print(f"STILL PRESENT: {line.strip()}")
if 'default_cle' not in c:
    print("default_cle fully removed")
