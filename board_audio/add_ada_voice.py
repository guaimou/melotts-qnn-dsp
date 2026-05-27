import re
with open('/home/fibo/AI model/tts_models/tts.py', 'r') as f:
    content = f.read()

old = '"a13":     "/home/fibo/melotts_qnn/lib_a13_pch/aarch64-ubuntu-gcc9.4/libmelotts_a13_pch.so",'
new = '"a13":     "/home/fibo/melotts_qnn/lib_a13_pch/aarch64-ubuntu-gcc9.4/libmelotts_a13_pch.so",\n    "default_ada": "/home/fibo/melotts_qnn/lib_adaround/aarch64-ubuntu-gcc9.4/libmelotts_adaround.so",'
content = content.replace(old, new)

with open('/home/fibo/AI model/tts_models/tts.py', 'w') as f:
    f.write(content)
print("QNN_SO_MAP updated: default_ada added")
