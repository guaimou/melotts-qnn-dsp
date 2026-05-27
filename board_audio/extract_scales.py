"""Extract quantization scales from QNN .cpp file."""
import re

with open('/home/fibo/melotts_qnn/gen_adaround_int8.cpp', 'r') as f:
    content = f.read()

# Find all scale/offset pairs
# QNN encoding format: scale=X.XXXX, offset=N
scales = re.findall(r'scale[= ]+([0-9.e+\-]+)', content)
offsets = re.findall(r'offset[= ]+(-?\d+)', content)

print(f"Found {len(scales)} scales, {len(offsets)} offsets")

# Find input tensor encoding (first scale in the graph)
# Look for the first tensor definition which should be the input
patterns = [
    (r'input.*?scale[= ]+([0-9.e+\-]+).*?offset[= ]+(-?\d+)', 'input'),
    (r'output.*?scale[= ]+([0-9.e+\-]+).*?offset[= ]+(-?\d+)', 'output'),
]

for pattern, label in patterns:
    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
    if matches:
        print(f"\n{label}: scale={matches[0][0]}, offset={matches[0][1]}")

# Search for QNN tensor encoding patterns
enc_matches = re.findall(r'Qnn_Tensor_t.*?scale.*?([0-9.e+\-]+).*?offset.*?(-?\d+)', content, re.DOTALL)
if enc_matches:
    print(f"\nFirst 5 QNN tensor encodings:")
    for i, (s, o) in enumerate(enc_matches[:5]):
        print(f"  Tensor {i}: scale={s}, offset={o}")

# Also look for _qnn_scale or encoding patterns
encoding_patterns = re.findall(r'encoding.*?scale[= ]+([0-9.e+\-]+).*?offset[= ]+(-?\d+)', content, re.DOTALL | re.IGNORECASE)
if encoding_patterns:
    print(f"\nEncoding matches:")
    for s, o in encoding_patterns[:5]:
        print(f"  scale={s}, offset={o}")

# Dump the first few scales directly
print(f"\nFirst 10 scales found: {scales[:10]}")
print(f"First 10 offsets found: {offsets[:10]}")
