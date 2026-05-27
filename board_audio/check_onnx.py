import numpy as np
import onnxruntime as ort
import os

z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5

# Check all ONNX models
paths = [
    "/home/fibo/melotts_qnn/models/generator_default.onnx",
    "/home/fibo/AI model/tts_models/before/melo_minimal/melo_minimal_bundle/models/default_zh/generator_default.onnx",
]
for path in paths:
    if os.path.exists(path):
        s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        y = s.run(None, {'z': z})[0].squeeze()
        print(f"OK: {os.path.basename(os.path.dirname(path))}/generator_default.onnx")
        print(f"    std={y.std():.6f}, len={len(y)}, first5={y[:5]}")
    else:
        print(f"MISSING: {path}")

# Also compare if both exist
if os.path.exists(paths[0]) and os.path.exists(paths[1]):
    s1 = ort.InferenceSession(paths[0], providers=['CPUExecutionProvider'])
    s2 = ort.InferenceSession(paths[1], providers=['CPUExecutionProvider'])
    y1 = s1.run(None, {'z': z})[0].squeeze()
    y2 = s2.run(None, {'z': z})[0].squeeze()
    diff = np.abs(y1 - y2).max()
    corr = np.corrcoef(y1, y2)[0, 1]
    print(f"\nComparison: diff_max={diff:.10f}, corr={corr:.10f}")
