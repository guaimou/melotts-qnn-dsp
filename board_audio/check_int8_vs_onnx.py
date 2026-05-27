"""Compare INT8 DSP vs ONNX FP32 (same z_latent) — verify INT8 pipeline works."""
import types, sys, time, numpy as np
m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
import tts
import onnxruntime as ort
from scipy.io import wavfile

# Get a random z (same for both tests)
z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5

# 1. ONNX FP32
print("=== ONNX FP32 ===")
onnx_path = "/home/fibo/melotts_qnn/models/generator_default.onnx"
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
y_onnx = sess.run(None, {'z': z})[0].squeeze()
print(f"std={y_onnx.std():.6f}, len={len(y_onnx)}")

# 2. W8A16 DSP
print("\n=== W8A16 DSP ===")
tts.QNN_SPECSUB_ENABLED = False
y_w8a16 = tts._qnn_generate(z, voice="default")
print(f"std={y_w8a16.std():.6f}, len={len(y_w8a16)}")

# 3. INT8 DSP — need a voice that still uses INT8
# default is now W8A16. Check if default_w8a16 exists
# Actually we need to add a temporary INT8 voice back
# Or just use a2 which is INT8 but different encoder
# Simplest: manually call the INT8 path

# Get the INT8 params manually
old_int8_params = {"in_scale": 0.0635628774762154, "in_offset": -133, "out_scale": 0.0016634501516819, "out_offset": -117}

# Use the W8A16 API but with INT8 encoding — this will test if the DSP model itself is broken
# Actually, the .so model IS W8A16. We can't test INT8 on the same .so.
# Let me check if there's an INT8 .so still available

print("\n=== Comparison ===")
min_len = min(len(y_onnx), len(y_w8a16))
a_onnx = y_onnx[:min_len]
a_w8a16 = y_w8a16[:min_len]

corr = np.corrcoef(a_onnx, a_w8a16)[0, 1]
diff = np.abs(a_onnx - a_w8a16)
signal_power = np.var(a_onnx)
noise_power = np.var(a_onnx - a_w8a16)
snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

print(f"Correlation: {corr:.6f}")
print(f"SNR: {snr:.1f} dB")
print(f"Max diff: {diff.max():.6f}")
print(f"ONNX first 10: {np.round(a_onnx[:10], 6)}")
print(f"W8A16 first 10: {np.round(a_w8a16[:10], 6)}")
