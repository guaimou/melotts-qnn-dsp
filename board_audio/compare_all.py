"""Compare CLE ONNX vs default ONNX vs W8A16 DSP (same random z)."""
import sys, numpy as np, onnxruntime as ort

z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5

# 1. Default ONNX
print("=== Default ONNX ===")
s_def = ort.InferenceSession("/home/fibo/melotts_qnn/models/generator_default.onnx", providers=['CPUExecutionProvider'])
y_def = s_def.run(None, {'z': z})[0].squeeze()
print(f"std={y_def.std():.6f}, first5={y_def[:5]}")

# 2. CLE ONNX
print("\n=== CLE ONNX ===")
s_cle = ort.InferenceSession("/home/fibo/melotts_qnn/generator_cle.onnx", providers=['CPUExecutionProvider'])
y_cle = s_cle.run(None, {'z': z})[0].squeeze()
print(f"std={y_cle.std():.6f}, first5={y_cle[:5]}")

# 3. Compare
for label, a, b in [("Default vs CLE", y_def, y_cle)]:
    diff = np.abs(a - b)
    corr = np.corrcoef(a, b)[0, 1]
    signal_power = np.var(a)
    noise_power = np.var(a - b)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    print(f"\n{label}:")
    print(f"  corr={corr:.6f}, SNR={snr:.1f}dB, max_diff={diff.max():.10f}")

# 4. W8A16 DSP (use cached API, will be fast if already loaded)
print("\n=== W8A16 DSP ===")
import types
m = types.ModuleType("torchaudio")
m.load = lambda *a,**kw: (None, None)
m.info = lambda *a,**kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
import tts
y_dsp = tts._qnn_generate(z, voice="default")
print(f"std={y_dsp.std():.6f}, len={len(y_dsp)}, first5={y_dsp[:5]}")

# 5. Compare DSP vs CLE ONNX
print(f"\nW8A16 DSP vs CLE ONNX:")
min_len = min(len(y_cle), len(y_dsp))
a = y_cle[:min_len]
b = y_dsp[:min_len]
diff = np.abs(a - b)
corr = np.corrcoef(a, b)[0, 1]
signal_power = np.var(a)
noise_power = np.var(a - b)
snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
print(f"  corr={corr:.6f}, SNR={snr:.1f}dB, max_diff={diff.max():.6f}")
print(f"  CLE first10: {np.round(a[:10], 6)}")
print(f"  DSP first10: {np.round(b[:10], 6)}")
