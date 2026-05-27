"""Compare W8A16 DSP vs ONNX FP32 reference (same z_latent)."""
import types, sys, time, numpy as np
m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
import tts
import onnxruntime as ort
from scipy.io import wavfile

text = "你好，这是一个测试语音合成效果。"
VOICE = "default"

# 1. Encode once
print("=== Encoding ===")
z, T_enc = tts._qnn_encode(text, voice=VOICE, speed=1.0)
print(f"z shape={z.shape}, T={T_enc}")

# 2. ONNX FP32 reference (ground truth)
print("\n=== ONNX FP32 Reference ===")
QNN_CHUNK_T = 128
HOP_LENGTH = 512
overlap_t = 16
hop_t = QNN_CHUNK_T - overlap_t

onnx_path = "/home/fibo/AI model/tts_models/before/melo_minimal/melo_minimal_bundle/models/default_zh/generator_default.onnx"
if not __import__('os').path.exists(onnx_path):
    # Try alternative paths
    onnx_path = "/home/fibo/melotts_qnn/models/generator_default.onnx"

print(f"ONNX model: {onnx_path}")
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

t0 = time.time()
audio_parts = []
fade_in = np.linspace(0, 1, overlap_t * HOP_LENGTH)
fade_out = np.linspace(1, 0, overlap_t * HOP_LENGTH)
pos = 0
while pos < T_enc:
    end = min(pos + QNN_CHUNK_T, T_enc)
    chunk_len = end - pos
    z_chunk = z[0:1, :, pos:end]
    if chunk_len < QNN_CHUNK_T:
        z_chunk = np.pad(z_chunk, ((0,0),(0,0),(0,QNN_CHUNK_T-chunk_len)), mode='constant')

    y = sess.run(None, {'z': z_chunk.astype(np.float32)})[0]
    y_f = y.reshape(1, 1, -1)[0, 0]
    valid = y_f[:chunk_len * HOP_LENGTH]

    if pos > 0 and len(valid) >= len(fade_in) and len(audio_parts[-1]) >= len(fade_out):
        valid[:len(fade_in)] *= fade_in
        audio_parts[-1][-len(fade_out):] *= fade_out
        valid[:len(fade_in)] += audio_parts[-1][-len(fade_out):]

    audio_parts.append(valid)
    pos += hop_t

audio_onnx = np.concatenate(audio_parts)
t_onnx = time.time() - t0
print(f"ONNX FP32: {t_onnx:.1f}s, len={len(audio_onnx)}")

# 3. W8A16 DSP
print("\n=== W8A16 DSP ===")
tts.QNN_SPECSUB_ENABLED = False
t0 = time.time()
audio_dsp = tts._qnn_generate(z, voice=VOICE)
t_dsp = time.time() - t0
print(f"W8A16 DSP: {t_dsp:.1f}s, len={len(audio_dsp)}")

# 4. Compare raw audio (no post-processing)
min_len = min(len(audio_onnx), len(audio_dsp))
a_onnx = audio_onnx[:min_len]
a_dsp = audio_dsp[:min_len]

# Correlation
corr = np.corrcoef(a_onnx, a_dsp)[0, 1]

# SNR = 10*log10(signal_power / noise_power)
noise = a_onnx - a_dsp
signal_power = np.var(a_onnx)
noise_power = np.var(noise)
snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

# MSE
mse = np.mean((a_onnx - a_dsp) ** 2)

# Per-segment SNR (quiet vs loud)
win = int(44100 * 0.01)
n = min_len // win
e_onnx = np.array([np.std(a_onnx[i*win:(i+1)*win]) for i in range(n)])
e_dsp = np.array([np.std(a_dsp[i*win:(i+1)*win]) for i in range(n)])
quiet_thresh = np.percentile(e_onnx, 20)
loud_thresh = np.percentile(e_onnx, 80)
quiet_mask = e_onnx < quiet_thresh
loud_mask = e_onnx > loud_thresh

if np.sum(quiet_mask) > 0:
    quiet_snr = 10 * np.log10(np.mean(e_onnx[quiet_mask]**2) / np.mean((e_onnx[quiet_mask] - e_dsp[quiet_mask])**2))
else:
    quiet_snr = float('inf')

if np.sum(loud_mask) > 0:
    loud_snr = 10 * np.log10(np.mean(e_onnx[loud_mask]**2) / np.mean((e_onnx[loud_mask] - e_dsp[loud_mask])**2))
else:
    loud_snr = float('inf')

print(f"\n=== Comparison (same z_latent) ===")
print(f"Correlation: {corr:.6f}")
print(f"SNR (overall): {snr:.1f} dB")
print(f"SNR (quiet seg): {quiet_snr:.1f} dB")
print(f"SNR (loud seg): {loud_snr:.1f} dB")
print(f"MSE: {mse:.8f}")
print(f"Max abs diff: {np.max(np.abs(a_onnx - a_dsp)):.6f}")

# 5. Save both for listening
def save_wav(audio, path):
    a = audio - np.mean(audio)
    peak = np.max(np.abs(a))
    if peak > 0:
        a = a / peak * 0.95
    samples = np.clip(a * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(path, 44100, samples)

save_wav(audio_onnx, "/home/fibo/melotts_qnn/ref_onnx_fp32.wav")
save_wav(audio_dsp, "/home/fibo/melotts_qnn/test_w8a16_v2.wav")
print("\nSaved: ref_onnx_fp32.wav, test_w8a16_v2.wav")

# Also save raw (no peak norm) for fair comparison
wavfile.write("/home/fibo/melotts_qnn/ref_onnx_raw.wav", 44100,
              np.clip((audio_onnx - np.mean(audio_onnx)) * 32767, -32768, 32767).astype(np.int16))
wavfile.write("/home/fibo/melotts_qnn/test_w8a16_raw.wav", 44100,
              np.clip((audio_dsp - np.mean(audio_dsp)) * 32767, -32768, 32767).astype(np.int16))
print("Saved raw versions too")
