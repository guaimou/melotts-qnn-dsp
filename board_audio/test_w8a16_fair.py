"""Fair W8A16 vs INT8 comparison: same z_latent."""
import types, sys, time, numpy as np
m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
import tts
from scipy.io import wavfile
import os

text = "你好，这是一个测试语音合成效果。"

# Encode once
print("Encoding...")
z, T_enc = tts._qnn_encode(text, voice="default", speed=1.0)
print(f"z shape={z.shape}, T={T_enc}")

# Generate with W8A16 (raw, no post-processing)
print("\n=== W8A16 raw DSP ===")
tts.QNN_SPECSUB_ENABLED = False
audio_w8a16 = tts._qnn_generate(z, voice="default_w8a16")

# Generate with INT8 CLE
print("=== INT8 CLE raw DSP ===")
audio_int8 = tts._qnn_generate(z, voice="default")

tts.QNN_SPECSUB_ENABLED = True

# Analyze raw (no post-processing)
def analyze(a, label):
    af = a
    win = int(44100 * 0.01)
    n = len(af) // win
    e = [np.std(af[i*win:(i+1)*win]) for i in range(n)]
    quiet = np.mean([x for x in e if x < np.percentile(e, 10)])
    loud = np.mean([x for x in e if x > np.percentile(e, 90)])
    return {"label": label, "quiet": quiet, "loud": loud, "total": np.std(af)}

r1 = analyze(audio_w8a16, "W8A16 raw")
r2 = analyze(audio_int8, "INT8 raw")

print(f"\n{'Method':<15} {'Quiet RMS':>12} {'Loud RMS':>12} {'DR':>8} {'Total':>10}")
for r in [r1, r2]:
    dr = 20*np.log10(r["loud"]/r["quiet"]) if r["quiet"]>0 else 0
    print(f"{r['label']:<15} {r['quiet']:>12.6f} {r['loud']:>12.4f} {dr:>7.1f}dB {r['total']:>10.4f}")

imp = (1 - r1["quiet"] / r2["quiet"]) * 100
print(f"\nW8A16 quiet RMS improvement: {imp:+.1f}%")

# Now apply specsub + norm to both and save for listening
for label, audio, path in [
    ("W8A16", audio_w8a16, "/home/fibo/melotts_qnn/fair_w8a16.wav"),
    ("INT8", audio_int8, "/home/fibo/melotts_qnn/fair_int8.wav"),
]:
    a = tts._apply_spectral_subtraction(audio, sr=44100, alpha=2.0, beta=0.01)
    a = a - np.mean(a)
    peak = np.max(np.abs(a))
    if peak > 0:
        a = a / peak * 0.95
    samples = np.clip(a * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(path, 44100, samples)

print("Saved: fair_w8a16.wav, fair_int8.wav")
