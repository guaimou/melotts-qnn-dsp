"""Noise shaping post-processing for DSP INT8 output."""
import types, sys, time, numpy as np

m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
from tts import _qnn_synthesize_to_wav
from scipy.io import wavfile

text = "你好，这是一个测试语音合成效果。"
VOICE = "default_cle"

print("=== Noise Shaping Post-Processing ===")

# 1. Generate audio via single-pass DSP INT8
print("\n[1] Generating base audio (single-pass INT8)...")
t0 = time.time()
_qnn_synthesize_to_wav(text, "/tmp/base_audio.wav", voice=VOICE, speed=1.0)
gen_time = time.time() - t0

sr, audio_int16 = wavfile.read("/tmp/base_audio.wav")
audio = audio_int16.astype(np.float64) / 32768.0
print(f"  Generated in {gen_time:.1f}s, len={len(audio)}, sr={sr}")

# 2. Noise estimation
print("\n[2] Estimating noise profile...")

# STFT parameters
n_fft = 2048
hop = 512
win = np.hanning(n_fft)

# Compute STFT manually
n_frames = (len(audio) - n_fft) // hop + 1
stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
for i in range(n_frames):
    frame = audio[i*hop:i*hop+n_fft] * win
    stft[:, i] = np.fft.rfft(frame)

magnitude = np.abs(stft)
phase = np.angle(stft)

# Estimate noise: take median of quietest 20% frames
frame_energies = np.sum(magnitude**2, axis=0)
quiet_thresh = np.percentile(frame_energies, 20)
quiet_frames = frame_energies < quiet_thresh
noise_mag = np.median(magnitude[:, quiet_frames], axis=1) if np.sum(quiet_frames) > 0 else np.zeros(n_fft//2+1)
print(f"  Quiet frames: {np.sum(quiet_frames)}/{n_frames}")

# 3. Spectral subtraction
print("\n[3] Applying spectral subtraction...")

# Oversubtraction factor (higher = more aggressive noise removal)
alpha = 2.0
# Spectral floor (prevents musical noise)
beta = 0.01

# Subtract noise from magnitude
clean_mag = np.maximum(magnitude - alpha * noise_mag[:, np.newaxis], beta * magnitude)

# Reconstruct
t0 = time.time()
clean_stft = clean_mag * np.exp(1j * phase)
clean_frames = np.fft.irfft(clean_stft, n=n_fft, axis=0)

# Overlap-add
clean_audio = np.zeros(len(audio))
for i in range(n_frames):
    clean_audio[i*hop:i*hop+n_fft] += clean_frames[:, i] * win

# Normalize
if len(clean_audio) > len(audio):
    clean_audio = clean_audio[:len(audio)]

proc_time = time.time() - t0
print(f"  Processing time: {proc_time*1000:.1f}ms")

# 4. Soft noise gate (alternative approach)
print("\n[4] Applying soft noise gate...")
t0 = time.time()

# Compute short-time RMS
win_len = int(sr * 0.02)  # 20ms
hop_len = win_len // 4
n_wins = (len(audio) - win_len) // hop_len + 1
rms = np.array([np.sqrt(np.mean(audio[i*hop_len:i*hop_len+win_len]**2)) for i in range(n_wins)])

# Noise floor estimate
noise_rms = np.percentile(rms, 10)

# Compute per-sample gain (smooth)
gain = np.ones(len(audio))
for i in range(n_wins):
    if rms[i] > 0:
        g = np.clip(1.0 - noise_rms / rms[i], 0.05, 1.0)  # min gain 5%
        start = i * hop_len
        end = min(start + win_len, len(audio))
        # Smooth gain: raise to power (1-exp(-SNR))
        snr = max(0, rms[i] / max(noise_rms, 1e-10) - 1)
        g_smooth = g ** 0.5  # soft knee
        gain[start:end] = np.minimum(gain[start:end], g_smooth + (1-g_smooth) * (1-np.exp(-snr)))

gated_audio = audio * gain
gate_time = time.time() - t0
print(f"  Gate time: {gate_time*1000:.1f}ms, noise_rms={noise_rms:.6f}")

# 5. Save all versions
print("\n[5] Saving results...")

def save_wav(path, sig):
    sig = sig - np.mean(sig)
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak * 0.95
    wavfile.write(path, sr, np.clip(sig * 32767, -32768, 32767).astype(np.int16))

save_wav("/home/fibo/melotts_qnn/test_spectral_sub.wav", clean_audio)
save_wav("/home/fibo/melotts_qnn/test_noise_gate.wav", gated_audio)
save_wav("/home/fibo/melotts_qnn/test_base_norm.wav", audio)  # re-normalize same way

# 6. Analysis
print("\n[6] Analysis...")

def analyze(a, label):
    """Compute noise metrics."""
    win_l = int(sr * 0.01)
    n_w = len(a) // win_l
    e = np.array([np.std(a[i*win_l:(i+1)*win_l]) for i in range(n_w)])
    q = np.mean([x for x in e if x < np.percentile(e, 10)])
    l = np.mean([x for x in e if x > np.percentile(e, 90)])
    total_rms = np.std(a)
    return {"label": label, "quiet_rms": q, "loud_rms": l, "total_rms": total_rms,
            "dr": 20*np.log10(l/q) if q>0 else 0}

results = [
    analyze(audio, "Original INT8"),
    analyze(clean_audio, "Spectral Sub"),
    analyze(gated_audio, "Noise Gate"),
]

print(f"{'Method':<18} {'Quiet RMS':>10} {'Loud RMS':>10} {'Dyn Range':>10}")
for r in results:
    print(f"{r['label']:<18} {r['quiet_rms']:>10.6f} {r['loud_rms']:>10.4f} {r['dr']:>9.1f}dB")

# Compare improvement
base_q = results[0]["quiet_rms"]
for r in results[1:]:
    imp = (1 - r["quiet_rms"] / base_q) * 100
    print(f"\n{r['label']}: quiet RMS {imp:+.1f}% vs original")

import os
for f in ["/home/fibo/melotts_qnn/test_spectral_sub.wav",
          "/home/fibo/melotts_qnn/test_noise_gate.wav",
          "/home/fibo/melotts_qnn/test_base_norm.wav"]:
    print(f"  {os.path.basename(f)}: {os.path.getsize(f)} bytes")
