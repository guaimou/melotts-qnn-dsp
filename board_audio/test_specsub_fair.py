"""Fair spectral subtraction test: same z_latent as e2e test."""
import types, sys, time, numpy as np
m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
import tts

text = "你好，这是一个测试语音合成效果。"
VOICE = "default_cle"

# Encode once
print("Encoding...")
z_saved, T_saved = tts._qnn_encode(text, voice=VOICE, speed=1.0)
print(f"z shape={z_saved.shape}")

# Monkey-patch
orig_encode = tts._qnn_encode
tts._qnn_encode = lambda *a, **kw: (z_saved, T_saved)

# Generate raw audio (NO noise gate, NO normalization)
print("Generating raw DSP output...")
tts.QNN_NOISE_GATE_ENABLED = False
audio_raw = tts._qnn_generate(z_saved, voice=VOICE)

# Normalize (same as _qnn_synthesize_to_wav does)
audio_raw = audio_raw - np.mean(audio_raw)
peak = np.max(np.abs(audio_raw))
if peak > 0:
    audio_raw_norm = audio_raw / peak * 0.95
else:
    audio_raw_norm = audio_raw

# ---- Method 1: Original (just normalized) ----
print("Method 1: Original...")
orig_samples = np.clip(audio_raw_norm * 32767, -32768, 32767).astype(np.int16)

# ---- Method 2: Spectral Subtraction ----
print("Method 2: Spectral Subtraction...")
n_fft = 2048
hop = 512
win = np.hanning(n_fft)

n_frames = (len(audio_raw_norm) - n_fft) // hop + 1
stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
for i in range(n_frames):
    frame = audio_raw_norm[i*hop:i*hop+n_fft] * win
    stft[:, i] = np.fft.rfft(frame)

magnitude = np.abs(stft)
phase = np.angle(stft)

# Noise estimate from quietest 20% frames
frame_energies = np.sum(magnitude**2, axis=0)
quiet_thresh = np.percentile(frame_energies, 20)
quiet_frames = frame_energies < quiet_thresh
noise_mag = np.median(magnitude[:, quiet_frames], axis=1) if np.sum(quiet_frames) > 0 else np.zeros(n_fft//2+1)
print(f"  Quiet frames for noise est: {np.sum(quiet_frames)}/{n_frames}")

# Oversubtraction
alpha = 2.0
beta = 0.01
clean_mag = np.maximum(magnitude - alpha * noise_mag[:, np.newaxis], beta * magnitude)

# Reconstruct
clean_stft = clean_mag * np.exp(1j * phase)
clean_frames = np.fft.irfft(clean_stft, n=n_fft, axis=0)
clean_audio = np.zeros(len(audio_raw_norm))
for i in range(n_frames):
    clean_audio[i*hop:i*hop+n_fft] += clean_frames[:, i] * win

if len(clean_audio) > len(audio_raw_norm):
    clean_audio = clean_audio[:len(audio_raw_norm)]

# Re-normalize
clean_audio = clean_audio - np.mean(clean_audio)
peak_c = np.max(np.abs(clean_audio))
if peak_c > 0:
    clean_audio = clean_audio / peak_c * 0.95
ss_samples = np.clip(clean_audio * 32767, -32768, 32767).astype(np.int16)

# ---- Method 3: Noise Gate (for comparison) ----
print("Method 3: Noise Gate...")
tts.QNN_NOISE_GATE_ENABLED = True
gated_raw = tts._apply_noise_gate(audio_raw_norm, sr=44100, strength=0.5)
gated_raw = gated_raw - np.mean(gated_raw)
peak_g = np.max(np.abs(gated_raw))
if peak_g > 0:
    gated_raw = gated_raw / peak_g * 0.95
ng_samples = np.clip(gated_raw * 32767, -32768, 32767).astype(np.int16)

# Restore
tts._qnn_encode = orig_encode
tts.QNN_NOISE_GATE_ENABLED = True

# Save all three
from scipy.io import wavfile
wavfile.write("/home/fibo/melotts_qnn/compare_original.wav", 44100, orig_samples)
wavfile.write("/home/fibo/melotts_qnn/compare_specsub.wav", 44100, ss_samples)
wavfile.write("/home/fibo/melotts_qnn/compare_gate.wav", 44100, ng_samples)

# Analysis
def analyze(a, label):
    af = a.astype(float)
    win_l = int(44100 * 0.01)
    n = len(af) // win_l
    e = [np.std(af[i*win_l:(i+1)*win_l]) for i in range(n)]
    quiet = np.mean([x for x in e if x < np.percentile(e, 10)])
    loud = np.mean([x for x in e if x > np.percentile(e, 90)])
    return {"label": label, "quiet": quiet, "loud": loud, "total": af.std()}

results = [
    analyze(orig_samples, "Original"),
    analyze(ss_samples, "Spectral Sub"),
    analyze(ng_samples, "Noise Gate"),
]

print(f"\n{'Method':<18} {'Quiet RMS':>12} {'Loud RMS':>12} {'DR':>8} {'Total':>10}")
for r in results:
    dr = 20*np.log10(r["loud"]/r["quiet"]) if r["quiet"]>0 else 0
    print(f"{r['label']:<18} {r['quiet']:>12.0f} {r['loud']:>12.0f} {dr:>7.1f}dB {r['total']:>10.0f}")

base_q = results[0]["quiet"]
for r in results[1:]:
    print(f"\n{r['label']}: {(1-r['quiet']/base_q)*100:+.1f}% vs original")

print("\nAll saved in /home/fibo/melotts_qnn/compare_*.wav")
