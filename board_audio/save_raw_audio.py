"""Save raw DSP output without any post-processing."""
import types, sys, time, numpy as np
m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
import tts
from scipy.io import wavfile

text = "你好，这是一个测试语音合成效果。"
VOICE = "default"

# Disable specsub
tts.QNN_SPECSUB_ENABLED = False

# Full pipeline with NO post-processing
print("Encoding...")
z, T = tts._qnn_encode(text, voice=VOICE, speed=1.0)
print(f"z shape={z.shape}, T={T}")

print("Generating raw W8A16...")
audio_raw = tts._qnn_generate(z, voice=VOICE)
print(f"raw: len={len(audio_raw)}, min={audio_raw.min():.6f}, max={audio_raw.max():.6f}")

# Version 1: DC removal + peak norm only (no specsub)
audio_v1 = audio_raw - np.mean(audio_raw)
peak = np.max(np.abs(audio_v1))
if peak > 0:
    audio_v1 = audio_v1 / peak * 0.95
samples_v1 = np.clip(audio_v1 * 32767, -32768, 32767).astype(np.int16)
wavfile.write("/home/fibo/melotts_qnn/w8a16_no_specsub.wav", 44100, samples_v1)
print("Saved: w8a16_no_specsub.wav")

# Version 2: DC removal only, no peak norm (preserve dynamics)
audio_v2 = audio_raw - np.mean(audio_raw)
samples_v2 = np.clip(audio_v2 * 32767, -32768, 32767).astype(np.int16)
wavfile.write("/home/fibo/melotts_qnn/w8a16_no_norm.wav", 44100, samples_v2)
print("Saved: w8a16_no_norm.wav")

# Version 3: Full pipeline with MILDER specsub (alpha=1.0)
tts.QNN_SPECSUB_ALPHA = 1.0
audio_v3 = tts._apply_spectral_subtraction(audio_raw, sr=44100, alpha=1.0, beta=0.01)
audio_v3 = audio_v3 - np.mean(audio_v3)
peak = np.max(np.abs(audio_v3))
if peak > 0:
    audio_v3 = audio_v3 / peak * 0.95
samples_v3 = np.clip(audio_v3 * 32767, -32768, 32767).astype(np.int16)
wavfile.write("/home/fibo/melotts_qnn/w8a16_mild_specsub.wav", 44100, samples_v3)
print("Saved: w8a16_mild_specsub.wav")

# Also save INT8 for comparison
print("\nGenerating INT8 with CLE...")
audio_i8 = tts._qnn_generate(z, voice="default_w8a16")
# Actually default is now W8A16. Let's use the INT8 path manually.
# Wait - default IS W8A16 now. Let me generate INT8 separately.
# The INT8 code path still exists if we call with the old params
# Actually both default and default_w8a16 point to W8A16 now.
# We need a separate voice for INT8.

# Use a2 voice which is INT8 (same encoder architecture, different weights)
print("Generating INT8 via a2 voice...")
audio_i8 = tts._qnn_generate(z, voice="a2")
audio_i8 = audio_i8 - np.mean(audio_i8)
peak = np.max(np.abs(audio_i8))
if peak > 0:
    audio_i8 = audio_i8 / peak * 0.95
samples_i8 = np.clip(audio_i8 * 32767, -32768, 32767).astype(np.int16)
wavfile.write("/home/fibo/melotts_qnn/int8_a2.wav", 44100, samples_i8)
print("Saved: int8_a2.wav")

tts.QNN_SPECSUB_ALPHA = 2.0
tts.QNN_SPECSUB_ENABLED = True
