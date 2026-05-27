"""Debug z_latent and audio output comparison."""
import types, sys, time, numpy as np

m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")

# Import TTS internals directly
from tts import _qnn_encode, _qnn_generate, _qnn_get_api, _QNN_PARAMS

text = "你好，这是一个测试语音合成效果。"

# Test 1: z_latent for default_ada
print("=== Encoding with default_ada ===")
t0 = time.time()
z_ada, T_ada = _qnn_encode(text, voice="default_ada", speed=1.0)
t_enc = time.time() - t0
print(f"Encoder: {t_enc:.2f}s, z shape={z_ada.shape}, T={T_ada}")
print(f"z stats: mean={z_ada.mean():.6f}, std={z_ada.std():.6f}, min={z_ada.min():.6f}, max={z_ada.max():.6f}")

# Test 2: z_latent for default (pch)
print("\n=== Encoding with default (pch) ===")
t0 = time.time()
z_pch, T_pch = _qnn_encode(text, voice="default", speed=1.0)
t_enc2 = time.time() - t0
print(f"Encoder: {t_enc2:.2f}s, z shape={z_pch.shape}, T={T_pch}")
print(f"z stats: mean={z_pch.mean():.6f}, std={z_pch.std():.6f}, min={z_pch.min():.6f}, max={z_pch.max():.6f}")

# Compare z_latent
print(f"\n=== z_latent comparison ===")
if z_ada.shape == z_pch.shape:
    diff = np.abs(z_ada - z_pch)
    corr = np.corrcoef(z_ada.flatten(), z_pch.flatten())[0, 1]
    print(f"Same shape. diff_max={diff.max():.10f}, corr={corr:.10f}")
else:
    print(f"DIFFERENT shapes! ada={z_ada.shape}, pch={z_pch.shape}")

# Test 3: Audio generation for both
print("\n=== Generating audio ===")
t0 = time.time()
audio_ada = _qnn_generate(z_ada, voice="default_ada")
t_gen_ada = time.time() - t0
print(f"AdaRound DSP: {t_gen_ada:.2f}s, audio shape={audio_ada.shape}")
print(f"  audio stats: min={audio_ada.min():.6f}, max={audio_ada.max():.6f}, std={audio_ada.std():.6f}")

t0 = time.time()
audio_pch = _qnn_generate(z_pch, voice="default")
t_gen_pch = time.time() - t0
print(f"PCH DSP: {t_gen_pch:.2f}s, audio shape={audio_pch.shape}")
print(f"  audio stats: min={audio_pch.min():.6f}, max={audio_pch.max():.6f}, std={audio_pch.std():.6f}")

# Save for listening
from scipy.io import wavfile
wavfile.write("/home/fibo/melotts_qnn/debug_ada.wav", 44100, (audio_ada * 32767).astype(np.int16))
wavfile.write("/home/fibo/melotts_qnn/debug_pch.wav", 44100, (audio_pch * 32767).astype(np.int16))
print("\nSaved debug_ada.wav and debug_pch.wav")
