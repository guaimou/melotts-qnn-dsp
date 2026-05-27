"""Check raw DSP output for W8A16 and INT8."""
import types, sys, time, numpy as np
m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
import tts

text = "你好。"
VOICE = "default"

# Encode
z, T = tts._qnn_encode(text, voice=VOICE, speed=1.0)
print(f"z shape={z.shape}, T={T}")

# Get raw DSP output for INT8 (default was switched to W8A16, but we still have INT8 code path)
# Actually default is now W8A16. Let's test both explicitly.
tts.QNN_SPECSUB_ENABLED = False

# W8A16
print("\n=== W8A16 raw DSP (default voice) ===")
audio_w8 = tts._qnn_generate(z, voice="default")
print(f"len={len(audio_w8)}")
print(f"first 50: {np.round(audio_w8[:50], 6)}")
print(f"last 50: {np.round(audio_w8[-50:], 6)}")
print(f"stats: min={audio_w8.min():.6f}, max={audio_w8.max():.6f}, std={audio_w8.std():.6f}")
# Check zero-crossings
signs = np.sign(audio_w8)
zero_x = np.sum(np.diff(signs) != 0)
print(f"zero-crossings: {zero_x} ({zero_x/len(audio_w8)*100:.1f}%)")

# Generate with one random z to test if it's the DSP or the encoder
print("\n=== W8A16 with random z ===")
z_rand = np.random.randn(1, 192, 128).astype(np.float32) * 0.5
audio_rand = tts._qnn_generate(z_rand, voice="default")
print(f"len={len(audio_rand)}")
print(f"first 50: {np.round(audio_rand[:50], 6)}")
print(f"stats: min={audio_rand.min():.6f}, max={audio_rand.max():.6f}, std={audio_rand.std():.6f}")

# Check output uint16 values directly (without dequant)
print("\n=== W8A16 raw uint16 output check ===")
api = tts._qnn_get_api("default")
ret = api.Execute_float({"z": z_rand.flatten().tolist()})
y_u8 = np.array(api.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
y_u16 = y_u8.view(np.uint16)
print(f"uint16 output: len={len(y_u16)}, min={y_u16.min()}, max={y_u16.max()}")
print(f"first 20 uint16: {y_u16[:20]}")
print(f"mean uint16: {y_u16.astype(float).mean():.1f}")
# Check if all values are the same
unique = len(np.unique(y_u16))
print(f"unique uint16 values: {unique}/{len(y_u16)}")
