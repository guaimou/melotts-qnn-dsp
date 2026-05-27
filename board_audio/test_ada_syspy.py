"""Test AdaRound TTS with system Python + torchaudio mock."""
import types, sys
m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
from tts import _qnn_synthesize_to_wav

text = "你好，这是一个测试语音合成效果。"
print(f"Testing: {text} with voice=default_ada")
import time
t0 = time.time()
result = _qnn_synthesize_to_wav(text, "/tmp/test_ada.wav", voice="default_ada", speed=1.0)
elapsed = time.time() - t0
print(f"TTS done in {elapsed:.1f}s -> {result}")

# Compare with pch
print(f"\nComparing with original pch...")
t0 = time.time()
result2 = _qnn_synthesize_to_wav(text, "/tmp/test_pch.wav", voice="default", speed=1.0)
elapsed2 = time.time() - t0
print(f"TTS done in {elapsed2:.1f}s -> {result2}")

# Copy to stable location
import os
os.system("cp /tmp/test_ada.wav /home/fibo/melotts_qnn/test_adaround.wav")
os.system("cp /tmp/test_pch.wav /home/fibo/melotts_qnn/test_pch_compare.wav")
print("Files saved")

# Audio stats
from scipy.io import wavfile
sr, audio = wavfile.read("/tmp/test_ada.wav")
print(f"AdaRound: sr={sr}, len={len(audio)}, dur={len(audio)/sr:.2f}s, min={audio.min()}, max={audio.max()}")
sr2, audio2 = wavfile.read("/tmp/test_pch.wav")
print(f"PCH: sr={sr2}, len={len(audio2)}, dur={len(audio2)/sr2:.2f}s, min={audio2.min()}, max={audio2.max()}")
