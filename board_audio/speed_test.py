"""Warm-cache speed comparison: AdaRound vs PCH."""
import types, sys, time
m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
from tts import _qnn_synthesize_to_wav

text = "你好，这是一个测试语音合成效果。"

# Warm up both voices
print("Warming up...")
_qnn_synthesize_to_wav(text, "/tmp/warm_ada.wav", voice="default_ada", speed=1.0)
_qnn_synthesize_to_wav(text, "/tmp/warm_pch.wav", voice="default", speed=1.0)
print("Warm up done\n")

# Test AdaRound (3 runs)
print("=== AdaRound (warm) ===")
for i in range(3):
    t0 = time.time()
    _qnn_synthesize_to_wav(text, f"/tmp/speed_ada_{i}.wav", voice="default_ada", speed=1.0)
    elapsed = time.time() - t0
    print(f"  Run {i+1}: {elapsed:.2f}s")

# Test PCH (3 runs)
print("\n=== PCH (warm) ===")
for i in range(3):
    t0 = time.time()
    _qnn_synthesize_to_wav(text, f"/tmp/speed_pch_{i}.wav", voice="default", speed=1.0)
    elapsed = time.time() - t0
    print(f"  Run {i+1}: {elapsed:.2f}s")
