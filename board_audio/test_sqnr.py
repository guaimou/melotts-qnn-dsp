"""Quick test: AdaRound SQNR vs PCH."""
import types, sys, time, numpy as np

m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
from tts import _qnn_synthesize_to_wav
from scipy.io import wavfile

text = "你好，这是一个测试语音合成效果。"

# Test SQNR
print("=== AdaRound SQNR ===")
t0 = time.time()
_qnn_synthesize_to_wav(text, "/tmp/test_sqnr.wav", voice="default_ada_sqnr", speed=1.0)
print(f"Done in {time.time()-t0:.1f}s")

# Test PCH
print("\n=== PCH (baseline) ===")
t0 = time.time()
_qnn_synthesize_to_wav(text, "/tmp/test_pch2.wav", voice="default", speed=1.0)
print(f"Done in {time.time()-t0:.1f}s")

# Compare
import os
os.system("cp /tmp/test_sqnr.wav /home/fibo/melotts_qnn/test_sqnr.wav")
os.system("cp /tmp/test_pch2.wav /home/fibo/melotts_qnn/test_pch2.wav")

sr1, sqnr = wavfile.read("/tmp/test_sqnr.wav")
sr2, pch = wavfile.read("/tmp/test_pch2.wav")
print(f"\nSQNR: len={len(sqnr)}, dur={len(sqnr)/sr1:.2f}s, min={sqnr.min()}, max={sqnr.max()}")
print(f"PCH:  len={len(pch)}, dur={len(pch)/sr2:.2f}s, min={pch.min()}, max={pch.max()}")

# Noise floor
def noise_floor(audio, sr):
    win = int(sr * 0.01)
    n = len(audio) // win
    e = [np.std(audio[i*win:(i+1)*win]) for i in range(n)]
    q = np.percentile(e, 10)
    quiet_e = np.mean([x for x in e if x < q])
    return quiet_e

print(f"\nNoise floor (quiet 10%):")
print(f"  SQNR: {noise_floor(sqnr.astype(float), sr1):.1f}")
print(f"  PCH:  {noise_floor(pch.astype(float), sr2):.1f}")
