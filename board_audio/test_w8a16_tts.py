"""Test W8A16 TTS end-to-end with quality comparison."""
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

# Test W8A16
print("=== W8A16 TTS ===")
t0 = time.time()
tts._qnn_synthesize_to_wav(text, "/tmp/test_w8a16.wav", voice="default_w8a16", speed=1.0)
t_w8a16 = time.time() - t0
print(f"Done in {t_w8a16:.1f}s")

# Test INT8 CLE for comparison
print("\n=== INT8 CLE TTS ===")
t0 = time.time()
tts._qnn_synthesize_to_wav(text, "/tmp/test_int8.wav", voice="default", speed=1.0)
t_int8 = time.time() - t0
print(f"Done in {t_int8:.1f}s")

# Analyze
def analyze(path, label):
    sr, a = wavfile.read(path)
    af = a.astype(float)
    win = int(sr * 0.01)
    n = len(af) // win
    e = [np.std(af[i*win:(i+1)*win]) for i in range(n)]
    quiet = np.mean([x for x in e if x < np.percentile(e, 10)])
    loud = np.mean([x for x in e if x > np.percentile(e, 90)])
    return {"label": label, "quiet": quiet, "loud": loud, "total": af.std(), "len": len(af)}

r1 = analyze("/tmp/test_w8a16.wav", "W8A16")
r2 = analyze("/tmp/test_int8.wav", "INT8 CLE")

print(f"\n{'Method':<15} {'Time':>8} {'Quiet':>8} {'Loud':>8} {'DR':>7} {'Dur':>7}")
for r in [r1, r2]:
    dr = 20*np.log10(r["loud"]/r["quiet"]) if r["quiet"]>0 else 0
    dur = r["len"]/44100
    t = t_w8a16 if "W8A16" in r["label"] else t_int8
    print(f"{r['label']:<15} {t:>7.1f}s {r['quiet']:>8.0f} {r['loud']:>8.0f} {dr:>6.1f}dB {dur:>6.2f}s")

os.system("cp /tmp/test_w8a16.wav /home/fibo/melotts_qnn/test_w8a16.wav")
os.system("cp /tmp/test_int8.wav /home/fibo/melotts_qnn/test_int8.wav")
print("\nSaved: test_w8a16.wav, test_int8.wav")
