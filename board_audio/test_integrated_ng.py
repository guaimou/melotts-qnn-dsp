"""Test integrated noise gate in TTS module."""
import types, sys, time, numpy as np
m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
import tts

text = "你好，这是一个测试语音合成效果。"

print("=== Integrated Noise Gate Test ===")
print(f"Gate enabled: {tts.QNN_NOISE_GATE_ENABLED}")
print(f"Gate strength: {tts.QNN_NOISE_GATE_STRENGTH}")

# Test 1: With gate (default)
print("\n[1] With noise gate...")
t0 = time.time()
tts._qnn_synthesize_to_wav(text, "/tmp/test_ng_on.wav", voice="default_cle", speed=1.0)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")

# Test 2: Without gate
print("\n[2] Without noise gate...")
tts.QNN_NOISE_GATE_ENABLED = False
t0 = time.time()
tts._qnn_synthesize_to_wav(text, "/tmp/test_ng_off.wav", voice="default_cle", speed=1.0)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")

# Restore
tts.QNN_NOISE_GATE_ENABLED = True

# Analysis
from scipy.io import wavfile
def analyze(path, label):
    sr, a = wavfile.read(path)
    af = a.astype(float)
    win = int(sr * 0.01)
    n = len(af) // win
    e = [np.std(af[i*win:(i+1)*win]) for i in range(n)]
    quiet = np.mean([x for x in e if x < np.percentile(e, 10)])
    loud = np.mean([x for x in e if x > np.percentile(e, 90)])
    dr = 20 * np.log10(loud / quiet) if quiet > 0 else 0
    print(f"\n{label}:")
    print(f"  Quiet RMS: {quiet:.0f}, Loud RMS: {loud:.0f}, DR: {dr:.1f}dB")
    print(f"  Total RMS: {af.std():.0f}, Peak: {max(abs(a.min()), abs(a.max()))}")
    return quiet

q_on = analyze("/tmp/test_ng_on.wav", "Gate ON")
q_off = analyze("/tmp/test_ng_off.wav", "Gate OFF")
print(f"\nImprovement: {(1-q_on/q_off)*100:.1f}%")

# Save for comparison
import os
os.system("cp /tmp/test_ng_on.wav /home/fibo/melotts_qnn/test_ng_on.wav")
os.system("cp /tmp/test_ng_off.wav /home/fibo/melotts_qnn/test_ng_off.wav")
print("\nSaved: test_ng_on.wav, test_ng_off.wav")
